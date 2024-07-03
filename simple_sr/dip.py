import sys
sys.path.append('/home/studio-lab-user/diffpam/')

import os
import subprocess
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from simple_sr.unet import Unet
from simple_sr.utils import load_yaml
from util.measure import Measure
from simple_sr.utils import move_to_device, tensors_to_scalars
from torchvision import transforms as T


class DIPTrainer():
    
    def __init__(self, args):
        super().__init__()
    
        self.task_config = load_yaml(args.task_config)
        self.measure = Measure()
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.work_dir = 'simple_sr/results'
        self.first_val = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_size = self.task_config['hyperparams']['hidden_size']
        dim_mults = self.task_config['hyperparams']['dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        self.model = Unet(hidden_size, out_dim=1, dim_mults=dim_mults).to(device)
        
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.task_config['hyperparams']['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 2000, 0.5)

    def training_step(self, noise, img_hr):
        pred_res = self.model(noise)
        
        mask_ratio = self.task_config['measurement']['mask_opt']['mask_ratio']
        rx, ry = mask_ratio
        mask = torch.zeros_like(img_hr, device=img_hr.device)
        mask[..., ::rx, ::ry] = 1
        
        if self.task_config['hyperparams']['loss_type'] == 'l1':
            loss = ((img_hr - pred_res)*mask).abs().mean()
        elif self.task_config['hyperparams']['loss_type'] == 'l2':
            loss = F.mse_loss(img_hr*mask, pred_res*mask)
        else:
            raise NotImplementedError()

        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss
    
    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = 0
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        
        transforms = T.Compose([
                            T.CenterCrop(self.task_config['measurement']['mask_opt']['image_size']),
                            T.ToTensor(),
                            T.Normalize(0.5, 0.5),
                        ])
        
        hr_image = Image.open(self.task_config['data']['train'])
        hr_image = transforms(hr_image).unsqueeze(0).to(self.device)

        noise = (torch.rand(*hr_image.size())*0.1).to(self.device)
        initial_step = self.global_step
        train_pbar = tqdm(range(initial_step, 5000,))

        for step in train_pbar:
                    
            model.train()
            optimizer.zero_grad()
            noise_reg = torch.normal(0, 0.05, hr_image.size())
            losses, total_loss = self.training_step(noise+noise_reg, hr_image)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_pbar.set_postfix(**tensors_to_scalars(losses))
            
            if training_step % 50 == 0:
                with torch.no_grad():
                    model.eval()
                    self.test(model, noise, hr_image)
                    
            training_step += 1

    def test(self, model, noise, hr_image):
        self.results = {k: 0 for k in self.metric_keys}
        self.n_samples = 0
        self.gen_dir = "simple_sr/generation/"

        subprocess.check_call(f'rm -rf {self.gen_dir}', shell=True)
        os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False

        with torch.no_grad():
            model.eval()
            gen_dir = self.gen_dir

            img_sr, ret = self.sample_and_test(model, noise, hr_image)

            if img_sr is not None:
                metrics = list(self.metric_keys)
                for k in metrics:
                    self.results[k] += ret[k]*ret['n_samples']
                self.n_samples += ret['n_samples']

                img_sr = self.tensor2img(img_sr)
                img_hr = self.tensor2img(hr_image)

                for k, (sr, hr) in enumerate(zip(img_sr, img_hr)):
                    sr = Image.fromarray(sr.squeeze())
                    hr = Image.fromarray(hr.squeeze())
                    sr.save(f"{gen_dir}/outputs/{k}[SR].png")
                    hr.save(f"{gen_dir}/outputs/{k}[HR].png")

            print({k: round(self.results[k] / self.n_samples, 3) for k in metrics}, 'total:', self.n_samples)

    def sample_and_test(self, model, noise, hr_image):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_sr = model(noise)

        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], hr_image[b])
            ret['psnr'] += s['psnr']/img_sr.shape[0]
            ret['ssim'] += s['ssim']/img_sr.shape[0]
            ret['lpips'] += s['lpips']/img_sr.shape[0]
            ret['n_samples'] += 1
        return img_sr, ret

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    trainer = DIPTrainer(args)
    trainer.train()