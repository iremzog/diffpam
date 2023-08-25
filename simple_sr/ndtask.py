import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.append('/home/studio-lab-user/dps_pam')

from simple_sr.unet import Unet
from simple_sr.trainer import Trainer
from simple_sr.dataset import PAMDataset, get_dataloader
from simple_sr.utils import load_yaml
from guided_diffusion.measurements import get_operator
from util.img_utils import mask_generator

class NDTrainer(Trainer):
    
    def __init__(self, args):
        super().__init__()
    
        self.task_config = load_yaml(args.task_config)
        
        measure_config = self.task_config['measurement']
        self.operator = get_operator(device=self.device, **measure_config['operator'])
        
        if measure_config['operator']['name'] == 'inpainting':
            self.mask_gen = mask_generator(**measure_config['mask_opt'])
    
    def build_train_dataloader(self):
        return get_dataloader(self.task_config['data']['train'], self.operator, 
                              mask_gen=self.mask_gen,
                              image_size=self.task_config['measurement']['mask_opt']['image_size'],
                              batch_size=8, train=True, device=self.device)

    def build_val_dataloader(self):
        return get_dataloader(self.task_config['data']['test'], self.operator, 
                              mask_gen=self.mask_gen,
                              image_size=self.task_config['measurement']['mask_opt']['image_size'],
                              batch_size=1, train=False, device=self.device)

    def build_test_dataloader(self):
        return get_dataloader(self.task_config['data']['test'], self.operator, 
                              mask_gen=self.mask_gen,
                              image_size=self.task_config['measurement']['mask_opt']['image_size'],
                              batch_size=1, train=False, device=self.device)

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

    def training_step(self, batch):
        img_lr, img_hr = batch
        res = img_hr-img_lr
        pred_res = self.model(img_lr)
        
        if self.task_config['hyperparams']['loss_type'] == 'l1':
            loss = (res - pred_res).abs().mean()
        elif self.task_config['hyperparams']['loss_type'] == 'l2':
            loss = F.mse_loss(res, pred_res)
        elif self.task_config['hyperparams']['loss_type'] == 'ssim':
            loss = (res - pred_res).abs().mean()
            loss = loss + (1 - self.ssim_loss(res, pred_res))
        else:
            raise NotImplementedError()

        return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss

    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_lr, img_hr = sample
        img_res = self.model(img_lr)
        img_sr = img_res+img_lr

        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b])
            s0 = self.measure.measure(img_lr[b], img_hr[b])
            ret['psnr'] += s['psnr']/img_sr.shape[0]
            ret['ssim'] += s['ssim']/img_sr.shape[0]
            ret['lpips'] += s['lpips']/img_sr.shape[0]
            ret['lr_psnr'] += s0['psnr']/img_lr.shape[0]
            ret['n_samples'] += 1
        return img_sr, ret
    

class NDPAM(NDTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_cls = PAMDataset
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    trainer = NDPAM(args)

    if not args.infer:
        trainer.train()
    else:
        trainer.test()
