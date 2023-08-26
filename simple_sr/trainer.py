import os
import subprocess
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from util.measure import Measure
from simple_sr.utils import plot_img, move_to_device, load_checkpoint, save_checkpoint, tensors_to_scalars


class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir='simple_sr/results/',
                                             name='tb_logs')
        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.work_dir = 'simple_sr/results/'
        self.first_val = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        EFS_PATH_LOG_DIR = "/".join(log_dir.strip("/").split('/')[1:-1])
        print(EFS_PATH_LOG_DIR)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self):
        raise NotImplementedError
        
    def build_val_dataloader(self):
        raise NotImplementedError

    def build_test_dataloader(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(model, optimizer,
                                                           self.work_dir,
                                                           self.device)
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        dataloader = self.build_train_dataloader()

        initial_step = self.global_step
        train_pbar = tqdm(range(initial_step, 5001, len(dataloader)))

        for step in train_pbar:
            for batch in dataloader:
                if training_step % 1000 == 0:
                    with torch.no_grad():
                        model.eval()
                        self.validate(training_step)
                    save_checkpoint(model, optimizer, self.work_dir, training_step, 20)
                model.train()
                batch = move_to_device(batch, self.device)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()
                scheduler.step()
                training_step += 1
                self.global_step = training_step
                if training_step % 100 == 0:
                    self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))

    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        mean_metrics = {k: 0 for k in self.metric_keys}
        for batch_idx, batch in pbar:
            if self.first_val and batch_idx > 10: 
                break
            batch = move_to_device(batch, self.device)
            img, ret = self.sample_and_test(batch)
            img_lr, img_hr = batch
            if img is not None:
                self.logger.add_image(f'Pred_{batch_idx}', plot_img(img[0]), self.global_step)
                if self.global_step <= 1000:
                    self.logger.add_image(f'HR_{batch_idx}', plot_img(img_hr[0]), self.global_step)
                    self.logger.add_image(f'LR_{batch_idx}', plot_img(img_lr[0]), self.global_step)
            metrics = {k: np.mean(ret[k]) for k in self.metric_keys}
            mean_metrics.update({k: metrics[k]/len(val_dataloader)+mean_metrics[k] for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))

        if not self.first_val:
            self.log_metrics({f'val/{k}': v for k, v in mean_metrics.items()}, training_step)
            print('Val results:', metrics)
        else:
            print('Sanity val results:', mean_metrics)
        self.first_val = False

    def test(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        load_checkpoint(model, optimizer, self.work_dir, self.device)
        optimizer = None

        self.results = {k: 0 for k in self.metric_keys}
        self.n_samples = 0
        self.gen_dir = "simple_sr/generation/"

        subprocess.check_call(f'rm -rf {self.gen_dir}', shell=True)
        os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False

        # if hasattr(self.model.denoise_fn, 'make_generation_fast_'):
        #     self.model.denoise_fn.make_generation_fast_()

        with torch.no_grad():
            model.eval()
            test_dataloader = self.build_test_dataloader()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for batch_idx, batch in pbar:
                move_to_device(batch, self.device)
                gen_dir = self.gen_dir
                img_lr, img_hr = batch
                item_name = batch_idx

                img_sr, ret = self.sample_and_test(batch)
                img_lr, img_hr = batch

                if img_sr is not None:
                    metrics = list(self.metric_keys)
                    for k in metrics:
                        self.results[k] += ret[k]*ret['n_samples']
                    self.n_samples += ret['n_samples']

                    img_sr = self.tensor2img(img_sr)
                    img_hr = self.tensor2img(img_hr)
                    img_lr = self.tensor2img(img_lr)

                    for k, (sr, hr, lr) in enumerate(zip(img_sr, img_hr, img_lr)):
                        sr = Image.fromarray(sr.squeeze())
                        hr = Image.fromarray(hr.squeeze())
                        lr = Image.fromarray(lr.squeeze())
                        sr.save(f"{gen_dir}/outputs/{item_name}_{k}[SR].png")
                        hr.save(f"{gen_dir}/outputs/{item_name}_{k}[HR].png")
                        lr.save(f"{gen_dir}/outputs/{item_name}_{k}[LR].png")

            print({k: round(self.results[k] / self.n_samples, 3) for k in metrics}, 'total:', self.n_samples)

    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

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
