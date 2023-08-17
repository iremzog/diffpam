import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from util.tools import clear_color


class Measure:
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net)

    def measure(self, imgSR, imgHR):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        
        lpips = self.lpips(imgSR, imgHR)
        psnr = self.psnr(clear_color(imgSR)*255, clear_color(imgHR)*255)
        ssim = self.ssim(clear_color(imgSR)*255, clear_color(imgHR)*255)
        
        res = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
        return {k: float(v) for k, v in res.items()}

    def lpips(self, imgA, imgB, model=None):
        device = next(self.model.parameters()).device
        tA = imgA.to(device)
        tB = imgB.to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):

        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=255, channel_axis=2)
        # score, diff = ssim(np.squeeze(imgA), np.squeeze(imgB), full=True, multichannel=False, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)