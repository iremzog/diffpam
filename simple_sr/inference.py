import torch
import sys

sys.path.append('/home/studio-lab-user/dps_pam')
from simple_sr.unet import Unet


class SimpleSR():
    def __init__(self, model_path=None, device=None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Unet(64, out_dim=1, dim_mults=[1, 2, 2, 4]).to(device)
        
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt['state_dict']['model'])
        
        self.model = model
        self.device = device
        
    def inference(self, lr_img):
        
        if lr_img.size[1] == 3:
            lr_img = lr_img[:, 0:1, ...]
            
        simple_res = self.model(lr_img)
        simple_sr = simple_res + lr_img
        simple_sr = simple_sr.repeat(1, lr_img.size[1], 1, 1)
        
        return simple_sr