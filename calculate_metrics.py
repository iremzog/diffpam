import os
import argparse
import json
from glob import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
from util.measure import Measure

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_config', type=str)
    # parser.add_argument('--diffusion_config', type=str)
    # parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--ref_dir', type=str)
    args = parser.parse_args()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    # Do Inference
    results = []
    root = args.input_dir
    ref_dir = args.ref_dir
    fpaths = sorted(glob(root + '/**/*.jpeg', recursive=True)+glob(root + '/**/*.png', recursive=True))
    gt_fpaths = sorted(glob(ref_dir + '/**/*.jpeg', recursive=True)+glob(ref_dir + '/**/*.png', recursive=True))
    
    measure = Measure()
    ssim = StructuralSimilarityIndexMeasure(data_range=None)
    mean_absolute_error = MeanAbsoluteError()
    
    for i, fpath in enumerate(fpaths):
        
        try:
            image_number = int(fpath.split("/")[-1].split(".")[0])
        except ValueError:
            image_number = int(fpath.split("/")[-1].split(".")[0].split("_")[-1])
            
        print(image_number, int(gt_fpaths[i].split("/")[-1].split(".")[0]))
        assert image_number == int(gt_fpaths[i].split("/")[-1].split(".")[0])
        
        ref_img = Image.open(gt_fpaths[i]).convert('RGB')
        img = Image.open(fpath).convert('RGB')
        
        img = transform(img).to(device)
        ref_img = transform(ref_img).to(device)
    
        s = measure.measure(img, ref_img)
        
        s['img'] = image_number
        pytorch_ssim = ssim(img.unsqueeze(0), ref_img.unsqueeze(0)).detach().cpu().numpy()
        mae = mean_absolute_error(img.unsqueeze(0), ref_img.unsqueeze(0)).detach().cpu().numpy()
        s['ssim_pytorch'] = float(pytorch_ssim)
        s['mae'] = float(mae)
        print(f"{i}:: PSNR: {s['psnr']}, SSIM: {s['ssim']}, PSSIM: {s['ssim_pytorch']}, LPIPS: {s['lpips']}, MAE: {s['mae']}")
        results.append(s)

    with open(os.path.join(root, 'results.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
