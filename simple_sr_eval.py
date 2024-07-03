import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import json

from util.dataloader import get_dataset, get_dataloader
from util.logger import get_logger
from util.measure import Measure
from simple_sr.dataset import get_lr_image
from simple_sr.inference import SimpleSR


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    task_config = load_yaml(args.task_config)
   
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    
    # Working directory
    out_path = os.path.join(args.save_dir, "rrdb")
    os.makedirs(out_path, exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.CenterCrop(measure_config['mask_opt']['image_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Do Inference
    results = []
    measure = Measure()
    
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        ref_img = ref_img.to(device)
            
        rx = measure_config['mask_opt']['mask_ratio'][0]
        ry = measure_config['mask_opt']['mask_ratio'][1]
        proxy_ref = get_lr_image(ref_img, rx, ry, mode='bilinear')
        
        sr_model = SimpleSR(model_path=task_config['simple_sr']['model_path'], device=device)
        simple_sr = sr_model.inference(proxy_ref)

        # Results
        names = ['RRDB-Net', 'Interpolation']

        for j, img in enumerate([simple_sr, proxy_ref]):
            s = measure.measure(img, ref_img)
            print(f"{names[j]}:: PSNR: {s['psnr']}, SSIM: {s['ssim']}, LPIPS: {s['lpips']}")
            s['method'] = names[j]
            results.append(s)
    
    with open(os.path.join(out_path, 'results.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
