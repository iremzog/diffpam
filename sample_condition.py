from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
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
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input', type=str, default='noise')
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'inverse']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.CenterCrop(model_config['image_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    results = []
    
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
            
            inverse_measurement = operator.transpose(y_n)
            proxy_ref = inverse_measurement
            
            if measure_config['mask_opt']['mask_type'] == 'periodic':
                rx = measure_config['mask_opt']['mask_ratio'][0]
                ry = measure_config['mask_opt']['mask_ratio'][1]
                proxy_ref = get_lr_image(y_n, rx, ry, mode='bilinear')

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
            # Creating proxy X ref
            inverse_measurement = operator.transpose(y_n)
            proxy_ref = inverse_measurement
        
        if args.input == 'noise':
            approximate_t = None
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        elif args.input == 'interpolation':
            approximate_t = 500
            x_start = sampler.q_sample(proxy_ref, t=approximate_t)
        elif args.input == 'simple_sr':
            sr_model = SimpleSR(model_path=task_config['simple_sr']['model_path'], device=device)
            simple_sr = sr_model.inference(proxy_ref)
        
            # Sampling
            approximate_t = 200
            x_start = sampler.q_sample(simple_sr, t=approximate_t)
        else:
            print('Please enter valid input choice!')
            break
        
        # x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sr_img = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path, t=approximate_t)
        
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'inverse', f'interpolation_{fname}'), clear_color(proxy_ref))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sr_img))
        
        if args.input == 'simple_sr':
            plt.imsave(os.path.join(out_path, 'inverse', f'unet_{fname}'), clear_color(simple_sr))
        
            # Results
            names = ['DiffPam', 'UNet', 'Interpolation', 'Input']
            measure = Measure()

            for j, img in enumerate([sr_img, simple_sr, proxy_ref, inverse_measurement]):
                s = measure.measure(img, ref_img)
                print(f"{names[j]}:: PSNR: {s['psnr']}, SSIM: {s['ssim']}, LPIPS: {s['lpips']}")
                s['method'] = names[j]
                results.append(s)
        else:
            # Results
            names = ['DiffPam', 'Interpolation', 'Input']
            measure = Measure()

            for j, img in enumerate([sr_img, proxy_ref, inverse_measurement]):
                s = measure.measure(img, ref_img)
                print(f"{names[j]}:: PSNR: {s['psnr']}, SSIM: {s['ssim']}, LPIPS: {s['lpips']}")
                s['method'] = names[j]
                results.append(s)
    
    with open(os.path.join(out_path, 'results.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
