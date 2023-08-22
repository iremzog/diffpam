import argparse
import yaml
import torch

from ..guided_diffusion.measurements import get_operator
from ..util.img_utils import mask_generator
from ..util.measure import Measure
from .model import SimpleSRUnet
from .dataloader import get_dataloader, get_lr_image


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    
    # Load configurations
    task_config = load_yaml(args.task_config)

    # Load model
    model = SimpleSRUnet(dim=32, out_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, 0.5)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])
    
    # Prepare dataloader
    trainloader = get_dataloader(task_config['data']['train'], operator, 
                                 mask_gen=mask_gen,
                                 image_size=task_config['measurement']['mask_opt']['image_size'], 
                                 batch_size=32, train=True)
    validloader = get_dataloader(task_config['data']['test'], operator, 
                                 mask_gen=mask_gen,
                                 image_size=task_config['measurement']['mask_opt']['image_size'], 
                                 batch_size=4, train=False)
    measure = Measure()
    
    for e in range(task_config['epochs']):
        
        running_loss = 0.
        
        model.train()
        for j, data in enumerate(trainloader):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        epoch_loss = running_loss/len(trainloader)
        print(f'Epoch {e+1}/{task_config["epochs"]}, loss: {epoch_loss}')
        
        if (e+1)%10 == 0:
            model.eval()
            
            for j, data in enumerate(validloader):
                
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                upsampled = get_lr_image(labels, rx=5, ry=1, mode='bilinear')
                
                s = measure.measure(outputs, labels)
                
                print('-'*100)
                print(f"SR:: Loss: {loss} PSNR: {s['psnr']}, SSIM: {s['ssim']}, LPIPS: {s['lpips']}")
                s = measure.measure(upsampled, labels)
                print(f"LR:: PSNR: {s['psnr']}, SSIM: {s['ssim']}, LPIPS: {s['lpips']}")
            
            torch.save(model.state_dict(), f'models/checkpoint_{e}.pt')


if __name__ == '__main__':
    main()