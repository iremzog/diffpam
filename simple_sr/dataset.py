import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

def is_image_path(f):
    if any(['jpeg' in f, 'jpg' in f, 'png' in f]):
        return True
    return False

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(data_path, ratio, image_size=256, 
                   batch_size=32, train=True, device=None):

    if train:
        transforms = T.Compose([
                                    T.RandomCrop(image_size),
                                    T.RandomHorizontalFlip(p=0.3),
                                    T.RandomVerticalFlip(p=0.3),
                                    T.RandomApply(
                                        [T.RandomRotation(20, interpolation=Image.BILINEAR)],
                                        p=0.3),
                                    T.RandomApply(
                                        [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                                        p=0.5),
                                    T.ToTensor(),
                                    T.Normalize(0.5, 0.5),
                                ])
        
        datalist = [f for f in os.listdir(data_path) if is_image_path(f)]
        multiple_list = [f for f in datalist for i in range(10)]  # 10 random crops in an image
        dataset = PAMDataset(ratio, multiple_list, data_path, transforms, device=device)
        
    else:
        transforms = T.Compose([
                                    T.CenterCrop(image_size),
                                    T.ToTensor(),
                                    T.Normalize(0.5, 0.5),
                                ])

        datalist = [f for f in os.listdir(data_path) if is_image_path(f)]
        dataset = PAMDataset(ratio, datalist, data_path, transforms, device=device)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=train)

    return dataloader


def get_lr_image(hr_image, rx, ry, mode='bilinear'):

    dim_x = hr_image.shape[2]
    dim_y = hr_image.shape[3]

    surplusx = int(np.ceil(dim_x/rx)*rx-dim_x)
    surplusx = -surplusx if surplusx != 0 else None
    surplusy = int(np.ceil(dim_y/ry)*ry-dim_y)
    surplusy = -surplusy if surplusy != 0 else None

    uu_input = torch.nn.functional.interpolate(hr_image[:, :, ::rx, ::ry],
                                               scale_factor=(rx, ry),
                                               mode=mode)[:, :, :surplusx, :surplusy]

    return uu_input


class PAMDataset(torch.utils.data.Dataset):
    def __init__(self, ratio, img_list, img_dir, transforms, device=None):

        self.ratio = ratio
        self.img_dir = img_dir
        self.img_list = img_list
        self.transforms = transforms
        self.device = device
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        hr_image = Image.open(img_path)
        hr_image = self.transforms(hr_image).unsqueeze(0).to(self.device)
        lr_image = get_lr_image(hr_image, self.ratio[0], self.ratio[1])

        return lr_image[0], hr_image[0]