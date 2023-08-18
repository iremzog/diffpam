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


def get_dataloader(data_path, operator, mask_gen=None,
                   image_size=256, batch_size=32, train=True):

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

        # multiple_list = [f for f in datalist for i in range(10)]  # 10 random crops in an image
    else:
        transforms = T.Compose([
                                    T.CenterCrop(image_size),
                                    T.ToTensor(),
                                    T.Normalize(0.5, 0.5),
                                ])

    datalist = [f for f in os.listdir(data_path) if is_image_path(f)]

    dataset = PAMDataset(operator, datalist, data_path, transforms, mask_gen=mask_gen)
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
    def __init__(self, operator, img_list, img_dir, transforms, mask_gen=None):

        self.operator = operator
        self.img_dir = img_dir
        self.transforms = transforms
        self.mask_gen = mask_gen
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        hr_image = self.transforms(Image.open(img_path)).unsqueeze(0)
        
        if self.mask_gen is not None:
            mask = self.mask_gen(hr_image)
            lr_image = self.operator.forward(hr_image, mask=mask)
        else:
            lr_image = self.operator.forward(hr_image)

        return lr_image[0], hr_image[0]
