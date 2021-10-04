from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch


class Local_crops_Legacy(nn.Module):
    def __init__(self, n_crops, crop_size, scale_max, last_transform):
        super().__init__()
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.scale_max = scale_max
        self.patch_transform = transforms.Compose([transforms.RandomCrop(size=crop_size),
                                                   last_transform
                                                   ])

    def forward(self, img):
        crops = []
        # first scale the image
        zoom_level = (np.random.uniform(1 / self.scale_max, 1), np.random.uniform(1 / self.scale_max, 1))
        iw, ih = img.size
        zoomw = max(self.crop_size, iw * zoom_level[0])
        zoomh = max(self.crop_size, ih * zoom_level[1])
        img = img.resize((int(round(zoomw)), int(round(zoomh))), Image.BICUBIC)

        # take random crops
        for _ in range(self.n_crops):
            crop = self.patch_transform(img)
            crops.append(crop)

        return torch.stack(crops)


class Local_crops(nn.Module):
    def __init__(self, n_crops, max_cover, last_transform, flip=False):
        super().__init__()
        self.n_crops = n_crops
        self.max_cover = max_cover

        transforms_lst = [last_transform]
        if flip:
            transforms_lst += [transforms.RandomHorizontalFlip()]

        self.last_transform = transforms.Compose(transforms_lst)

    def forward(self, img):
        crops = []
        h = img.size[1]
        size = int(round(np.random.uniform(0.15 * h, self.max_cover * h)))
        t = transforms.Compose([transforms.RandomCrop(size), self.last_transform])
        for _ in range(self.n_crops):
            crop = t(img)
            crops.append(crop)
        return torch.stack(crops)


class Global_crops(nn.Module):
    def __init__(self, n_crops, min_cover, last_transform, flip=False):
        super().__init__()
        self.n_crops = n_crops
        self.min_cover = min_cover

        transforms_lst = [last_transform]
        if flip:
            transforms_lst += [transforms.RandomHorizontalFlip()]

        self.last_transform = transforms.Compose(transforms_lst)

    def forward(self, img):
        crops = []
        h = img.size[1]
        size = int(round(np.random.uniform(self.min_cover * h, h)))
        t = transforms.Compose([transforms.RandomCrop(size), self.last_transform])
        for _ in range(self.n_crops):
            crop = t(img)
            crops.append(crop)
        return torch.stack(crops)


dino_structure_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.5
    ),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)
])

dino_texture_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5)
])
