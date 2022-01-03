import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch


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
        t = transforms.Compose([transforms.RandomCrop(min(size, img.size[0])), self.last_transform])
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
