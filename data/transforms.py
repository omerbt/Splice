from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import random


def apply_same_transform(x, y, transform):
    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    seed = random.randint(0, 2 ** 32 - 1)
    set_seed(seed)
    x = transform(x)
    set_seed(seed)
    y = transform(y)
    return x, y


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


def test_same_transform():
    noisearray1 = np.random.rand(250, 250, 3)
    x = Image.fromarray(noisearray1.astype('uint8'))
    noisearray2 = np.copy(x)
    y = Image.fromarray(noisearray2.astype('uint8'))
    transform = dino_structure_transforms
    x, y = apply_same_transform(x, y, transform)
    assert list(x.getdata()) == list(y.getdata())
