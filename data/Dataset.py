import os.path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path

from data.transforms import Global_crops, Local_crops


class SingleImageDataset(Dataset):
    def __init__(self, cfg):
        # normalization to be applied to every crop after augmentation
        norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            norm_transform
        ])

        self.local_A_patches = Local_crops(n_crops=cfg['local_A_crops_n_crops'],
                                           max_cover=cfg['local_A_crops_max_cover'],
                                           last_transform=self.base_transform)

        self.local_B_patches = Local_crops(n_crops=cfg['local_B_crops_n_crops'],
                                           max_cover=cfg['local_B_crops_max_cover'],
                                           last_transform=self.base_transform)

        self.global_A_patches = Global_crops(n_crops=cfg['global_A_crops_n_crops'],
                                             min_cover=cfg['global_A_crops_min_cover'],
                                             last_transform=self.base_transform)

        self.global_B_patches = Global_crops(n_crops=cfg['global_B_crops_n_crops'],
                                             min_cover=cfg['global_B_crops_min_cover'],
                                             last_transform=self.base_transform)

        # open images
        dir_A = os.path.join(cfg['dataroot'], 'A')
        dir_B = os.path.join(cfg['dataroot'], 'B')
        A_path = os.listdir(dir_A)[0]
        B_path = os.listdir(dir_B)[0]
        self.A_img = Image.open(os.path.join(dir_A, A_path)).convert('RGB')
        self.B_img = Image.open(os.path.join(dir_B, B_path)).convert('RGB')

        if cfg['A_resize'] > 0:
            self.A_img = transforms.Resize(cfg['A_resize'])(self.A_img)

        if cfg['B_resize'] > 0:
            self.B_img = transforms.Resize(cfg['B_resize'])(self.B_img)

        if cfg['direction'] == 'BtoA':
            self.A_img, self.B_img = self.B_img, self.A_img

        print("Image sizes %s and %s" % (str(self.A_img.size), str(self.B_img.size)))

    def get_A(self):
        return self.base_transform(self.A_img).unsqueeze(0)

    def __getitem__(self, index):
        A = self.get_A()
        A_global = self.global_A_patches(self.A_img)
        B_global = self.global_B_patches(self.B_img)
        A_local = self.local_A_patches(self.A_img)
        B_local = self.local_B_patches(self.B_img)

        return {'A': A, 'A_global': A_global, 'B_global': B_global, 'A_local': A_local, 'B_local': B_local}

    def __len__(self):
        return 1
