from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):

    def __init__(self, A_img, B_img, cfg):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([resize_transform,
                                                    transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
                                                    imagenet_norm
                                                    ])
        self.local_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
            imagenet_norm
        ])

        A = transforms.Compose([transforms.ToTensor(), resize_transform, imagenet_norm])(A_img).unsqueeze(0)
        B = transforms.Compose([transforms.ToTensor(), resize_transform, imagenet_norm])(B_img).unsqueeze(0)
        self.target_global_ssim = self.extractor.get_keys_self_sim_from_input(A.to(device), layer_num=11).detach()
        self.target_global_cls_token = self.extractor.get_feature_from_input(B.to(device))[-1][0, 0, :].detach()
        self.B = B.to(device)

    def forward(self, outputs, inputs):
        losses = {}

        loss_G = 0

        losses['loss_ssim'] = self.calculate_ssim_loss(outputs['x'])
        losses['loss_cls'] = self.calculate_global_cls_loss(outputs['x'], self.cfg['lambda_cls'])
        losses['loss_patch_ssim'] = self.calculate_local_ssim_loss(outputs['x_local'], inputs['A_local'])
        losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
        losses['loss_global_cls_cropped'] = self.calculate_global_cls_cropped_loss(outputs['x_global'], inputs['B_global'])
        losses['loss_global_cls'] = self.calculate_global_cls_loss(outputs['x_global'], self.cfg['lambda_global_cls'])
        losses['loss_local_cls'] = self.calculate_local_cls_loss(outputs['x_local'])
        losses['loss_crops_cls'] = self.calculate_crops_cls_loss(outputs['x'])
        losses['loss_idt_B'] = F.l1_loss(outputs['y_local'], inputs['B_local'])

        loss_G += losses['loss_ssim'] * self.cfg['lambda_ssim']
        loss_G += losses['loss_cls'] * self.cfg['lambda_cls']
        loss_G += losses['loss_patch_ssim'] * self.cfg['lambda_patch_ssim']
        loss_G += losses['loss_global_ssim'] * self.cfg['lambda_global_ssim']
        loss_G += losses['loss_global_cls_cropped'] * self.cfg['lambda_global_cls_cropped']
        loss_G += losses['loss_global_cls'] * self.cfg['lambda_global_cls']
        loss_G += losses['loss_local_cls'] * self.cfg['lambda_local_cls']
        loss_G += losses['loss_idt_B'] * self.cfg['lambda_identity']

        loss_G += losses['loss_crops_cls'] * self.cfg['lambda_crops_cls']

        losses['loss'] = loss_G
        return losses

    def calculate_ssim_loss(self, output):
        if self.cfg['lambda_ssim'] == 0:
            return 0
        x = self.global_transform(output)
        ssim = self.extractor.get_keys_self_sim_from_input(x, layer_num=11)
        loss = F.mse_loss(ssim, self.target_global_ssim)
        return loss

    def calculate_local_ssim_loss(self, outputs, inputs):
        if self.cfg['lambda_patch_ssim'] == 0:
            return 0
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.local_transform(a)
            b = self.local_transform(b)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_global_ssim_loss(self, outputs, inputs):
        if self.cfg['lambda_global_ssim'] == 0:
            return 0
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_global_cls_cropped_loss(self, outputs, inputs):
        if self.cfg['lambda_global_cls_cropped'] == 0:
            return 0
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            target_cls_token = self.extractor.get_feature_from_input(a.unsqueeze(0))[-1][0, 0, :]
            cls_token = self.extractor.get_feature_from_input(b.unsqueeze(0))[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_cls_loss(self, outputs, loss_lambda):
        if loss_lambda == 0:
            return 0
        loss = 0.0
        for a in outputs:  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, self.target_global_cls_token)
        return loss

    def calculate_local_cls_loss(self, outputs):
        if self.cfg['lambda_local_cls'] == 0:
            return 0
        loss = 0.0
        for a in outputs:
            a = self.local_transform(a).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, self.target_global_cls_token)
        return loss

    def calculate_crops_cls_loss(self, output):
        if self.cfg['lambda_crops_cls'] == 0:
            return 0
        x = self.global_transform(output)
        loss = 0.0
        n_crops = 4
        for _ in range(n_crops):
            a = transforms.RandomCrop(self.cfg['local_crops_crop_size'])(x)
            b = transforms.RandomCrop(self.cfg['local_crops_crop_size'])(self.B)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_local_ssim(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.local_transform(a)
            b = self.local_transform(b)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss
