from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F

from data.transforms import Global_crops, Local_crops
from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):

    def __init__(self, B_img, cfg):
        super().__init__()

        self.cfg = cfg
        self.B_img = B_img
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
                                                    imagenet_norm
                                                    ])
        self.local_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
            imagenet_norm
        ])

        self.register_buffer("step", torch.zeros(1))
        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_local_cls=cfg['lambda_local_cls'],
            lambda_local_ssim=0,
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_local_identity=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self):
        if self.step == self.cfg['cls_warmup']:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_local_ssim'] = self.cfg['lambda_local_ssim']
            self.lambdas['lambda_local_identity'] = self.cfg['lambda_local_identity']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

    def update_step(self):
        self.step += 1
        self.update_lambda_config()

    def forward(self, outputs, inputs):
        self.update_step()
        losses = {'step': self.step}
        loss_G = 0

        if self.lambdas['lambda_local_ssim'] > 0:
            losses['loss_local_ssim'] = self.calculate_local_ssim_loss(outputs['x_local'], inputs['A_local'])
            loss_G += losses['loss_local_ssim'] * self.lambdas['lambda_local_ssim']

        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
            loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        if self.lambdas['lambda_local_cls'] > 0:
            losses['loss_local_cls'] = self.calculate_local_crop_cls_loss(outputs['x_local'], inputs['B_local'])
            loss_G += losses['loss_local_cls'] * self.lambdas['lambda_local_cls']

        if self.lambdas['lambda_local_identity'] > 0:
            losses['loss_local_id_B'] = F.l1_loss(outputs['y_local'], inputs['B_local'])
            loss_G += losses['loss_local_id_B'] * self.lambdas['lambda_local_identity']

        if self.lambdas['lambda_global_identity'] > 0:
            losses['loss_global_id_B'] = F.l1_loss(outputs['y_global'], inputs['B_global'])
            loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses

    def calculate_local_ssim_loss(self, outputs, inputs):
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
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = b.unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_local_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.local_transform(a).unsqueeze(0).to(device)
            b = b.unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_cls_loss(self, outputs):
        loss = 0.0
        for a in outputs:  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, self.target_global_cls_token)
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
