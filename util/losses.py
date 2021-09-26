from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):

    def __init__(self, B_img, cfg):
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

        B = transforms.Compose([transforms.ToTensor(), resize_transform, imagenet_norm])(B_img).unsqueeze(0)
        self.target_global_cls_token = self.extractor.get_feature_from_input(B.to(device))[-1][0, 0, :].detach()

    def forward(self, outputs, inputs):
        losses = {}

        loss_G = 0

        losses['loss_patch_ssim'] = self.calculate_local_ssim_loss(outputs['x_local'], inputs['A_local'])
        losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
        losses['loss_global_cls'] = self.calculate_global_cls_loss(outputs['x_global'])
        # losses['loss_local_cls'] = self.calculate_cls_loss(outputs['x_local'])
        losses['loss_idt_B'] = F.l1_loss(outputs['y_local'], inputs['B_local'])

        loss_G += losses['loss_patch_ssim'] * self.cfg['lambda_patch_ssim']
        loss_G += losses['loss_global_ssim'] * self.cfg['lambda_global_ssim']
        loss_G += losses['loss_global_cls'] * self.cfg['lambda_global_cls']
        # loss_G += losses['loss_local_cls'] * self.cfg.lambda_local_cls
        loss_G += losses['loss_idt_B'] * self.cfg['lambda_identity']

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
