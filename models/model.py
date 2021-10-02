import torch
from . import networks


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = networks.define_G(cfg['init_type'], cfg['init_gain']).to(device)
        self.cfg = cfg

    def forward(self, input):
        outputs = {}
        # global patches from structure image
        if self.cfg['lambda_global_cls'] + self.cfg['lambda_global_ssim'] > 0:
            outputs['x_global'] = self.netG(input['A_global'])

        # entire structure image
        if self.cfg['lambda_entire_ssim'] > 0:
            outputs['x_entire'] = self.netG(input['A'])

        # local patches from structure image
        if self.cfg['lambda_local_ssim'] > 0:
            outputs['x_local'] = self.netG(input['A_local'])

        # local patches from texture image
        if self.cfg['lambda_local_identity'] > 0:
            outputs['B_local'] = self.netG(input['B_local'])

        # global patches from texture image
        outputs['y_global'] = self.netG(input['B_global'])

        return outputs
