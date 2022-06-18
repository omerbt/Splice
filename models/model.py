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
        if self.cfg['lambda_entire_ssim'] > 0 and input['step'] % self.cfg['entire_A_every'] == 0:
            outputs['x_entire'] = self.netG(input['A'])

        # global patches from texture image
        outputs['y_global'] = self.netG(input['B_global'])

        return outputs
