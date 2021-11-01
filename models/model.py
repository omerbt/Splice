import torch
from . import networks


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = networks.define_G(cfg['init_type'], cfg['init_gain'], cfg['upsample_mode']).to(device)
        self.cfg = cfg

    def forward(self, input):
        outputs = {}
        # global patches from structure image
        if self.cfg['lambda_global_cls'] + self.cfg['lambda_global_ssim'] > 0:
            outputs['x_global'] = self.netG(input['A_global'] + 0.1 * torch.rand_like(input['A_global']).to(input['A_global'].device))

        # entire structure image
        if self.cfg['lambda_entire_ssim'] + self.cfg['lambda_entire_cls'] > 0 and input['step'] % self.cfg['entire_A_every'] == 0:
            outputs['x_entire'] = self.netG(input['A'] + 0.1 * torch.rand_like(input['A']).to(input['A'].device))

        # global patches from texture image
        outputs['y_global'] = self.netG(input['B_global'])

        return outputs
