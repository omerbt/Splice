import torch
from . import networks


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = networks.define_G(cfg['init_type'], cfg['init_gain'], cfg['upsample_mode'],
                                      cfg['downsample_mode']).to(device)
        self.cfg = cfg
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, input):
        outputs = {}
        # global patches from structure image
        if self.cfg['lambda_global_cls'] + self.cfg['lambda_global_ssim'] > 0:
            outputs['x_global'] = self.netG(
                input['A_global'] + self.noise.expand_dims(*input['A_global'].shape).normal_() * 10)

        # entire structure image
        if self.cfg['lambda_entire_ssim'] + self.cfg['lambda_entire_cls'] > 0 and input['step'] % self.cfg[
            'entire_A_every'] == 0:
            outputs['x_entire'] = self.netG(input['A'] +
                                            self.noise.expand_dims(*input['A'].shape).normal_() * 10)

        # global patches from texture image
        outputs['y_global'] = self.netG(input['B_global'])

        return outputs
