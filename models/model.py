import torch
from . import networks


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = networks.define_G(cfg['init_type'], cfg['init_gain']).to(device)

    def forward(self, input):
        x_entire = self.netG(input['A'])  # entire structure image
        x_local = self.netG(input['A_local'])  # local patches from structure image
        y_local = self.netG(input['B_local'])  # local patches from texture image
        x_global = self.netG(input['A_global'])  # global patches from structure image
        y_global = self.netG(input['B_global'])  # global patches from texture image

        return {'x_entire': x_entire, 'x_local': x_local, 'y_local': y_local, 'x_global': x_global,
                'y_global': y_global}
