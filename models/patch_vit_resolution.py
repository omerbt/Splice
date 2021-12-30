import types
import torch.nn.modules.utils as nn_utils
import torch
from torch import nn
import math


def _fix_positional_encoding(patch_size, stride_hw):

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f'got wrong grid size for {h}x{w} with patch_size {patch_size} and stride {stride_hw} got {h0}x{w0}={h0*w0} expecting {npatch}'
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic', 
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding    


def patch_vit_resolution(vit_model, stride):
    patch_size = vit_model.patch_embed.patch_size
    if stride == patch_size:
        # nothing to do
        return vit_model
    
    stride = nn_utils._pair(stride)
    
    assert all([(patch_size // s_) * s_ == patch_size for s_ in stride]), f'stride {stride} should divide patch_size {patch_size}'
    
    # fix the stride
    vit_model.patch_embed.proj.stride = stride

    # fix the positional encoding code
    vit_model.interpolate_pos_encoding = types.MethodType(_fix_positional_encoding(patch_size, stride), vit_model)

    return vit_model

