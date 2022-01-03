from pathlib import Path
from torch.optim import lr_scheduler
import numpy as np
import torch
from torchvision.transforms import ToPILImage


def get_scheduler(optimizer, lr_policy, n_epochs=None, n_epochs_decay=None, lr_decay_iters=None):
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch) / float(n_epochs_decay + 1)
            return max(lr_l, 0)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    elif lr_policy == 'none':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def get_optimizer(cfg, params):
    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=cfg['lr'],
                                     betas=(cfg['optimizer_beta1'], cfg['optimizer_beta2']))
    elif cfg['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=cfg['lr'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'])
    else:
        return NotImplementedError('optimizer [%s] is not implemented', cfg['optimizer'])
    return optimizer


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_result(image_t, dataroot):
    image = ToPILImage()(image_t)
    path = Path(f"{dataroot}/out")
    path.mkdir(exist_ok=True, parents=True)
    image.save(f"{path}/output.png")
