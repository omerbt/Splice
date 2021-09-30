from torch.optim import lr_scheduler
import numpy as np
import torch


def get_scheduler(optimizer, cfg, n_epochs=None, n_epochs_decay=None, lr_decay_iters=None):
    if cfg['scheduler_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch) / float(n_epochs_decay + 1)
            return max(lr_l, 0)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg['scheduler_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif cfg['scheduler_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif cfg['scheduler_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    elif cfg['scheduler_policy'] == 'cosine_dino':
        scheduler = cosine_scheduler(cfg['base_lr'], cfg['final_lr'], cfg['n_epochs'], cfg['warmup_epochs_prop'], cfg['start_warmup_lr'])
    elif cfg['scheduler_policy'] == 'none':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', cfg['scheduler_policy'])
    return scheduler


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs_prop=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs_prop * epochs)
    if warmup_epochs_prop > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule
