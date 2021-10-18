import logging
import torch
import wandb
import numpy as np
import random
import os
from data.Dataset import SingleImageDataset
from models.model import Model
from util.losses import LossG
from util.util import tensor2im, get_scheduler, get_optimizer
import yaml

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model():
    with open("conf/default/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    wandb.init(project='afhq_cross_narek', entity='vit-vis', config=config)
    cfg = wandb.config

    # set seed
    seed = cfg['seed']
    if seed == -1:
        seed = np.random.randint(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

    # create dataset, loader
    dataset = SingleImageDataset(cfg)

    # define model
    model = Model(cfg)

    # define loss function
    criterion = LossG(dataset.B_img, cfg)

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.netG.parameters())

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    for epoch in range(1, cfg['n_epochs'] + 1):
        inputs = dataset[0]
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, inputs)
        loss_G = losses['loss']
        log_data = losses
        log_data['epoch'] = epoch

        # update learning rate
        lr = optimizer.param_groups[0]['lr']
        log.info('learning rate = %.7f' % lr)
        log_data["lr"] = lr

        # log current generated entire image to wandb
        if epoch % cfg['log_images_freq'] == 0:
            img_A = dataset.get_A().to(device)
            with torch.no_grad():
                output = model.netG(img_A)
            image_numpy_output = tensor2im(output)
            log_data["img_output"] = [wandb.Image(image_numpy_output)]
            if cfg['log_crops']:
                structure_crops = inputs['A_global']
                texture_crops = inputs['B_global']
                output_crops = outputs['x_global']
                output_texture_crops = outputs['y_global']
                image_structure_crop_numpy_output = tensor2im(output_crops)
                image_texture_crop_numpy_output = tensor2im(output_texture_crops)
                log_data["structure_crop_input"] = [wandb.Image(structure_crops)]
                log_data["texture_crop_input"] = [wandb.Image(texture_crops)]
                log_data["structure_crop_output"] = [wandb.Image(image_structure_crop_numpy_output)]
                log_data["texture_crop_output"] = [wandb.Image(image_texture_crop_numpy_output)]

        loss_G.backward()
        optimizer.step()
        scheduler.step()
        wandb.log(log_data)


if __name__ == '__main__':
    train_model()
