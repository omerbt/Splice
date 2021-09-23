import sys

sys.path.insert(0, "/home/labs/leeat/omerba/Develop/texture-mapping")


import logging
import hydra
from hydra import utils
import torch
import wandb
import numpy as np
import random
import os
from data.Dataset import SingleImageDataset
from models.model import Model
from util.losses import LossG
from util.util import tensor2im, get_scheduler
from omegaconf import OmegaConf
import yaml

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# @hydra.main(config_path='conf/default', config_name='config')
def train_model():
    config = yaml.load('conf/default/config.yaml')
    wandb.init(project='semantic-texture-transfer', entity='vit-vis', config=config)
    cfg = wandb.config

    # set seed
    if cfg['seed'] == -1:
        seed = np.random.randint(2 ** 32)
        cfg['seed'] = seed
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    log.info(f'seed: {cfg["seed"]}')

    # create dataset, loader
    dataset = SingleImageDataset(cfg)

    # define model
    model = Model(cfg)

    # define loss function
    criterion = LossG(dataset.B_img, cfg)

    # define optimizer, scheduler
    optimizer = torch.optim.Adam(model.netG.parameters(),
                                 lr=cfg['lr'],
                                 betas=(cfg['optimizer_beta1'], cfg['optimizer_beta2']))

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    # logging

    for epoch in range(1, cfg['n_epochs'] + 1):
        inputs = dataset[0]
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, inputs)
        loss_G = losses['loss']
        loss_G.backward()
        optimizer.step()

        # log losses
        wandb.log(losses)

        # update learning rate
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        log.info('learning rate = %.7f' % lr)
        wandb.log({"lr": lr})

        # log current generated entire image to wandb
        if epoch % cfg['log_images_freq'] == 0:
            model.netG.eval()
            img_A = dataset.get_A().to(device)
            with torch.no_grad():
                output = model.netG(img_A)
            image_numpy = tensor2im(output)
            wandb.log({"img": [wandb.Image(image_numpy)]})
            model.netG.train()


if __name__ == '__main__':
    train_model()
