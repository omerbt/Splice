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
    wandb.init(project='texture-mapping', entity='vit-vis', config=config)
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
