import sys

sys.path.insert(0, "/home/labs/waic/narekt/projects/texture-mapping")


import logging
# import hydra
# from hydra import utils
import torch
import wandb
import numpy as np
import random
import os
from data.Dataset import SingleImageDataset
from models.model import Model
from util.losses import LossG
from util.util import tensor2im, get_scheduler, get_optimizer
# from omegaconf import OmegaConf
import yaml

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_cfg(cfg):
    if (
            (cfg['scheduler_policy'] == 'cosine_dino' and (
                    cfg['base_lr'] < 0 or cfg['final_lr'] < 0 or cfg['warmup_epochs_prop'] < 0 or cfg['start_warmup_lr'] < 0))
            or
            (cfg['scheduler_policy'] != 'cosine_dino' and (
            cfg['base_lr'] > 0 or cfg['final_lr'] > 0 or cfg['warmup_epochs_prop'] > 0 or cfg['start_warmup_lr'] > 0))
    ):
        return False

    return True


def make_scheduler_step(optimizer, scheduler, iter, cfg):
    if cfg['scheduler_policy'] == 'cosine_dino':
        optimizer.param_groups[0]['lr'] = scheduler[iter]
    else:
        scheduler.step()


# @hydra.main(config_path='conf/default', config_name='config')
def train_model():
    with open("conf/default/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    wandb.init(project='semantic-texture-transfer', entity='vit-vis', config=config)
    cfg = wandb.config

    is_cfg_valid = validate_cfg(cfg)
    if not is_cfg_valid:
        print("invalid config, aborting the run...")
        # run = api.run(f'vit-vis/semantic-texture-transfer/{wandb.run.id}')
        # run.delete()
        return

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
    criterion = LossG(dataset.A_img, dataset.B_img, cfg)

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.netG.parameters())

    scheduler = get_scheduler(optimizer,
                              cfg,
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

        # log gradient total norm
        grad_norm = 0.0
        for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        wandb.log({'grad_norm': grad_norm})

        # log losses
        wandb.log(losses)

        # update learning rate
        make_scheduler_step(optimizer, scheduler, epoch, cfg)
        lr = optimizer.param_groups[0]['lr']
        log.info('learning rate = %.7f' % lr)
        wandb.log({"lr": lr, "epoch": epoch})

        # log current generated entire image to wandb
        if epoch % cfg['log_images_freq'] == 0:
            print(f'epoch: {epoch}')
            model.netG.eval()
            img_A = dataset.get_A().to(device)
            with torch.no_grad():
                output = model.netG(img_A)
            image_numpy = tensor2im(output)
            wandb.log({"img": [wandb.Image(image_numpy)]})
            model.netG.train()


if __name__ == '__main__':
    train_model()
