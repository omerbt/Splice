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

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path='conf/default', config_name='config')
def train_model(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    # set seed
    if cfg.exp.seed == -1:
        seed = np.random.randint(2 ** 32)
        cfg.exp.seed = seed
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    torch.manual_seed(cfg.exp.seed)

    # create dataset, loader
    dataset = SingleImageDataset(cfg.dataset)

    # define model
    model = Model(cfg.model)

    # define loss function
    criterion = LossG(dataset.B_img, cfg.loss)

    # define optimizer, scheduler
    optimizer = torch.optim.Adam(model.netG.parameters(),
                                 lr=cfg.training.optimizer.lr,
                                 betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2))

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg.training.scheduler.policy,
                              n_epochs=cfg.training.n_epochs,
                              n_epochs_decay=cfg.training.scheduler.n_epochs_decay,
                              lr_decay_iters=cfg.training.scheduler.lr_decay_iters)

    # logging
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    for epoch in range(1, cfg.training.n_epochs + 1):
        inputs = dataset[0]
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, inputs)
        loss_G = losses['total']
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
        if epoch % cfg.logging.log_images_freq == 0:
            model.netG.eval()
            img_A = dataset.get_A().to(device)
            with torch.no_grad():
                output = model.netG(img_A)
            image_numpy = tensor2im(output)
            wandb.log({"img": [wandb.Image(image_numpy)]})
            model.netG.train()


if __name__ == '__main__':
    train_model()
