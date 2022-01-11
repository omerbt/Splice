import torch
import numpy as np
import random
from data.Dataset import SingleImageDataset
from models.model import Model
from util.losses import LossG
from util.util import get_scheduler, get_optimizer, save_result
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(dataroot):
    with open("conf/default/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    cfg = config

    if dataroot is not None:
        cfg['dataroot'] = dataroot

    # set seed
    seed = cfg['seed']
    if seed == -1:
        seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

    # create dataset, loader
    dataset = SingleImageDataset(cfg)

    # define model
    model = Model(cfg)

    # define loss function
    criterion = LossG(cfg)

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.netG.parameters())

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    with tqdm(range(1, cfg['n_epochs'] + 1)) as tepoch:
        for epoch in tepoch:
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
            log_data["lr"] = lr
            tepoch.set_description(f"Epoch {log_data['epoch']}")
            tepoch.set_postfix(loss=log_data["loss"].item(), lr=log_data["lr"])

            # log current generated entire image
            if epoch % cfg['log_images_freq'] == 0:
                img_A = dataset.get_A().to(device)
                with torch.no_grad():
                    output = model.netG(img_A)
                save_result(output[0], cfg['dataroot'])

            loss_G.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataroot", type=str)
    args = parser.parse_args()
    dataroot = args.dataroot

    train_model(dataroot)
