import time
import torch
from torch.utils.data import DataLoader
from options.base_options import BaseOptions
from data.Dataset import SingleImageDataset
from models.model import Model
from util.util import tensor2im, get_scheduler
import wandb

if __name__ == '__main__':
    # get configuration
    opt = BaseOptions().parse()

    # create dataset, loader
    dataset = SingleImageDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # define model
    model = Model(opt)

    # define optimizer, scheduler
    optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = get_scheduler(optimizer, opt)

    # logging
    wandb.init(project=opt.project, entity='vit-vis', config=opt)

    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataloader):
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            # unpack data from dataset and apply preprocessing
            model.set_input(data)

            # calculate loss functions, get gradients, update network weights
            optimizer.zero_grad()
            model.forward()
            losses = model.compute_G_loss()
            loss_G = losses['total']
            loss_G.backward()
            optimizer.step()

            # log losses
            wandb.log(losses)

            # log current generated entire image to wandb
            if total_iters % opt.log_images_freq == 0:
                model.netG.eval()
                img_A = dataset.get_one_image()
                with torch.no_grad():
                    fake_img = model.netG(img_A)
                image_numpy = tensor2im(fake_img)
                wandb.log({"img": [wandb.Image(image_numpy)]})
                model.netG.train()

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # update learning rates at the end of every epoch.
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        wandb.log({"lr": lr})
