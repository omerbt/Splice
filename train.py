import time
import torch
from options.train_options import TrainOptions
from data import singleimage_dataset
from torch.utils.data import DataLoader
from models.model import Model
from util.util import tensor2im
import wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = singleimage_dataset(opt)
    loader = DataLoader(dataset, batch_size=opt.batch_size,
                        shuffle=not opt.serial_batches,
                        num_workers=int(opt.num_threads),
                        drop_last=True if opt.isTrain else False)

    model = Model(opt)

    wandb.init(project=opt.project, entity='vit-vis', config=opt)

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        dataset.set_epoch(epoch)

        for i, data in enumerate(loader):  # inner loop within one epoch
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.log_images_freq == 0:  # log current generated entire image to wandb
                model.netG.eval()
                img_A = dataset.dataset.get_one_image()
                with torch.no_grad():
                    fake_img = model.netG(img_A)
                image_numpy = tensor2im(fake_img)
                wandb.log({"img": [wandb.Image(image_numpy)]})
                model.netG.train()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()  # update learning rates at the end of every epoch.
