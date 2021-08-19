import numpy as np
import torch

from util import resize_right, interp_methods
from .base_model import BaseModel
from . import networks
import util.util as util
from .extractor import VitExtractor
from torchvision.transforms import Resize
from torchvision import transforms


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--use_cls', type=util.str2bool, nargs='?', const=True, default=False,
                            help='whether to use class descriptor loss')
        parser.add_argument('--cls_lambda', type=float, default=1.0, help='weight for class descriptor loss')
        parser.add_argument('--lambda_global_ssim', type=float, default=0.0, help='weight for global ssim loss')
        parser.add_argument('--lambda_patch_ssim', type=float, default=1.0, help='weight for patch ssim loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True)  # set lambda_dino to 1
        elif opt.CUT_mode.lower() == "fastcut":
            # set lambda_dino to 10
            parser.set_defaults(nce_idt=False, flip_equivariance=True, n_epochs=150, n_epochs_decay=50)
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'patch_ssim']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if opt.lambda_global_ssim > 0 and self.isTrain:
            self.loss_names += ['global_ssim']

        if opt.use_cls and self.isTrain:
            self.loss_names += ['cls']

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.extractor = VitExtractor(model_name='dino_vitb8', device='cuda')

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.isTrain and (self.opt.use_cls or self.opt.lambda_global_ssim > 0):
            self.global_A = input['A_global'][0].to(self.device)
            self.global_B = input['B_global'][0].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        if self.isTrain and (self.opt.use_cls or self.opt.lambda_global_ssim > 0):
            self.global_fake = self.netG(self.global_A)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # self similarity loss between real_A and fake_B
        self.loss_patch_ssim = self.calculate_patch_ssim_loss()

        self.loss_G = self.loss_G_GAN + self.loss_patch_ssim * self.opt.lambda_patch_ssim

        if self.opt.use_cls:
            # global class loss between B and fake
            self.loss_cls = self.calculate_cls_loss()
            self.loss_G += self.loss_cls
        if self.opt.lambda_global_ssim > 0:
            # global ssim loss
            self.loss_global_ssim = self.calculate_global_ssim()
            self.loss_G += self.loss_global_ssim * self.opt.lambda_global_ssim
        return self.loss_G

    def calculate_patch_ssim_loss(self):
        # self similarity loss between real_A and fake_B
        ssim_loss = 0.0
        for i in range(self.real_A.shape[0]):  # avoid memory limitations
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(self.real_A[i].unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(self.fake_B[i].unsqueeze(0), layer_num=11)
            ssim_loss += torch.nn.MSELoss()(keys_ssim, target_keys_self_sim)
        return ssim_loss

    def calculate_cls_loss(self):
        # class token similarity between real_B and fake_B
        fake_new_size = util.calc_size(self.global_fake, 224, max_size=480)
        fake_resized = resize_right.resize(self.global_fake, out_shape=fake_new_size)
        B_new_size = util.calc_size(self.global_B, 224, max_size=480)
        B_resized = resize_right.resize(self.global_B, out_shape=B_new_size)

        fake_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # imagenet normalization
        ])
        B_transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # imagenet normalization
        ])

        fake = fake_transform(fake_resized)
        B = B_transform(B_resized)
        target_cls_token = self.extractor.get_feature_from_input(B)[-1][0, 0, :].detach()
        cls_token = self.extractor.get_feature_from_input(fake)[-1][0, 0, :]
        cls_loss = torch.nn.MSELoss()(cls_token, target_cls_token)
        return cls_loss

    def calculate_global_ssim(self):
        # resize_layer = resize_right.ResizeLayer(in_shape, out_shape=None,
        #                                          interp_method=interp_methods.cubic, support_sz=None,
        #                                          antialiasing=True)

        fake_transform = transforms.Compose([
            Resize(224, max_size=480),
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # imagenet normalization
        ])
        A_transform = transforms.Compose([
            Resize(224, max_size=480),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # imagenet normalization
        ])
        fake = fake_transform(self.global_fake)
        A = A_transform(self.global_A)
        target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(A, layer_num=11).detach()
        keys_ssim = self.extractor.get_keys_self_sim_from_input(fake, layer_num=11)
        ssim_loss = torch.nn.MSELoss()(keys_ssim, target_keys_self_sim)
        return ssim_loss
