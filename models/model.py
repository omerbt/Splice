import numpy as np
import torch
from . import networks
import util.util as util
from .extractor import VitExtractor
from torchvision.transforms import Resize
from torchvision import transforms


class Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--cls_lambda', type=float, default=1.0, help='weight for class descriptor loss')
        parser.add_argument('--lambda_global_ssim', type=float, default=0.0, help='weight for global ssim loss')
        parser.add_argument('--lambda_patch_ssim', type=float, default=1.0, help='weight for patch ssim loss')
        parser.add_argument('--lambda_GAN', type=float, default=0.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--use_class_token', type=util.str2bool, nargs='?', const=True, default=True,
                            help="Whether to use class token as the descriptor (or keys)")
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            # set lambda_dino to 1
            parser.set_defaults(lambda_identity=1)
        elif opt.CUT_mode.lower() == "fastcut":
            # set lambda_dino to 10
            parser.set_defaults(lambda_identity=0, flip_equivariance=True, n_epochs=150, n_epochs_decay=50)
        else:
            raise ValueError(opt.CUT_mode)

        parser.add_argument('--lambda_R1', type=float, default=1.0,
                            help='weight for the R1 gradient penalty')
        parser.add_argument('--lambda_identity', type=float, default=1.0,
                            help='the "identity preservation loss"')

        parser.set_defaults(
            dataset_mode="singleimage",
            netG="stylegan2",
            stylegan2_G_num_downsampling=1,
            netD="stylegan2",
            gan_mode="nonsaturating",
            num_patches=1,
            ngf=10,
            ndf=8,
            lr=0.002,
            beta1=0.0,
            beta2=0.99,
            load_size=1024,
            crop_size=64,
            preprocess="zoom_and_patch",
        )

        if is_train:
            parser.set_defaults(preprocess="zoom_and_patch",
                                batch_size=16,
                                save_epoch_freq=1,
                                save_latest_freq=20000,
                                n_epochs=8,
                                n_epochs_decay=8,
                                )
        else:
            parser.set_defaults(preprocess="none",  # load the whole image as it is
                                batch_size=1,
                                num_test=1,
                                )

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.init_type, opt.init_gain, opt)

        self.extractor = VitExtractor(model_name=opt.dino_model_name, device=self.device)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # imagenet normalization
        resize_transform = Resize(opt.global_patch_size, max_size=480)
        if self.opt.skip_activation == 'tanh':
            self.global_fake_transform = transforms.Compose([
                resize_transform,
                transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
                imagenet_norm
            ])
            self.local_real_transform = self.local_fake_transform = transforms.Compose([
                transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # [-1, 1] -> [0, 1]
                imagenet_norm
            ])
        else:
            self.global_fake_transform = transforms.Compose([
                resize_transform,
                imagenet_norm
            ])
            self.local_real_transform = self.local_fake_transform = transforms.Compose([
                imagenet_norm
            ])
        self.global_real_transform = transforms.Compose([
            resize_transform,
            imagenet_norm
        ])
        # fake_new_size = util.calc_size(self.global_fake.shape, 224, max_size=480)
        # fake_resized = resize_right.resize(self.global_fake.shape, out_shape=fake_new_size)


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
        if self.opt.cls_lambda > 0 or self.opt.lambda_global_ssim > 0:
            self.global_A = input['A_global'][0].to(self.device)
            self.global_B = input['B_global'][0].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.lambda_identity > 0 else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = np.random.random() < 0.5
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.lambda_identity > 0:
            self.idt_B = self.fake[self.real_A.size(0):]
        if self.opt.cls_lambda + self.opt.lambda_global_ssim > 0:
            self.global_fake = self.netG(self.global_A)


    def compute_G_loss(self):
        losses = {}
        fake = self.fake_B
        self.loss_G = 0

        if self.opt.lambda_GAN > 0.0:
            # G(A) should fake the discriminator
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            losses['loss_G_GAN'] = self.loss_G_GAN
            self.loss_G += self.loss_G_GAN * self.opt.lambda_GAN

        if self.opt.lambda_patch_ssim > 0:
            # self similarity loss between real_A and fake_B
            self.loss_patch_ssim = self.calculate_patch_ssim_loss()
            losses['loss_patch_ssim'] = self.loss_patch_ssim
            self.loss_G += self.loss_patch_ssim * self.opt.lambda_patch_ssim

        if self.opt.lambda_global_ssim > 0:
            # global ssim loss
            self.loss_global_ssim = self.calculate_global_ssim()
            losses['loss_global_ssim'] = self.loss_global_ssim
            self.loss_G += self.loss_global_ssim * self.opt.lambda_global_ssim

        if self.opt.cls_lambda > 0:
            # global class loss between B and fake
            self.loss_cls = self.calculate_cls_loss()
            losses['loss_cls'] = self.loss_cls
            self.loss_G += self.loss_cls * self.opt.cls_lambda

        if self.opt.lambda_identity > 0:
            # |G(I)-I| for I crop from B
            self.loss_idt_B = torch.nn.functional.l1_loss(self.idt_B, self.real_B)
            losses['loss_idt_B'] = self.loss_idt_B
            self.loss_G += self.loss_idt_B * self.opt.lambda_identity

        losses['total'] = self.loss_G
        return losses


    def calculate_patch_ssim_loss(self):
        # self similarity loss between real_A and fake_B
        ssim_loss = 0.0
        for i in range(self.real_A.shape[0]):  # avoid memory limitations
            A = self.local_real_transform(self.real_A[i])
            fake = self.local_fake_transform(self.fake_B[i])
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(A.unsqueeze(0),
                                                                               layer_num=11).detach()
            keys_ssim = self.extractor.get_keys_self_sim_from_input(fake.unsqueeze(0), layer_num=11)
            ssim_loss += torch.nn.MSELoss()(keys_ssim, target_keys_self_sim)
        return ssim_loss


    def calculate_cls_loss(self):
        # class token similarity between real_B and fake_B
        fake = self.global_fake_transform(self.global_fake)
        B = self.global_real_transform(self.global_B)
        if self.opt.use_class_token:
            target_cls_token = self.extractor.get_feature_from_input(B)[-1][0, 0, :].detach()
            cls_token = self.extractor.get_feature_from_input(fake)[-1][0, 0, :]
            cls_loss = torch.nn.MSELoss()(cls_token, target_cls_token)
        else:
            # use class keys
            assert self.opt.dino_model_name == 'dino_vits8'
            # head_idx = [0, 2, 4, 5]  # relevant for dino_vits8
            head_idx = [4]  # relevant for dino_vits8
            target_cls_keys = self.extractor.get_keys_from_input(B, 11).detach()[head_idx]
            cls_keys = self.extractor.get_keys_from_input(fake, 11)[head_idx]
            # get concatenated features
            h, t, d = target_cls_keys.shape
            target_cls_keys = target_cls_keys.transpose(0, 1).reshape(-1, h * d)[0]
            cls_keys = cls_keys.transpose(0, 1).reshape(-1, h * d)[0]
            cls_loss = torch.nn.MSELoss()(cls_keys, target_cls_keys)
        return cls_loss


    def calculate_global_ssim(self):
        # keys self similarity between real_A and fake_B
        fake = self.global_fake_transform(self.global_fake)
        A = self.global_real_transform(self.global_A)
        target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(A, layer_num=11).detach()
        keys_ssim = self.extractor.get_keys_self_sim_from_input(fake, layer_num=11)
        ssim_loss = torch.nn.MSELoss()(keys_ssim, target_keys_self_sim)
        return ssim_loss


    def calculate_global_identity(self):
        fake = self.global_fake_transform(self.global_fake)
        A = self.global_real_transform(self.global_A)
