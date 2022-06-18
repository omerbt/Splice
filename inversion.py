from models.extractor import VitExtractor
from torchvision import transforms as T
import torch
from models.unet.skip import skip
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invert(args):
    # load the image
    input_img = Image.open(args.image_path)
    input_img = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    # network configurations
    net = skip(args.input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[4, 4, 4, 4, 4, 4],
               filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
               downsample_mode='avg', pad='reflection').to(device)
    net_input_saved = torch.randn((1, args.input_depth, input_img.shape[-2], input_img.shape[-1])).to(device)

    # define the extractor
    dino_preprocess = T.Compose([
        T.Resize(224),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vit_extractor = VitExtractor(args.dino_model_name, device)

    def extract_feature(x):
        if args.feature == 'cls':
            f = vit_extractor.get_feature_from_input(dino_preprocess(x))[args.layer][:, 0, :]
        elif args.feature == 'keys':
            f = vit_extractor.get_keys_from_input(dino_preprocess(x), args.layer)
        else:
            raise ValueError('feature {} not supported.'.format(args.feature))
        return f

    # calculate the target feature from the input image
    with torch.no_grad():
        ref_feature = extract_feature(input_img)

    # optimization configurations
    optimizer = torch.optim.Adam(net.parameters(), lr=args.LR)
    criterion = torch.nn.MSELoss()

    # inversion loop
    for i in tqdm(range(args.n_iter)):
        if args.feature == 'cls':
            # we're adding noise to the input at each step as a regularization
            if i < args.reduce_noise_stage_1_iter:
                net_input = net_input_saved + (torch.randn(net_input_saved.shape).to(device) * 10)
            elif i < args.reduce_noise_stage_2_iter:
                net_input = net_input_saved + (torch.randn(net_input_saved.shape).to(device) * 2)
            else:
                net_input = net_input_saved + (torch.randn(net_input_saved.shape).to(device) * 0.5)

        optimizer.zero_grad()
        current_feature = extract_feature(net(net_input))

        loss = criterion(current_feature, ref_feature)
        loss.backward()
        optimizer.step()

        if i % args.log_freq == 0:
            result_img = net(net_input)[0].detach().cpu().clone()
            result_img = T.ToPILImage()(result_img)
            result_img.save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--feature", type=str, help='DINO-ViT feature to invert. options: cls | keys')
    parser.add_argument("--layer", type=int, default=11,
                        help='Transformer layer from which to extract the feature, between 0-11')
    parser.add_argument("--dino_model_name", type=str, default='dino_vitb8')
    parser.add_argument("--image_path", type=str, default='datasets/feature_visualization/limes.jpeg',
                        help='path to the image to be used for the inversion.')
    parser.add_argument("--save_path", type=str, required=True, help='path to save the result.')
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--input_depth", type=int, default=32)
    parser.add_argument("--LR", type=float, default=0.01)
    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--reduce_noise_stage_1_iter", type=int, default=10000)
    parser.add_argument("--reduce_noise_stage_2_iter", type=int, default=15000)
    args = parser.parse_args()
    invert(args)
