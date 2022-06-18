from models.extractor import VitExtractor
from torchvision import transforms as T
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize(args):
    # load the image
    input_img = Image.open(args.image_path)
    input_img = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    # define the extractor
    dino_preprocess = T.Compose([
        T.Resize(224),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vit_extractor = VitExtractor(args.dino_model_name, device)

    # calculate the keys
    with torch.no_grad():
        keys_self_sim = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(input_img), args.layer)

    pca = PCA(n_components=3)
    pca.fit(keys_self_sim[0].cpu().numpy())
    components = pca.components_[None, ...]

    # reshape the reduced keys to the image shape
    patch_size = vit_extractor.get_patch_size()
    patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
    patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)
    pca_image = components[:, :, 1:].reshape(3, patch_h_num, patch_w_num).transpose(1, 2, 0)
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    h, w, _ = pca_image.shape
    pca_image = Image.fromarray(np.uint8(pca_image * 255))
    pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
    pca_image.save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default='datasets/feature_visualization/limes.jpeg', )
    parser.add_argument("--layer", type=int, default=11,
                        help='Transformer layer from which to extract the feature, between 0-11')
    parser.add_argument("--dino_model_name", type=str, default='dino_vitb8',
                        help='options: dino_vitb8 | dino_vits8 | dino_vitb16 | dino_vits16')
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    visualize(args)
