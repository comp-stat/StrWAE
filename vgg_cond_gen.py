import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as F

from models import StrWAE_embedder
from networks.convnet2 import Encoder, Decoder
from utils.dataset import VGGFace2_h5

import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # arguments
    args = argparse.ArgumentParser()
    args.add_argument("--random-seed", type=int, default=1)
    args.add_argument("--data-dir", type=str, default="./data")
    args.add_argument("--checkpoint-path", type=str)
    args = args.parse_args()

    torch.manual_seed(args.random_seed)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device("cuda") == device:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load_model
    model = StrWAE_embedder(
        in_channels=3,
        base_channels=128,
        hidden_size=32,
        disc_size=256,
        latent_dim=32,
        label_dim=64,
        attr_dim=7,
        linear_bn=False,
        kernel_sizes=[[5, 5, 5, 5, 5, 3, 3, 3], [5, 5, 5, 5, 5, 3, 3, 3]],
        input_size=128,
        # conv_layers=3,
        scaling_steps=[1, 1],
        conv_steps=[1, 1],
        block_layers=[4, 4],
        skip=[False, True],
        fc_layers=[1, 1],
        disc_layers=5,
        learning_rate=0.001,
        learning_rate_gan=0.001,
        encoder=Encoder,
        decoder=Decoder,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # test samples
    test_dataset = VGGFace2_h5(args.data_dir, train=False, attr=False)

    test_idx = [56280, 61156, 68657, 84886, 92298, 97773, 102644, 126099, 130330, 137081] # identity
    k = len(test_idx)

    test_images = torch.cat([test_dataset[i][0].unsqueeze(0) for i in test_idx], dim=0)
    test_images = test_images.to(device)

    # encoding & decoding
    with torch.no_grad():
        test_z, test_s_hat, test_y_hat = model.encode(test_images)

    n_gen = 6
    new_z_smpl = torch.randn((n_gen, model.latent_dim)).to(device)

    # Reconstruction & Class-preserving generation
    with torch.no_grad():
        recon_images = model.decode(test_z, test_y_hat, test_s_hat).cpu()
        gen_images = model.decode(
            new_z_smpl.repeat_interleave(k, dim=0), 
            test_y_hat.repeat((n_gen, 1)), 
            test_s_hat.repeat((n_gen, 1))
        ).cpu()

    cpu_test_images = test_images.cpu()
    total_images = torch.cat(
        [cpu_test_images, recon_images, gen_images], 
        dim=0
    )
    grid = make_grid(total_images, nrow=k, padding=0)
    save_image(grid, "vgg_generation.png")
    
    # Style transfer
    with torch.no_grad():
        style_transfer_images = model.decode(
            test_z.repeat_interleave(k, dim=0), 
            test_y_hat.repeat((k, 1)), 
            test_s_hat.repeat((k, 1))
        ).cpu()

    total_style_transfer = torch.cat(
        [torch.zeros((1, 3, 128, 128)), cpu_test_images] + \
            [torch.cat([cpu_test_images[i].unsqueeze(0), style_transfer_images[i*k:(i+1)*k]], dim=0) for i in range(k)],
        dim=0
    )

    grid_style_transfer = make_grid(total_style_transfer, nrow=k+1, padding=0)
    save_image(grid_style_transfer, "vgg_style_transfer.png")

    # Attribute manipulation & interpolation
    attr_list = ["Male", "Long Hair", "Beard", "Hat", "Eyeglasses", "Sunglasses", "Mouth open"]
    attr_target = 2
    attr_high_val, attr_low_val = 50.0, -25.0
    
    attr_manipulate_list = [0, 2, 5, 6] # Male, Beard, Sunglasses, Mouth open
    interpolate_id_target = {0: 4, 2: 2, 5: 3, 6: 1}
    for attr_target in attr_manipulate_list:
        test_high_s = torch.zeros_like(test_s_hat)
        test_high_s[:, :] = test_s_hat[:, :]
        test_high_s[:, attr_target] = attr_high_val

        test_low_s = torch.zeros_like(test_s_hat)
        test_low_s[:, :] = test_s_hat[:, :]
        test_low_s[:, attr_target] = attr_low_val

        with torch.no_grad():
            attr_high_images = model.decode(test_z, test_y_hat, test_high_s).cpu()
            attr_low_images = model.decode(test_z, test_y_hat, test_low_s).cpu()

        attr_manipulate_images = torch.cat(
            [attr_high_images, cpu_test_images, attr_low_images],
            dim=0
        )
        attr_manipulate_grid = make_grid(attr_manipulate_images, nrow=k, padding=0)
        save_image(attr_manipulate_grid, f"vgg_manipulate_{attr_list[attr_target]}.png")

        # interploation
        test_interpolate_s = torch.stack([test_s_hat[interpolate_id_target[attr_target]]] * k, dim=0)
        test_interpolate_s[:, attr_target] = torch.from_numpy(np.linspace(attr_low_val, attr_high_val, num=k)).to(device)
        with torch.no_grad():
            attr_interpolate_images = model.decode(
                torch.stack([test_z[interpolate_id_target[attr_target]]] * k, dim=0),
                torch.stack([test_y_hat[interpolate_id_target[attr_target]]] * k, dim=0),
                test_interpolate_s
            ).cpu()
        
        attr_interpolate_grid = make_grid(attr_interpolate_images, nrow=k, padding=0)
        save_image(attr_interpolate_grid, f"vgg_interpolate_{attr_list[attr_target]}.png")

    plt.close()