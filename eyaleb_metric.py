import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from models import StrWAE_stack_joint
from networks.convnet2 import Encoder, Decoder
from utils.dataset import extended_yaleb_pkl
from utils.metric import eyaleb_metric

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

"""
    Labeling light directions into 5 categories
    in order to fit Logistic regression from latent representations
"""
def light(s):
    if abs(s[0]) <= 0.2:
        if abs(s[1]) <= 0.2:
            return 0
    if s[0] >= 0.0:
        if s[1] >= 0.0:
            return 1
        else:
            return 2
    else:
        if s[1] >= 0.0:
            return 3
        else:
            return 4


if __name__ == "__main__":
    
    # arguments
    args = argparse.ArgumentParser()
    args.add_argument("--random-seed", type=int, default=2023)
    args.add_argument("--data-dir", type=str, default="./data")
    args.add_argument("--checkpoint-path", type=str)
    args = args.parse_args()

    torch.manual_seed(args.random_seed)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device("cuda") == device:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    train_dataset = extended_yaleb_pkl(args.data_dir, train=True)
    test_dataset = extended_yaleb_pkl(args.data_dir, train=False)
    
    train_y = train_dataset.tensors[1].numpy() # one-hot encoded
    test_y = test_dataset.tensors[1].numpy() # one-hot encoded
    train_s = train_dataset.tensors[2].numpy()
    test_s = test_dataset.tensors[2].numpy()

    # one-hot encoded (38 categories) -> label
    train_y = np.array([
        np.arange(38)[y.astype(bool)] for y in train_y
    ])
    test_y = np.array([
        np.arange(38)[y.astype(bool)] for y in test_y
    ])

    # categorize S
    train_s_c = np.array([light(s) for s in train_s])
    test_s_c = np.array([light(s) for s in test_s])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
    model = StrWAE_stack_joint(
        in_channels=1,
        base_channels=64,
        hidden_size=50,
        disc_size=16,
        latent_dim=2,
        label_dim=8,
        attr_dim=2,
        linear_bn=False,
        kernel_sizes=[[5, 5, 5, 3, 3], [3, 3, 3, 5, 5, 5]],
        input_size=128,
        scaling_steps=[1, 2],
        conv_steps=[0, 1],
        block_layers=[5, 2],
        skip=[False, False],
        fc_layers=[1, 1],
        disc_layers=5,
        learning_rate=0.001,
        learning_rate_gan=0.0005,
        encoder=Encoder,
        decoder=Decoder,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # for FID score
    fid = FID(normalize=True).to(device)

    # for sharpness
    filter = torch.nn.Conv2d(1, 1, kernel_size = 3, bias = False)
    filter.weight.data = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
    filter.to(device)

    # arrays for z1
    train_z1 = []
    test_z1 = []

    # obtain train_z1
    for x, y, s in tqdm(train_loader):
        x = x.to(device)
        s = s.to(device)[torch.randperm(x.shape[0])]
        with torch.no_grad():
            z, _, y_hat = model.encode(x)
            z1 = model.decoder_z1(torch.cat([z, y_hat], dim=1))
        train_z1.append(z1.to('cpu').numpy())

    # obtain test_z1 and compute generation FID score
    correct = 0
    sharpness = 0.0
    for x, y, s in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)
        s = s.to(device)[torch.randperm(x.shape[0])]
        with torch.no_grad():
            z, _, y_hat = model.encode(x)
            z_smpl = torch.randn_like(z).to(device)
            z1 = model.decoder_z1(torch.cat([z_smpl, y_hat], dim=1))
            new_x = model.decoder(torch.cat([z1, s], dim=1))
            y_logit = model.encoder_y.decode(y_hat)

        sharpness += filter(new_x).var(dim=[1,2,3]).sum().item()

        # 1 channel -> 3 channels
        x_ch3 = torch.cat([x] * 3, dim=1).to(device) 
        new_x_ch3 = torch.cat([new_x] * 3, dim=1).to(device)

        fid.update(x_ch3, real=True)
        fid.update(new_x_ch3, real=False)
        test_z1.append(z1.to('cpu').numpy())

        label = y.max(dim=1).indices
        pred_label = y_logit.max(dim=1).indices
        correct += (pred_label == label).sum().item()
    
    print(f"Y-embedder accuracy: {correct / len(test_y):.4f}")
    genfid = fid.compute()
    print(f"Gen FID: {genfid}\nSharpness: {sharpness / len(test_dataset)}")

    train_z1 = np.concatenate(train_z1, axis=0)
    test_z1 = np.concatenate(test_z1, axis=0)

    metric_lst = eyaleb_metric(
        train_z1, test_z1, train_s, test_s,
        train_s_c, test_s_c, train_y, test_y
    )
    
    # tSNE plots
    z1_embedded = TSNE(n_components=2).fit_transform(test_z1)
    
    plt.scatter(z1_embedded[:, 0], z1_embedded[:, 1], c = test_y)
    plt.savefig("person_tsne.png", dpi=300)
    
    plt.scatter(z1_embedded[:, 0], z1_embedded[:, 1], c = test_s_c)
    plt.savefig("light_tsne.png", dpi=300)
    
    # Conditional Generation
    # z1: computed from test images / s: 4 directions of light
    target = [124, 255, 274, 341, 370, 434] # indices of target identities
    
    x = torch.tensor(np.concatenate([np.array(test_dataset[i][0]) for i in target])).unsqueeze(1)
    s = torch.tensor(np.concatenate([np.array(test_dataset[i][2].unsqueeze(0)) for i in target]))
    new_s = torch.tensor([[0.3,0.3],[0.3,-0.3],[-0.3,-0.3],[-0.3,0.3]], device=device)
    with torch.no_grad():
        z, _, y_hat = model.encode(x.to(device))
        z1 = model.decoder_z1(torch.cat([z, y_hat], dim=1))
        recon = model.decoder(torch.cat([z1, s.to(device)], dim=1))
        
        # conditional generation
        new_x = model.decoder(torch.cat([z1.repeat((4,1)), new_s.repeat_interleave(6, dim=0)], dim=1))
    
    gen_image = make_grid(torch.cat([x, recon.to('cpu'), new_x.to('cpu')], dim=0), nrow=len(target))
    save_image(gen_image, "eyaleb_strwae.png")
        
