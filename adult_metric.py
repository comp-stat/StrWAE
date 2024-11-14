import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from models import StrWAE_stack_cond, HCV
from utils.dataset import Adult_pkl
from utils.metric import adult_metric

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
    
    # load dataset & dataloader
    train_dataset = Adult_pkl(
        args.data_dir, 
        train=True,
    )
    test_dataset = Adult_pkl(
        args.data_dir, 
        train=False, 
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False
    )

    train_y = train_dataset.data["label"].values
    test_y = test_dataset.data["label"].values
    train_s = train_dataset.data["sex"].values
    test_s = test_dataset.data["sex"].values

    model = StrWAE_stack_cond(
        hidden_size=100,
        disc_size=128,
        label_dim=1,
        latent_dim=32,
        linear_bn=False,
        input_size=113,
        fc_layers=2,
        disc_layers=4,
        learning_rate=0.001,
        learning_rate_gan=0.001,
        attr_dim=1
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # arrays for z1
    train_z1, test_z1 = [], []
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            data, label, attr = batch
            data = data.to(device)
            label = label.to(device)
            attr = attr.to(device)
            
            ingredient = model.first_operation(
                x=data, s=attr, y=label
            )
            train_z1.append(ingredient[0].cpu().numpy())
        
        for batch in tqdm(test_dataloader):
            data, label, attr = batch
            data = data.to(device)
            label = label.to(device)
            attr = attr.to(device)
            
            ingredient = model.first_operation(
                x=data, s=attr, y=label
            )
            test_z1.append(ingredient[0].cpu().numpy())

    train_z1 = np.concatenate(train_z1, axis=0)
    test_z1 = np.concatenate(test_z1, axis=0)
    
    for cls_name in ["lr"]:
        fair_metric_dict = adult_metric(train_z1, test_z1, train_s, test_s, train_y, test_y, cls_name=cls_name)

        for key, value in fair_metric_dict.items():
            print(f"{cls_name}_{key}: {value:.6f}")
    
