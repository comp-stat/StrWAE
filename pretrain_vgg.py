import argparse, importlib
from tqdm import tqdm
from itertools import cycle

import torch
from torch.utils.data import DataLoader, random_split

from utils.dataset import VGGFace2_h5

"""
Pretraining (attribute(S), identity(Y))-embedding networks.

Identities of facial images(X) are available for 3,141,890 images, 
however, attributes (e.g. gender, mouth open, etc.) are only available for 30,000 images.

1. Attribute-embedding network (h_1)
It maps X to s-dimensional logits (s=the number of binary attributes).

2. Identity-embedding network (h_2)
Note that since there are new identities in the test dataset, the identity-embedding network is required,
which maps X to k-dimensional embedding vectors (k=latent dimension).
During training the embedding network, it is combined with the decoder,
which maps k-dimensional embedding vectors to L-dimensional logits (L=the number of identities in the train dataset).

These embedding networks will be used for training StrWAE (S_hat <- h_1(X), Y_hat <- h_2(X)).
"""

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", type=str, default="./data")
    args.add_argument("--random-seed", type=int, default=2023)
    args.add_argument("--full", action="store_true")
    args = args.parse_args()

    # If full_mode==True, use full labeled dataset.
    # Else, use attr dataset
    full_mode = args.full
    print("full_mode:", full_mode)

    device = "cuda"
    # dataset = "vggface2"
    label_dataset = VGGFace2_h5(args.data_dir, train=True, attr=False)
    attr_dataset = VGGFace2_h5(args.data_dir, train=True, attr=True)
    print(f"label dset: {len(label_dataset)}\tattr_dset: {len(attr_dataset)}")

    # train-val split
    train_label_dataset, valid_label_dataset = random_split(
        label_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(args.random_seed)
    )
    train_attr_dataset, valid_attr_dataset = random_split(
        attr_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(args.random_seed)
    )

    train_label_loader = DataLoader(
        train_label_dataset,
        # batch_size=200,
        batch_size=1024,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )
    train_attr_loader = DataLoader(
        train_attr_dataset,
        # batch_size=4,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )

    valid_label_loader = DataLoader(
        valid_label_dataset,
        batch_size=512,
        # batch_size=200,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )
    valid_attr_loader = DataLoader(
        valid_attr_dataset,
        # batch_size=4,
        batch_size=512,
        num_workers=16,
        pin_memory=True,
        shuffle=False
    )
    print(f"[Train] label dloader: {len(train_label_loader)}\tattr_dloader: {len(train_attr_loader)}")
    print(f"[Validation] label dloader: {len(valid_label_loader)}\tattr_dloader: {len(valid_attr_loader)}")

    base_channels = 128
    label_dim = 8631
    latent_dim = 64
    attr_dim = 7
    input_size = 128
    learning_rate = 1e-3
    kernel_size = 5
    conv_layers = 4
    fc_layers = 0
    optim_beta = 0.9
    epoch = 200 # 2 if full_mode else 50
    save_path = (
        f"./checkpoints/embedders/vggface2_simple_jit.pt"
    )

    # networks
    encoder = getattr(
        importlib.import_module("networks.convnet"), "Encoder"
    )
    embedder = getattr(importlib.import_module("models.base"), "Embedder")

    in_channels = 3
    dict_param = {
        "in_channels": in_channels,
        "base_channels": base_channels,
        "label_dim": label_dim,
        "latent_dim": latent_dim,
        "attr_dim": attr_dim,
        "input_size": input_size,
        "kernel_size": kernel_size,
        "conv_layers": conv_layers,
        "fc_layers": fc_layers,
        "learning_rate": learning_rate,
        "optim_beta": optim_beta,
        "encoder": encoder,
    }

    model = embedder(**dict_param)
    model = model.to(device)

    dict_optim = model.get_optimizers()
    optimizer = dict_optim["optimizer"]

    train_early = 100
    valid_early = 100

    best_correct = 0
    # model = torch.compile(model) # pytorch 2.0: speed up
    for epoch in range(1, epoch+1):
        # train
        model.train()
        tqdm_train_loader = tqdm(
            zip(train_label_loader,
                cycle(train_attr_loader) if full_mode else train_attr_loader)
        )
        total_train_label_loss, total_train_attr_loss = 0, 0
        total_train_loss = 0
        for i, batch in enumerate(tqdm_train_loader):
            tqdm_train_loader.set_description(f"Train epoch {epoch}")

            if i >= train_early:
                break

            (data_label, label), (data_attr, attr)  = batch
            data_label = data_label.to(device) # 
            label = label.to(device)
            data_attr = data_attr.to(device)
            attr = attr.to(device)

            # training embedder & decoder
            optimizer.zero_grad()
            loss_labeled = model.get_losses(
                x=data_label,
                y=label,
                mode="labeled"
            )
            
            loss_attr = model.get_losses(
                x=data_attr,
                s=attr,
                mode="attribute"
            )
            
            weighted_loss = loss_labeled + 0.1 * loss_attr
            weighted_loss.backward()
            total_train_loss += weighted_loss.item()
            total_train_label_loss += loss_labeled.item()
            total_train_attr_loss += loss_attr.item()
            optimizer.step()
            
            tqdm_train_loader.set_postfix(
                loss=total_train_loss / (i+1),
                labeled=total_train_label_loss / (i+1),
                attribute=total_train_attr_loss/ (i+1)
            )
        
        # validation
        model.eval()
        tqdm_valid_loader = tqdm(
            zip(valid_label_loader, valid_attr_loader)
        )

        total_valid_label_loss, total_valid_attr_loss = 0, 0
        total_valid_label_correct, total_valid_attr_correct = 0, 0
        total_valid_label_size, total_valid_attr_size = 0, 0
        total_valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm_valid_loader):
                tqdm_valid_loader.set_description(f"Valid epoch {epoch}")
                
                if i >= valid_early:
                    break

                (data_label, label), (data_attr, attr)  = batch
                data_label = data_label.to(device) # 
                label = label.to(device)
                data_attr = data_attr.to(device)
                attr = attr.to(device)
                total_valid_label_size += label.size(0)
                total_valid_attr_size += attr.size(0)
                
                # Computing valid_loss, valid_acc
                valid_loss_labeled, _, y_hat = model.get_losses(
                    x=data_label,
                    y=label,
                    mode="labeled",
                    valid=True
                )

                valid_loss_attr, s_hat, _ = model.get_losses(
                    x=data_attr,
                    s=attr,
                    mode="attribute",
                    valid=True
                )

                weighted_valid_loss = valid_loss_labeled + 0.1 * valid_loss_attr

                total_valid_loss += weighted_valid_loss.item()
                total_valid_label_loss += valid_loss_labeled.item()
                total_valid_attr_loss += valid_loss_attr.item()
                _, prediction = torch.max(y_hat.data, dim=1)
                total_valid_label_correct += (prediction == label).sum().item()

                total_valid_attr_correct += (
                    ((s_hat > 0) == attr).sum().item()
                )
                # total_valid_attr_correct /= attr_dim

                tqdm_valid_loader.set_postfix(
                    loss=total_valid_loss / (i+1),
                    labeled=total_valid_label_loss / (i+1),
                    attribute=total_valid_attr_loss / (i+1),
                    acc=100 * total_valid_label_correct / total_valid_label_size,
                    acc_attr=100 * total_valid_attr_correct / (7 * total_valid_attr_size)
                )

        total_valid_correct = total_valid_label_correct + total_valid_attr_correct

        if total_valid_correct > best_correct:
            best_correct = total_valid_correct
            model_script = torch.jit.script(model)

            model_script.save(save_path)
            print(best_correct)