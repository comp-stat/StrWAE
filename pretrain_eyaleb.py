import argparse, importlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.dataset import extended_yaleb_pkl

"""
Pretraining identity(Y)-embedding network.

These embedding networks will be used for training StrWAE (Y_hat <- h(X)).
"""

if __name__ == "__main__":
    device = "cuda"

    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", type=str, default="./data")
    args.add_argument("--random-seed", type=int, default=2023)
    args = args.parse_args()

    torch.manual_seed(args.random_seed)

    train_dataset = extended_yaleb_pkl(args.data_dir, train=True)
    valid_dataset = extended_yaleb_pkl(args.data_dir, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        num_workers=16,
        pin_memory=True
    )
    
    base_channels = 64
    label_dim = 38
    latent_dim = 8
    input_size = 128
    learning_rate = 1e-3
    kernel_size = 3
    conv_layers = 5
    fc_layers = 0
    epoch = 50

    save_path = (
        f"./checkpoints/embedders/eyaleb_simple_jit.pt"
    )

    # networks
    encoder = getattr(
        importlib.import_module("networks.convnet2"), "Encoder"
    )
    embedder = getattr(importlib.import_module("models.base"), "Embedder")

    in_channels = 1
    dict_param = {
        "in_channels": in_channels,
        "base_channels": base_channels,
        "label_dim": label_dim,
        "latent_dim": latent_dim,
        "input_size": input_size,
        "conv_layers": conv_layers,
        "fc_layers": fc_layers,
        "learning_rate": learning_rate,
        "encoder": encoder,
        "activation": torch.nn.ReLU
    }

    model = embedder(**dict_param)
    model = model.to(device)

    dict_optim = model.get_optimizers()
    optimizer = dict_optim["optimizer"]
 
    best_acc = 0
    # model = torch.compile(model) # pytorch 2.0: speed up
    for epoch in range(1, epoch+1):
        # train
        model.train()
        tqdm_train_loader = tqdm(train_loader)
        total_train_loss = 0.
        total_train_size = 0
        for i, batch in enumerate(tqdm_train_loader):
            tqdm_train_loader.set_description(f"Train epoch {epoch}")

            data, label, attr = batch
            total_train_size += label.size(0)
            data = data.to(device) # 
            label = label.to(device)

            # training encoder, decoder
            optimizer.zero_grad()
            loss = model.get_losses(data, label, mode="labeled")
            loss.backward()
            total_train_loss += loss.item() * data.size(0)
            optimizer.step()
            
            tqdm_train_loader.set_postfix(
                loss=total_train_loss / total_train_size
            )
        
        # validation
        model.eval()
        tqdm_valid_loader = tqdm(valid_loader)

        total_valid_loss = 0.
        total_valid_correct = 0
        total_valid_size = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm_valid_loader):
                tqdm_valid_loader.set_description(f"Valid epoch {epoch}")
                
                # Parsing data
                data, label, attr = batch
                total_valid_size += label.size(0)
                data = data.to(device)
                label = label.to(device)
                
                # Computing valid_loss, valid_acc
                valid_loss, _, y_logit = model.get_losses(
                    data, label, mode="labeled", valid=True
                )
                
                total_valid_loss += valid_loss.item() * label.size(0)
                prediction = y_logit.max(dim=1).indices
                gt_label = label.max(dim=1).indices
                total_valid_correct += (prediction == gt_label).sum().item()

                tqdm_valid_loader.set_postfix(
                    loss=total_valid_loss / total_valid_size,
                    acc=100 * (total_valid_correct / total_valid_size)
                )

        if total_valid_correct > best_acc:
            best_acc = total_valid_correct
            model_script = torch.jit.script(model)
            model_script.save(save_path)
            print(f"{best_acc} / {len(valid_dataset)} = {100 * best_acc / len(valid_dataset)} %")
    
    model_script = torch.jit.script(model)
    model_script.save("checkpoints/embedders/eyaleb_last.pt")
