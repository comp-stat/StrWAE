import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.functional import one_hot

from networks.base import MLPBlock
import networks.resnet as resnet
import networks.convnet as simple

from utils import SVHNSearchDataset, CosineAnnealingWarmUpRestarts

"""
Training digit classifiers.
The classifiers are used to evaluate the conditional generated images.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.CrossEntropyLoss()


    if args.dataset.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True)
        ])
        dataset1 = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )
        train_loader = DataLoader(dataset1, batch_size=512, shuffle=True)
        test_loader = DataLoader(dataset2, batch_size=512, shuffle=False)

        encoder = simple.Encoder
        classifier_name = "simple"
        dict_args_encoder = {
            "in_channels": 1,
            "base_channels": 32,
            "hidden_size": 32,
            "latent_dim": 128,
            "kernel_size": 3,
            "input_size": 32,
            "conv_layers": 4,
            "fc_layers": 2,
        }

    elif args.dataset.lower() == "svhn":
        dataset1 = SVHNSearchDataset(
            "./data", split="extra",
            download=True, transform=transforms.ToTensor()
        )
        dataset2 = SVHNSearchDataset(
            "./data", split="test",
            download=True, transform=transforms.ToTensor()
        )
        train_loader = DataLoader(dataset1, batch_size=512, shuffle=True)
        test_loader = DataLoader(dataset2, batch_size=512, shuffle=False)

        encoder = resnet.Encoder
        classifier_name = "resnet"
        dict_args_encoder = {
            "in_channels": 3,
            "base_channels": 64,
            "latent_dim": 128,
            "input_size": 32,
            "layers": 4
        }

    model = nn.Sequential(
        # encoder(
        #     in_channels=in_channels,
        #     base_channels=base_channels,
        #     latent_dim=128,
        #     input_size=32,
        #     conv_layers=conv_layers,
        #     fc_layers=2,
        #     bias=False
        # ),
        encoder(**dict_args_encoder),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(inplace=True),
        nn.Linear(128, 10)
    ).to(device)

    epochs = 50
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=100,
        T_mult=1,
        eta_max=lr * 3,
        T_up=10,
        gamma=0.5
    )

    best_correct = 0
    for epoch in range(1, epochs+1):
        train(
            model, device, train_loader, criterion, optimizer, scheduler, epoch
        )
        correct = test(model, device, test_loader)
        if best_correct < correct:
            best_correct = correct
            model_script = torch.jit.script(model)
            model_script.save(
                # f"checkpoints/{args.dataset}_{classifier_name}_{base_channels}_jit.pt"
                f"checkpoints/{args.dataset.lower()}_{classifier_name}_jit.pt"
            )
            # torch.save(
            #     model.state_dict(),
            #     f"checkpoints/classifier_{args.dataset}_resnet.pt"
            # )
    print(best_correct / len(test_loader.dataset))


def train(model, device, dataloader, criterion, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    total = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        total += target.size(0)
    
    train_loss /= total
    print(f"[Train] Epoch: {epoch} Loss: {train_loss:.2f}")


def test(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"[Test] Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f})")
    return correct


if __name__ == "__main__":
    main()