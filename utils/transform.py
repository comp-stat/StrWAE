import torch
from torchvision import transforms

T = dict(
    mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32), antialias=True),
    ]),
    svhn = transforms.Compose([
        transforms.ToTensor(),
    ]),
    celeba = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.Resize(64),
        transforms.ToTensor(),
    ]),
    eyaleb = transforms.Compose([
        transforms.ToTensor(),
    ]),
    vggface2 = transforms.Compose([
        transforms.ToTensor(),
    ]),
)
