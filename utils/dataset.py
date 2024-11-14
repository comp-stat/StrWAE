import h5py
import pickle
import numpy as np
import pandas as pd

import torch
from torch.functional import F
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


class SVHNSearchDataset(datasets.SVHN):
    def __getitem__(self, index):
        image, label = self.data[index], int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image).permute(1, 2, 0)
        return image, label


class VGGFace2_h5(Dataset):
    def __init__(self, data_dir, train: bool = True, attr: bool = False):
        # Active attributes [1,0,0,0,0,1,1,1,1,1,1] among 11 attributes:
        #   male, longhair, Beard, hat, eyeglass, sunglass, mouth open
        self.attr = attr

        if attr:
            db = h5py.File(f"{data_dir}/VGGFace2/attr.h5", "r")
            self.x = db["image"]
            self.s = db["attribute"]
        else:
            if train:
                db = h5py.File(f"{data_dir}/VGGFace2/train.h5", "r")
            else:
                db = h5py.File(f"{data_dir}/VGGFace2/test.h5", "r")
            self.x = db["image"]
            self.y = db["label"]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # image range: [0, 1]
        if self.attr:
            return (
                torch.from_numpy(self.x[idx]) / 255.0,
                torch.from_numpy(self.s[idx, [0, 5, 6, 7, 8, 9, 10]])
            )
        return (
            torch.from_numpy(self.x[idx]) / 255.0,
            torch.from_numpy(np.array(self.y[idx])).long()
        )


def extended_yaleb_pkl(data_dir, train: bool = True):
    if train:
        data_path = f"{data_dir}/ExtendedYaleB/YaleBFaceTrain.dat"
    else:
        data_path = f"{data_dir}/ExtendedYaleB/YaleBFaceTest.dat"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = torch.from_numpy(data['image']).reshape((-1, 1, 128, 128)) / 255.
    S = torch.from_numpy(np.c_[data['azimuth'], data['elevation']]).type(torch.float32)
    S = S.div(torch.tensor([180.0, 90.0], dtype=torch.float32))

    Y = torch.from_numpy(data['person']).type(torch.long)
    num_classes = len(Y.unique())
    Y = F.one_hot(Y, num_classes)
    Y = Y.float()

    return TensorDataset(X, Y, S)
    
class Adult_pkl(Dataset):
    def __init__(self, dir_path, train=True):
        if train:
            self.path = f"{dir_path}/adult_train.pkl"
        else:
            self.path = f"{dir_path}/adult_test.pkl"
        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        y = torch.as_tensor(data.pop('label'), dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(data.pop('sex'), dtype=torch.float32).reshape(-1)
        x = torch.as_tensor(data, dtype=torch.float32)

        # x_dimension: 113, y & s: 1
        return [x, y, s]
