import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        data = torch.load(root)
        self.data = data["input"]
        self.targets = data["target"]
        self.transform = transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


