import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.datasets as dset
from PIL import Image
from typing import Optional, Callable

class TrainDataset(dset.CIFAR10):
    def __init__(self, root: str, train: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                download: bool = False, sample: bool = False):
        if sample:
            #sample_path = os.path.join(root, "sample.pth")
            all_data = torch.load(root)
            self.data = all_data["input"]
            self.targets = all_data["target"]
            self.transform=transform
            self.target_transform = target_transform
        else:
            super(TrainDataset, self).__init__(root, root, train, transform,
                                              target_transform, download)

#class TrainDataset(Dataset):
#    def __init__(self, root, transform=None, target_transform=None):
#        data = torch.load(root)
#        self.data = data["input"]
#        self.targets = data["target"]
#        self.transform = transform
#        self.target_transform=target_transform
#
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, index):
#        img, target = self.data[index], self.targets[index]
#
#        # doing this so that it is consistent with all other datasets
#        # to return a PIL Image
#        img = Image.fromarray(img)
#
#        if self.transform is not None:
#            img = self.transform(img)
#
#        if self.target_transform is not None:
#            target = self.target_transform(target)
#
#        return img, target


