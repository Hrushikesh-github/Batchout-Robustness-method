import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np

dataset = CIFAR10('~/data', train=True, transform=transforms.ToTensor(), download=True)
class_inds = [torch.where(dataset.targets == class_idx)[0] for class_idx in dataset.class_to_idx.values()]

train_label = np.array(dataset.targets)
a0 = np.where(y == 0)[0]
a1 = np.where(y == 1)[0]
a2 = np.where(y == 2)[0]
a3 = np.where(y == 3)[0]
a4 = np.where(y == 4)[0]
a5 = np.where(y == 5)[0]
a6 = np.where(y == 6)[0]
a7 = np.where(y == 7)[0]
a8 = np.where(y == 8)[0]
a9 = np.where(y == 9)[0]

print("ao shape: ", a0.shape)
class_idx = np.vstack([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9])

subset_idx = class_idx[:, 5:8].reshape(-1, 1)
train_data = dataset.data
X = train_data[subset_idx].squeeze()

y = train_label[subset_idx].squeeze()
