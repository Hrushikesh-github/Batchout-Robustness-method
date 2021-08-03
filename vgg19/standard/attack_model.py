import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('/home/hrushikesh/robust/cifar3')
sys.path.append('/home/hrushikesh/')
from torch_functions import *
from vgg_change import VGG
import argparse

data_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

batch_size = 32

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

aug = transforms.RandomChoice((transforms.RandomHorizontalFlip(p=1), transforms.RandomCrop(32, padding=4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
    transforms.RandomAffine(degrees=0, translate=(0, 0.1))))

train_transforms = transforms.Compose([aug, transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])

cifar_train = datasets.CIFAR10("/home/hrushikesh/torch/data", train=True, download=True, transform=train_transforms)
cifar_test = datasets.CIFAR10("/home/hrushikesh/torch/data", train=False, download=True, transform=test_transforms)

train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

model = VGG()
model.to(device)

model.load_state_dict(torch.load('expt3_bn_after_long/no_model_reg_85.pt'))
print("Model loaded..")

epsilons = [0.0, 1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255]

for i, model in enumerate([model]):
    test_acc, test_loss = epoch(test_loader, model, device)
    print("Test accuracy is: ", test_acc)
    for e in epsilons:
        adv_err, adv_loss = epoch_adversarial(test_loader, model, device, fgsm, e)
        print('Model: {}, error against adversaries for epsilon {} is: '.format(i + 1, e), "{:.6f}".format(adv_err), sep="\t")


