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
from vgg_batchout_1 import VGG

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

model = VGG(0)
model.to(device)

model_paths = '/home/hrushikesh/Downloads/model_reg_54.pt'
epsilons = [8/255, 0.0, 1/255]

model.load_state_dict(torch.load(model_paths))
print("Model loaded {}".format(model_paths))
test_acc, test_loss = epoch_perturbation(test_loader, model, device)
print("Test accuracy is: ", test_acc)
for e in epsilons:
    adv_err, adv_loss = epoch_perturbation_adversarial(test_loader, model, device, fgsm_batchout, e)
    print('Model: {}, error against adversaries for epsilon {} is: '.format(1, e), "{:.6f}".format(adv_err), sep="\t")



