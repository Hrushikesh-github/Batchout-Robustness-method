import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from vgg_change import VGG
import numpy as np
import foolbox as fb
from foolbox.attacks import PGD, FGSM

data_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

batch_size = 128

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])

cifar_test = datasets.CIFAR10("/home/hrushikesh/torch/data", train=False, download=True, transform=test_transforms)

test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

model = VGG()
model.to(device)
model.load_state_dict(torch.load('expt3_bn_after_long/no_model_reg_85.pt'))
model.eval()
print("Model loaded..")

bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds)

#attack = PGD()
attack = FGSM()

errors = []
for i, (X, y) in enumerate(test_loader):
    X, y = X.to(device), y.to(device)
    raw, clipped, is_adv = attack(fmodel, X, y, epsilons=0)
    #print(is_adv)
    error = float(is_adv.float().mean(axis=-1))
    errors.append(error)
    print(i)
    if i == 10:
        break
mean = np.mean(errors)
print(mean)

