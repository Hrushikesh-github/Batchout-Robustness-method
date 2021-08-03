import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('/home/hrushikesh/')
from torch_functions import *
from batchout_new import Batchout_New
from vgg import VGG
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

print(args)

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

opt = optim.SGD(model.parameters(), lr=1e-2)

# We need to initalize k=0 for correct plotting
k = 0

if args['model']:
    model.load_state_dict(torch.load(args['model']))
    print("Model loaded.. here we go")

print('Started Training')


for i in range(args['start_epoch'] + 1, args['start_epoch'] + 36):

    # Perform a epoch and get accuracies and loss values.
    acc, loss = epoch(train_loader, model, device, opt, reg=True, l2_lambda=0.0005)
    test_acc, test_loss = epoch(test_loader, model, device)


    plot_fig(acc, loss, test_acc, test_loss, startAt = args['start_epoch'], path='./progress_reg.png', jsonPath = './progress_reg.json', k=k)
    
    if (i) % 5 == 0:
        torch.save(model.state_dict(), "no_model_reg_{}.pt".format(i))
        # The saved model filename says the model is saved after completing those many epochs. For ex model_5.pt says 5 epochs are done
        # Now pass args['start_epoch'] = 5 (not 6) to continue the training.  
        print("Model saved")

    print("Epoch number:{}".format(i), *("{:.3f}".format(j) for j in (test_loss, test_acc)), sep="\t")

    # This is required to ensure the JSon file is not rewritten
    if not k == 1:
        k =1


