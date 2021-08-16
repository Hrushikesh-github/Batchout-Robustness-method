import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import sys
from torch_functions import *
import argparse

'''The Batchout_pre_hook function wil be used as a pre-forward hook and performs the batchout operation'''
def batchout_pre_hook(self, input):

    # THe value of paprameter n can be from a distribution or can have a fixed value (such as 0.2)
    n_random = np.random.choice([0.3, -0.2, 0], p=[0.6, 0.3, 0.1])
    #n = 0.2

    # In case the model is in eval mode, don't perform batchout
    if not self.training:
        # n=0 ensures standard training/evaluation. 
        n = 0.0

    # Input would be tuple of size 1. To obtain the tensor, we need to access
    x = input[0]

    # Batchout operation will be performed for the first and last pair, second and second-last pair etc
    # It must be ensured that for every epoch, the batch is shuffled to ensure efficient training.
    _r = np.linspace(x.shape[0]-1, 0, x.shape[0], dtype=int)

    # Obtain the feature sampled from batch
    _sample = x[_r]

    # Compute the direction of feature perturbation
    _d  = _sample - x

    # Augment and replace the 1st 'k' samples
    x = x + (n * _d).detach()

    return x

''' Defining the VGG model '''
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features1 = self._make_layers(3, 64)
        self.features2 = self._make_layers(64, 64)
        self.features3 = self._make_layers(64, 'M')
        self.features4 = self._make_layers(64, 128)
        self.features5 = self._make_layers(128, 128)
        self.features6 = self._make_layers(128, 'M')
        self.features7 = self._make_layers(128, 256)
        self.features8 = self._make_layers(256, 256)
        self.features9 = self._make_layers(256, 'M')
        self.features10 = self._make_layers(256, 512)
        self.features11 = self._make_layers(512, 512)
        self.features13 = self._make_layers(512, 512)
        self.features14 = self._make_layers(512, 'M')
        self.features12 = self._make_layers(512, 'M')
        self.classifier = nn.Linear(512, 10)
        self._initialize_weights()


    def forward(self, x):
        x = self.features1(x) #0
        x = self.features2(x) #1
        x = self.features3(x) #2 R
        x = self.features4(x) #3 0
        x = self.features5(x) #4 1
        x = self.features6(x) #5 R
        x = self.features7(x) #6 2
        x = self.features8(x) #7 3
        x = self.features8(x) #8 4
        x = self.features8(x) #9 5
        x = self.features9(x) #10 R

        x = self.features10(x) #11 6


        x = self.features13(x) #12
        x = self.features11(x) #13


        x = self.features13(x) #14
        x = self.features12(x) #15 R


        x = self.features13(x) #16
        x = self.features11(x) #17


        x = self.features13(x) #18
        x = self.features11(x) #19


        x = self.features14(x) #20 R
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, in_channels, cfg):
        layers = []
        if cfg == 'M':
            # Apply max pooling to reduce dimension
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Apply conv, there will be no reduction in dimensions
            layers += [nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm2d(cfg)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


''' Obtaining the Data and Training the model  '''
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

print(args)

batch_size = 32

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

aug = transforms.RandomChoice((transforms.RandomHorizontalFlip(p=1), transforms.RandomCrop(32, padding=4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
    transforms.RandomAffine(degrees=0, translate=(0, 0.1))))

train_transforms = transforms.Compose([aug, transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])

cifar_train = datasets.CIFAR10("~/data", train=True, download=True, transform=train_transforms)
cifar_test = datasets.CIFAR10("~/data", train=False, download=True, transform=test_transforms)

train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

model = VGG()
model.to(device)

''' We can register our forward pre hooks that will perform Batchout operation of convex combination '''
model.features13.register_forward_pre_hook(batchout_pre_hook)
model.features14.register_forward_pre_hook(batchout_pre_hook)

opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

# We need to initalize k=0 for correct plotting
k = 0

if args['model']:
    model.load_state_dict(torch.load(args['model']))
    print("Model loaded.. here we go")

print('Started Training')


for i in range(args['start_epoch'] + 1, args['start_epoch'] + 36):

    acc, loss = epoch(test_loader, model, device, opt, reg=True)
    test_acc, test_loss = epoch(test_loader, model, device)

    # A function to plot the accuracies and losses as well as store them in a JSON file
    plot_fig(acc, loss, test_acc, test_loss, startAt = args['start_epoch'], path='./progress_reg.png', jsonPath = './progress_reg.json', k=k)

    print("Epoch number:{}".format(i), *("{:.3f}".format(j) for j in (test_loss, test_acc)), sep="\t")

    if (i) % 5 == 0:
        torch.save(model.state_dict(), "model_reg_{}.pt".format(i))
        print("Model saved")


    # This is required to ensure the JSON file is not rewritten
    if not k == 1:
        k =1


