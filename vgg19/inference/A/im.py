import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import foolbox as fb
import progressbar
from collections import Counter
import matplotlib.pyplot as plt
from vgg_inference_model import VGG
import sys
sys.path.append("/home/reshikesh/hrushikesh/robust/vgg19/standard")
from vgg_change import VGG as AGG

batch_size = 31
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

cifar_train = CIFAR10("~/data", train=True, download=True, transform=transforms.ToTensor())
cifar_test = CIFAR10("~/data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=False)


classes = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the inference model
inf_model = VGG(0.3).to(device)
inf_model.load_state_dict(torch.load("/home/reshikesh/hrushikesh/robust/vgg19/batchout_many/n_3/model_reg_121.pt"))
inf_model.eval()

'''
# Create the foolbox model based on standard vgg. We use this to create adversarial images
preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], axis=-3)
model = AGG()
model.eval()
model.load_state_dict(torch.load("/home/reshikesh/hrushikesh/robust/vgg19/batchout_many/n_3/model_reg_121.pt"))
fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)
'''

# So there are two models now: Inference model which accepts two images and performs batchout and
# Foolbox model that will be used to get adversarial images

# initialize the progress bar
widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(test_loader.dataset) / 32, widgets=widgets).start()

perturb_X, perturb_Y = next(iter(train_loader)) # Get 32 images/classes
perturb_X = perturb_X.to(device)

# Get the accuracy for all the test images, so go through each batch
total_count_error = 0.0
total_mean_error = 0.0

for i, (batch_images, batch_labels) in enumerate(test_loader):
    batch_images = batch_images.to(device)

    mean_error = 0.0
    count_error = 0.0
    total_err = 0.0
    # Loop through each image in the current batch
    for image, label in zip(batch_images, batch_labels):
        class_predictions = []
        predictions = torch.zeros(32, 10)
        C = torch.zeros(32, 3, 32, 32).to(device)
        C[0] = image
        C[1:] = perturb_X
        
        # Obtain predictions, the 1st prediction will be of the image, the others will be predictions by perturbing
        pred = inf_model(C)

        # Get the most common prediction after perturbing
        count = Counter(pred.max(dim=1)[1].cpu().numpy()) # An array of shape 32
        common_pred_class = count.most_common()[0][0]

        # Add +1`if wrongly predicted
        count_error += common_pred_class != int(label)

        # Get the mean of predictions and update the mean error by adding +1 if mean predicts wrongly
        mean_pred = predictions.mean(0) # Returns array of (1, 10)
        mean_class = mean_pred.max(dim=0)[1].item()
        mean_error += (mean_class != label.item())
        #print(i)

        

    #print(count_error * 100 / 32, mean_error * 100 / 32)
    pbar.update(i)
    total_count_error += count_error
    total_mean_error += mean_error

print(total_count_error * 100 / len(test_loader.dataset), total_mean_error * 100 / len(test_loader.dataset))
pbar.finish()

