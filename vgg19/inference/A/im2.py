import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import numpy as np
import foolbox as fb
import progressbar
from collections import Counter
import matplotlib.pyplot as plt
from vgg_inference_model import VGG
import sys
sys.path.append("/home/reshikesh/hrushikesh/robust/vgg19/standard")
from vgg_change import VGG as AGG

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

f = transforms.ToTensor()
cifar_train = CIFAR10("~/data", train=True, download=True, transform=f)
cifar_test = CIFAR10("~/data", train=False, download=True, transform=f)

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
pbar = progressbar.ProgressBar(maxval=len(test_loader.dataset) / 30, widgets=widgets).start()

dataset = CIFAR10('~/data', train=True, transform=transforms.ToTensor(), download=True)

# Obtain some samples, say 30 or 40 to perturb with images. 
#train_label = np.array(dataset.targets)
y = np.array(dataset.targets)
print("y shape is: ", y.shape)

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
# Obtain index such that they have equal number of classes. This is done by choosing columns
subset_idx = class_idx[:, 5:8].reshape(-1, 1).squeeze()

su_dataloader = DataLoader(dataset=Subset(dataset, subset_idx), batch_size=30, shuffle=False)
perturb_X, perturb_Y = next(iter(su_dataloader))
perturb_X = perturb_X.to(device)
n_samples = perturb_X.shape[0] + 1

'''
train_data = dataset.data
perturb_X = train_data[subset_idx].squeeze()
perturb_X = torch.tensor(perturb_X).to('cuda')
n_samples = perturb_X.shape[0] + 1
perturb_Y = y[subset_idx].squeeze()
perturb_Y = torch.tensor(perturb_Y).to('cuda')
'''

print("perturb_X shape is: ", perturb_X.shape)
print("perturb_Y shape is: ", perturb_Y.shape)
# Initialize a tensor that will store current image in 1st index 
C = torch.zeros(n_samples, 3, 32, 32).to(device)
C[1:] = perturb_X

# Initialize a tensor that will store predictions
predictions = torch.zeros(n_samples, 10)

# Get the accuracy for all the test images, so go through each batch
total_count_error = 0.0
total_mean_error = 0.0

count_error_list, mean_error_list = [], []
# loop through each batch
for i, (batch_images, batch_labels) in enumerate(test_loader):
    batch_images = batch_images.to(device)

    mean_error = 0.0
    count_error = 0.0
    total_err = 0.0
    # Loop through each image in the batch
    for image, label in zip(batch_images, batch_labels):
        class_predictions = []
        C[0] = image
        #print("Shape of C: ", C.shape)
        
        # Obtain predictions, the 1st prediction will be of the image, the others will be predictions by perturbing
        pred = inf_model(C)
        #print("Pred shape is: ", pred.shape) # (31, 10)

        # Get the most common prediction after perturbing n_samples
        count = Counter(pred.max(dim=1)[1].cpu().numpy()) # An array of shape 32
        common_pred_class = count.most_common()[0][0]
        # Add +1`if wrongly predicted
        count_error += common_pred_class != int(label)

        # Get the mean of predictions and update the mean error by adding +1 if mean predicts wrongly
        mean_pred = pred.mean(0) # Returns array of (10)
        #print("mean_pred shape is: ", mean_pred.shape) 
        mean_class = mean_pred.max(dim=0)[1].item()
        #print("mean_class is: ", mean_class)
        mean_error += (mean_class != label.item())

        #print("Image class is {}, Common predicted class is {}, Mean predicted class is {}".format(label.item(),common_pred_class, mean_class))
        

    #print(count_error * 100 / batch_size, mean_error * 100 / batch_size)
    pbar.update(i)
    total_count_error += count_error
    total_mean_error += mean_error
    count_error_list.append(count_error)
    mean_error_list.append(mean_error)
    '''
    if i ==10:
        print(count_error_list)
        print(mean_error_list)
        print(total_count_error)
        print(total_mean_error)
        break
    '''

#print(total_count_error * 100 / (11 * 32), total_mean_error * 100 / (11 * 32))
print(total_count_error * 100 / len(test_loader.dataset), total_mean_error * 100 / len(test_loader.dataset))
#pbar.finish()

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

n_bins = 5
n, bins, patches = plt.hist(A, n_bins, facecolor='red', alpha=0.5)
n2, bins2, patches2 = plt.hist(B, n_bins, facecolor='yellow', alpha=1)
plt.xlabel("Number of errors")
plt.show()

