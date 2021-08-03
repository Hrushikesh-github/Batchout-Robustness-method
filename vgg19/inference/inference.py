import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import foolbox as fb
import matplotlib.pyplot as plt
from vgg_inference_model import VGG
import sys
sys.path.append("/home/reshikesh/hrushikesh/robust/vgg19/standard")
from vgg_change import VGG as AGG
from PIL import Image

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

cifar_train = datasets.CIFAR10("~/data", train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10("~/data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=True)


classes = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the inference model
inf_model = VGG(0.3).to(device)
inf_model.load_state_dict(torch.load("/home/reshikesh/hrushikesh/robust/vgg19/batchout_many/n_3/model_reg_121.pt"))
inf_model.eval()

# Create the foolbox model based on standard vgg. We use this to create adversarial images
preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], axis=-3)
model = AGG()
model.eval()
model.load_state_dict(torch.load("/home/reshikesh/hrushikesh/robust/vgg19/batchout_many/n_3/model_reg_121.pt"))
fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

# So there are two models now: Inference model which accepts two images and performs batchout and
# Foolbox model that will be used to get adversarial images

# Choose some random image from test set 
img = Image.open("/home/reshikesh/cifar10_fastai/test/airplane/0602.png")
img_class = torch.tensor([0]) # Airplane class is 0
img = transforms.ToTensor()(img)[None, :, :, :]
img = img.to(device)
img_class = img_class.to(device)

predictions = []
total_err = 0.

yp = inf_model(img)
print(yp)
#print(yp.max(dim=1)[1] != img_class)
print(nn.Softmax(dim=1)(yp).max())
print(classes[int(yp.argmax())])

print("----------------------------------------------------------")

X,y = next(iter(test_loader))
for x in X:
    x = x.to(device)
    img = img.squeeze()
    images = torch.stack((img, x))
    yp = inf_model(images)
    total_err += (yp[0].max(dim=1)[1] != img_class).sum().item()
        
print("Error rate is: ", total_err * 100 / 32)

'''
# Perform the same with FGSM adversarial image
attack = fb.attacks.FGSM()
raw, clipped, is_adv = attack(fmodel, img, img_class, epsilons=[8/255])
clipped_img = clipped[0]
#print(type(clipped_img), clipped_img.shape)

yp = inf_model(clipped_img)
print(yp)
#print(yp.max(dim=1)[1] != img_class)
print(nn.Softmax(dim=1)(yp).max())
print(classes[int(yp.argmax())])

predictions = []
total_err = 0.
X,y = next(iter(test_loader))
for x in X:
    x = x.to(device)
    cl_img = clipped_img.squeeze()
    images = torch.stack((cl_img, x))
    yp = inf_model(images)
    total_err += (yp.max(dim=1)[1] != img_class).sum().item()
        
print("Error rate is: ", total_err * 100 / 32)
'''

'''
# Perform the same with PGD adversarial image
attack = fb.attacks.PGD()
raw, clipped, is_adv = attack(fmodel, img, img_class, epsilons=[8/255])
clipped_img = clipped[0]
print("is_adv", is_adv)
#print(type(clipped_img), clipped_img.shape)

yp = inf_model(clipped_img)
print(yp)
#print(yp.max(dim=1)[1] != img_class)
print(nn.Softmax(dim=1)(yp).max())
print(classes[int(yp.argmax())])

predictions = []
counter = 0
total_err = 0.
X,y = next(iter(test_loader))
for x in X:
    counter += 1
    x = x.to(device)
    cl_img = clipped_img.squeeze()
    images = torch.stack((cl_img, x))
    yp = inf_model(images)
    #print(yp)
    total_err += (yp[0].max(dim=0)[1] != img_class).sum().item()
        
print("COUNTER: ", counter)
print("Error rate is: ", total_err * 100 / 32)
'''

