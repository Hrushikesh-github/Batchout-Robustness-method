import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import foolbox as fb
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

test_X, test_Y = next(iter(train_loader)) # Get 32 images/classes
test_X = test_X.to(device)

# Get the accuracy for all the images, so go through each batch
for batch_images, batch_labels in test_loader:
    batch_images = batch_images.to(device)
    # Loop through each image in the batch
    mean_error = 0.0
    count_error = 0.0
    for image, label in zip(batch_images, batch_labels):
        class_predictions = []
        predictions = torch.zeros(32, 10)
        # Loop through the images that we will perform batchout with the image
        for i, x in enumerate(test_X):
            
            # Pair the images to get a 4d tensor
            images = torch.stack((image, x))
            # Obtain the batchout prediction
            yp = inf_model(images)

            # Obtain the predicted class and append it to a list
            pred_class = classes[int(yp[0].argmax())]
            pred_class = yp[0].max(dim=0)[1]
            pred_class = int(pred_class.cpu().numpy())
            class_predictions.append(pred_class)

            # Store the predictions which will be later averaged
            predictions[i] = yp[0]

        # Get the mean predictions and class labels after batchout with all
        mean_predictions = predictions.mean(0)
        mean_pred_class = int(mean_predictions.max(dim=0)[1])
        
        count = Counter(class_predictions)
        common_pred_class = count.most_common()[0][0]

        count_error += common_pred_class != int(label)
        mean_error += mean_pred_class != int(label)

    print(count_error * 100 / 32, mean_error * 100 / 32)


