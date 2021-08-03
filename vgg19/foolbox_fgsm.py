import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import foolbox as fb
from foolbox.attacks import FGSM
import sys
sys.path.append("/home/hrushikesh/robust/vgg19/standard")
from vgg_change import VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} ".format(device))

batch_size = 250

test_transforms = transforms.Compose([transforms.ToTensor()])
cifar_test = datasets.CIFAR10("/home/hrushikesh/torch/data", train=False, download=True, transform=test_transforms)
test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=False)

#prefix = '/home/hrushikesh/robust/vgg19/batchout_all/results/middle_all/model_reg_'
model_paths = ['/home/hrushikesh/robust/vgg19/batchout_all/results/middle_all/model_reg_94.pt', '/home/hrushikesh/robust/vgg19/batchout_many/random_n/results/m=9/model_reg_123.pt']
#model_paths = [prefix + x for x in model_paths]
preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], axis=-3) # axis = -3 refers to the channels class. channels is generally 3rd from back
#epsilons = [0.0, 1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255,  8/255]
epsilons = [0.0, 0.5]
attack = fb.attacks.L2FastGradientAttack()
model = VGG()

for path in model_paths:
    print("Model at {} loaded".format(path))
    model.load_state_dict(torch.load(path))
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing) # bounds are before the preprocessing
    accs = []
    r_accs = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        acc = fb.utils.accuracy(fmodel, images, labels)
        
        raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
        robust_acc = (1 - is_adv.float().mean(axis=-1)) * 100
        robust_acc = robust_acc.cpu().numpy()
        
        accs.append(acc)
        r_accs.append(robust_acc)
    
    print('acc: ', np.mean(accs))
    #print('errors: ', np.average(r_accs , axis=0, weights=[0.3, 0.3, 0.3, 0.1]))
    print('errors: ', np.average(r_accs , axis=0))

