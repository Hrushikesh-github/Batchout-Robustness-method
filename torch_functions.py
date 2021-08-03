# Contains functions used to train the models and implementation of PGD, FGSM, although foolbox was used

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def epoch(loader, model, device='cuda', opt=None, reg=False, l2_lambda=0.0001):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        if not opt:
            model.eval()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            if reg:    
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
        if opt:
            model.train()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            if reg:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return (1 - total_err / len(loader.dataset)), total_loss / len(loader.dataset)



# My Encoder class was necessary to prevent the JSON file to start and end with "
class MyEncoder(json.JSONEncoder):
    def default( obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def plot_fig(acc, loss, test_acc, test_loss, startAt = 0, path='./progress.png', jsonPath = './progress.json', k=0):
    #print('k value is: ', k)
    if os.path.exists(jsonPath):
        H = json.loads(open(jsonPath).read())

    # check to see if a starting epoch was supplied
    # We also set a condition k = 0 to ensure this 'if code' is executed only once, otherwise the entries will be trimmed

        if startAt > 0 and k == 0:
        #loop over the entries in the history log and trim any
        #entries that are past the starting point
            for k in H.keys():
                H[k] = H[k][:startAt]

    # If path does not exist, then we need to initialize the dictionary
    if not os.path.exists(jsonPath):
        H = {}
        log = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
        for l in log:
            H[l] = []

    H['accuracy'].append(acc)
    H['loss'].append(loss)
    H['val_accuracy'].append(test_acc)
    H['val_loss'].append(test_loss)

    # Open the json file and write(i.e replace) the updated dictionary. In case the file is not present, new file is created.
    f=open(jsonPath,"w")
    f.write(json.dumps(H, cls = MyEncoder))
    f.close()

    N=np.arange(0, len(H["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="val_loss")
    plt.plot(N, H["accuracy"], label="train_accuracy")
    plt.plot(N, H["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy [Epoch {}]".format(len(H["loss"])))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    #save the fig
    plt.savefig(path)
    plt.close('all')

# Function to run batchout
# Only difference is we pass the labels value to model(X, y)
def epoch_perturbation(loader, model, device='cuda', opt=None, reg=False, l2_lambda=0.0001):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        if not opt:
            model.eval()
            yp = model(X, y)
            loss = nn.CrossEntropyLoss()(yp,y)
            if reg:    
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
        if opt:
            model.train()
            yp = model(X, y)
            loss = nn.CrossEntropyLoss()(yp, y)
            if reg:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return (1 - total_err / len(loader.dataset)), total_loss / len(loader.dataset)

def epoch_perturbation_adversarial(loader, model, device, attack, epsilon, opt=None, reg=False, l2_lambda=0.0001):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        # We update our delta unlike the standard epoch and the train the model to be robust
        delta = epsilon * attack(model, X, y)

        if not opt:
            model.eval()
            yp = model(X + delta, y)
            loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            model.train()
            yp = model(X + delta, y)
            loss = nn.CrossEntropyLoss()(yp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()


        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, device, attack, epsilon, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        # We update our delta unlike the standard epoch and the train the model to be robust
        delta = epsilon * attack(model, X, y)

        if not opt:
            model.eval()
            yp = model(X + delta)
            loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            model.train()
            yp = model(X + delta)
            loss = nn.CrossEntropyLoss()(yp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()


        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Function to perform FGSM attack
def fgsm(model, X, y):
    model.eval()
    # Initialize the perturbation - delta, which would require gradient calculation
    delta = torch.zeros_like(X, requires_grad=True)
    # Compute the loss and backpropagate it to get the gradients
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()

    # delta = epsilon * gradient of loss w.r.t delta. 
    # We will multiply epsilon in the epoch functions
    return delta.grad.detach().sign() 

def fgsm_batchout(model, X, y):
    """ Construct FGSM adversarial examples on the examples X"""
    model.eval()
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta, y), y)
    loss.backward()
    return delta.grad.detach().sign() # We will multiply epsilon in the epoch functions

def pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = (yp[:,y_targ] - yp.gather(1,y[:,None])[:,0]).sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

# Function to generate PGD attack
def pgd(model, X, y, epsilon, alpha, num_iter):
    
    # Initialize delta which will have same dimensions as image/batch of images - X
    # required_grad is important to get gradient of loss w.r.t delta
    delta = torch.zeros_like(X, requires_grad=True)
    
    # Loop
    for t in range(num_iter):
	# Get the loss value when we add the delta 
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
	# This loss is backward computed to get the gradients
        loss.backward()
	# With the gradients, we get the value of delta 
	# delta += (step_size * gradients)
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)

	# Zero the gradients, otherwise we would add them up
        delta.grad.zero_()

    # Return the perturbation
    return delta.detach()

