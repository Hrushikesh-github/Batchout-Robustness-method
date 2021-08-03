# Batchout - Adversarial Robustness Method, same time as standard training time

Batchout is a method to make a model robust to adversarial attacks. Using this method, **accuracy of 32.37% was achieved on VGG19 with a PGD (200 steps)** attack. The training takes the **same time as standard training time**. However, the defense could not be replicated on ResNet and DenseNet. 

Batchout is an augmentation technique in the feature space. The main aim of batchout is to make
the model adversarially robust. Batchout performs a simple linear convex combination between the features of two datapoints. 

x = x~1~ + n * (x~2~ - x~1~) where x~1~, x~2~ are values of features of data-points and n is a parameter. The method is slightly similar to Mixup, major difference is the convex combination takes place in the feature space, this combination can be applied across various feature spaces and no such augmentation happens among the class variable.

The current repo is an extension of the [original paper](http://dl.acm.org/citation.cfm?id=3293387). The original paper, released in 2017 proved it was better than A.T(trained using FGSM) on FGSM and DeepFool attacks (At that time PGD attack was not present). 

![image](https://user-images.githubusercontent.com/56476887/128053830-e082247d-0c88-48c8-a5fe-2ace7f10a637.png)

Typo in the 2nd line of the algorithm, we don't choose a single random sample.  

![image](https://user-images.githubusercontent.com/56476887/128053738-a3f6317b-45cf-4163-9771-4fddb4b06b31.png)

The main intuition of batchout is to push the decision boundary away from images making it more difficult to form adversaries. 


There are two primary changes made to the algorithm:

1. Batchout will be performed to all samples unlike original algorithm present in original paper where only k% of samples undergo.
2. Batchout will be mutual among two images.

Let us consider a simple scenario where there are three images: Cat, Dog and Frog. Assume due to randomness the cat image is perturbed with dog, the dog with frog and frog with cat. In the 1st layer of batchout, we see normal convex combination. But in the 2nd layer, the perturbed image of cat also has some perturbations from frog class, thus creating unnecessary perturbation. This also appears to be against intuition.   
![image](https://user-images.githubusercontent.com/56476887/128055785-1d9951fa-7c17-42ed-bd53-d235e0ab97b2.png)

### Conclusions

- Applying batchout on all layers does not lead to convergence for large values of n such as 0.3. Thus it was decided to leave initial layers unperturbed. 
- Applying batchout on alternative layers rather than all the layers gave better results.
- `n = Random[-.2, 0, .3]`  gave the best results. Here the values of n change for every batch. In the batchout paper,  n = U[0, 0.15] was frequently used where U is uniform distribution. 
- **The best PGD (200 steps) accuracy is 32.37%. The model gave 55.06% accuracy on PGD(10 steps).  37.62% FGSM accuracy.** However this model has very low accuracy 82.88%.
- Densenet and ResNet both of them failed to give any significant robust accuracies when batchout was applied on CIFAR10. 

### Code implementation

Batchout can be implemented in PyTorch either using Hooks or chaning the class module of the model. I used hooks for ResNet and DenseNet, since they had slightly complicated class implementation and modified the code for VGG19.

### Repo Contents
- torch_functions : Helper file contains functions to train model as well as FGSM, PGD attacks although Foolbox was used.
- vgg19 : Contains batchout implementation, various ablation studies and results on VGG19
- torch_hooks : Batchout implementation on ResNet and DenseNet
- terminology : Since it is a novel work, certain terminologies may not be understood, thus a detailed explaination is present here. 
