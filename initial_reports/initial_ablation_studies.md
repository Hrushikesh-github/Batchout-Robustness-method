# Experiments with BatchOut related Models

Few models were trained using Batchout and a modification of BatchOut. The main idea of Batchout is take into account the features of another class. The original batchout implementation takes the features of a random sample and then we compute the feature perturbation which is added to the original feature, this is called feature augmentation.  

There are broadly two sets of experiments, 

1. Implementing batchout paper

2. A modification of batchout, where batchout is applied only to the final layer but with few modifications

   

The results of 2nd set of experiments is written first and then results of my implementation of BatchOut.



## Final Layer Modified BatchOut Experiments

The major differences in the final layer batchout type are:

1. Perturbation for every image in the batch was done unlike batchout application which was done for only few images.
2. I passed the label value as well to ensure that the perturbation is done for a different class. the `_make_random`  does this. BatchOut does not guarantee this and there can be situation where the perturbation is w.r.t images of same class.

```python
class Batchout_New(nn.Module):
    def __init__(self, n):
        super(Batchout_New, self).__init__()
        self.n = n
	
    # A method to get list of random integers, such that the label is different 
    def _make_random(self, x, y):
        _r = np.random.randint(0, x.shape[0], x.shape[0])
        for i, r in enumerate(_r):
            if _r[i] == y[i]:
                while not _r[i] == y[i]:
                    _r[i] = np.random.randint(0, x.shape[0])

        return _r

    def forward(self, x, y):

        # Get integers from 0 to m, m is batch size  
        _r = self._make_random(x, y)
    
        # Obtain the feature randomly sampled from batch 
        _sample = x[_r]

        # Compute the direction of feature perturbation
        _d  = _sample - x
       
        # Augment and replace the 1st 'k' samples 
        x = x + (self.n * _d).detach()
        
        return x

```



For the above code to work, we need to pass both (x, y) to the model unlike model(x).

I applied this class only to the final layer of resnet model on CIFAR3 (only 3 classes of CIFAR10) and CIFAR10. This was not done while evaluating the model.

The resnet model had (9, 9, 9) bottleneck layers with filters learnt by starting layers: (16, 32, 64) respectively. This is the same architecture that was used before that was done in Tensorflow. There I obtained an accuracy of 91% on test set. Architecture can be visualized here /home/reshikesh/hrushikesh/dl4cv/conv/resnet.png (only difference is last layer has 4096 instead of 2048) or with torch_summary.

### CIFAR3

Took only the 1st three classes of CIFAR and trained the resnet model. Training accuracy was 94.94% and test accuracy was 90.66%. The json files, training progress, weights and code can be found in CIFAR3 folder.

Batchout based training gave 90.89 % training and 90.86% test accuracy. Note that both of them are trained till epoch 30. The *training graphs show that training was more volatile when we perturb*:

![progress_reg](/home/reshikesh/hrushikesh/robust/cifar3/perturbation/progress_reg.png) ![no_progress_reg (1)](/home/reshikesh/hrushikesh/robust/cifar3/standard/no_progress_reg (1).png)

Now comparing their robustness, 

![Screenshot from 2021-01-31 21-56-53](/home/reshikesh/hrushikesh/robust/cifar3/standard/Screenshot from 2021-01-31 21-56-53.png)

![Screenshot from 2021-01-31 21-43-31](/home/reshikesh/hrushikesh/robust/cifar3/perturbation/Screenshot from 2021-01-31 21-43-31.png)



We see that has perturbed model has around **7% greater robustness**.  For epsilon 5/255, standard model has 58% robust accuracy and perturbed model has 64% robust accuracy on CIFAR3. Batchout model, adversarially trained model and standard model at same epislon on CIFAR10 has robust accuracies 55%, 50% and 40% accuracy.



### CIFAR10

For CIFAR10, the same model was used. Previously I got 91% accuracy with the same model in tensorflow. I could have used the same learning rates but I didn't realize it when I trained the model.

Standard training training accuracy was 91.8% and test accuracy was **78.3 %** which is actually 5-6 % low than expected. I think it was because I decreased the learning rate a bit quickly. *But 78.3 % was close to accuracy mentioned by batchout paper, so I didn't train again*. 

Perturbed model has 93.9 % and **85.2 %** accuracy. Due to colab stopping suddenly, I didn't download the standard training monitor.

Now comparing the robustness, we see there is **15% increase in robustness**.



![Screenshot from 2021-02-02 22-04-37](/home/reshikesh/hrushikesh/robust/cifar/standard/Screenshot from 2021-02-02 22-04-37.png)



![Screenshot from 2021-02-02 22-04-48](/home/reshikesh/hrushikesh/robust/cifar/perturbation/Screenshot from 2021-02-02 22-04-48.png)

But as per Batchout for a Lenet type architecture, *at epsilon=5/255, accuracy was 55%, an adversarially trained model had 50% and standard model had 40%. However I obtained only 34% robust accuracy on resnet model and standard model got only 20% robust accuracy.* 

Does this show ResNet is not robust enough? As per the [Universal Adversarial Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) (Table-2) we see that a UAP on ResNet has the least effect on other architectures. ![Screenshot from 2021-02-03 08-15-45](/home/reshikesh/hrushikesh/Pictures/Screenshot from 2021-02-03 08-15-45.png)

This supports that ResNet are less robust than other models. ( Maybe because of addition, the perturbed input/feature still remains and is not completely removed. )

**Conclusions**: Applying batchout modified on final layer of Resnet on CIFAR gave some improvement. ResNet appears to be poor in robustness.







## BatchOut Paper Implementation Results

BatchOut was implemented with the following code:

```python
class Batchout(nn.Module):
    def __init__(self, k, n):
        super(Batchout, self).__init__()
        self.n = n
        self.k = k

    def forward(self, x):

        # Sample a random integer (with replacement) from 0 to m where m is batch size
        _r = np.random.randint(0, x.shape[0], x.shape[0])
    
        # Obtain the feature randomly sampled from batch 
        _sample = x[_r]

        # Compute the direction of feature perturbation
        _d  = _sample - x
       
        # Augment and replace the 1st 'k' samples 
        _number = int(self.k * x.shape[0])
        # Zero the perturbations that are beyond the 1st k samples. This is done to avoid
        # The following step which will give wrong gradients.
        #x[:_number] = x[:_number] + (self.n * _d[:_number]).detach()
        _d[_number:] = 0
        
        x = x + (self.n * _d).detach()
        
        return x

```

As per the architecture in the paper, implementation was done using CIFAR and MNIST.

## Results on CIFAR

The results I obtained were not close as mentioned in the paper. But there is some improvement with the standard model that I trained.

![cifar_batchout](/home/reshikesh/hrushikesh/batchout/new/cifar/cifar_batchout.png)

**Update**: After training various models, I realized that I have not trained the above models properly (Specifically, decreasing the learning rate too quickly). I obtained better results than the batchout paper in my vgg19 implementation but the architectures and other parameters are different. 

## Results on MNIST

MNIST data for epsilons 0.3 and 0.5![Screenshot from 2021-02-03 17-49-14](/home/reshikesh/hrushikesh/Pictures/Screenshot from 2021-02-03 17-49-14.png)MNIST data



![Screenshot from 2021-02-03 17-49-31](/home/reshikesh/hrushikesh/Pictures/Screenshot from 2021-02-03 17-49-31.png)epsilon: 0.3



![Screenshot from 2021-02-03 17-49-49](/home/reshikesh/hrushikesh/Pictures/Screenshot from 2021-02-03 17-49-49.png)epsilon: 0.5

For mnist, I got completely different results. Both of my standard models and batchout models were very poorly robust. One thing that confuses me is for epsilon=0.5, the images are very difficult to predict for human as well but as per paper both adversarially trained and proposed model achieve 95%+ accuracy





![Screenshot from 2021-02-03 09-50-45](/home/reshikesh/hrushikesh/batchout/new/mnist/batchout_results/Screenshot from 2021-02-03 09-50-45.png)

Batchout model giving very poor accuracy(up) and even the standard model as well(down) 



![Screenshot from 2021-02-03 09-10-21](/home/reshikesh/hrushikesh/batchout/new/mnist/standard_results/Screenshot from 2021-02-03 09-10-21.png)



I am not sure why I am getting such bad results.