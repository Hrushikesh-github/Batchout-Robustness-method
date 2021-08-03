## Report on Batchout 



Batchout is an augmentation technique in the feature space. The main aim of batchout is to make the model adversarially robust. Batchout is a simple linear convex combination between two datapoints

x = x~1~ + n * (x~2~ - x~1~) where x~1~, x~2~ are data-points and n is parameter. 

The technique is called batchout because the above augmentation takes place with datapoints belonging to the same batch. 

The original batchout paper implementation:

![image-20210611202006595](/home/reshikesh/.config/Typora/typora-user-images/image-20210611202006595.png)

Note that in 3rd line, it's a typo, we don't choose a single random sample but random samples. 

The main intuition of batchout is to push the decision boundary away from images making it more difficult to form adversaries. 

There are two primary changes made to the algorithm:

1. Batchout will be performed to all samples unlike original algorithm where only k% of samples undergo.
2. Batchout will be mutual.

Let us consider a simple scenario where there are three images: Cat, Dog and Frog. Assume due to randomness the cat image is perturbed with dog, the dog with frog and frog with cat. In the 1st layer of batchout, we see normal convex combination. But in the 2nd layer, the perturbed image of cat also has some perturbations from frog class. This appears to be against intuition.   

<img src="/home/reshikesh/diary/pics/b.jpeg" alt="WhatsApp Image 2021-05-10 at 9.03.00 PM" style="zoom:33%;" />

So the solution to this was thought to perturb the image of cat with dog and the dog with cat. The image of the frog will be perturbed with other image. Thus it is important for batchout to be mutual.

Using PyTorch, batchout can be applied by either:

1. Creating a new batchout class and change the forward class of model by adding the new class where we want. VGG models with simple forward function are best.
2. Using forward/pre-forward hooks. Models such as ResNet/DenseNet whose forward function is complex is best suitable.

It is better not to have batchout on initial layers due to convergence issues. The depth after which batchout will be applied was based on results obtained from "Feature space perturbation yields more transferable adversarial examples" paper. Most of the ablation studies were done using VGG19 and CIFAR10. The major conclusions are:

### Conclusions

- Applying batchout on all layers does not lead to convergence for large values of n such as 0.3. Thus it was decided to leave initial layers unperturbed. 
- Applying batchout on alternative layers rather than all the layers gave better results.
- `n = Random[-2, 0, 3]`  gave the best results. Here the values of n change for every batch. In the batchout paper,  n = U[0, 0.15] was frequently used where U is uniform distribution. 
- **The best PGD (200 steps) accuracy is 32.37%. The model gave 55.06% accuracy on PGD(10 steps).  37.62% FGSM accuracy.** However this model has very low accuracy 82.88%.
- Densenet and ResNet both of them failed to give any significant robust accuracies when batchout was applied on CIFAR10. 




### Few Points

Batchout is slightly different with mixup. We only take convex combination w.r.t images only. Intuitively in my opinion both are acceptable. Mixup tries to make smoother transition. Batchout tries to push the decision boundary away.  

1. Usually value of n choosen was from 0.2 or 0.3.  I applied negative values of -0.2 and -0.3 and the results were surprisingly good but not as good as 0.2 or 0.3. Why did I apply negative values? It was because of some misunderstanding why I was going through TRADES paper. Below we can see alt_-2 and  alt_-3  gave 27% and 19% respectively, better than the standard model 6.5% at epsilon 8/255. The values in "[]" refers to accuracies from 0 to 8/255 epsilon and value in "()" is accuracy  at l2 FGSM attack. 

   ![7](/home/reshikesh/diary/pics_paper/7.png)

   I tried a random combination of +3 and -2, but the results were somewhere in between as can be seen in random_3_-2 and random_3_-2_0.

2. I applied the batchout attack specifically on a pair of classes and implemented PGD targeted attack. The accuracy was poor here as well perhaps because our method is not strong with PGD.

3. The original implementation of batchout did not ensure that the two images that will be perturbed will be of same class. I applied batchout in both the cases, when it is ensured both classes are of same class and when it is not ensured. I see that strangely when it is not ensured, I achieve better accuracy(6% better accuracy). This can give some insight such as: **The linear space between images/features of same class is not necessarily classified as the class?** I tried to check whether mixup ensures both classes are different, but I didn't get any info.

4. Batchout was applied to all layers after a specific layer. As we can see in above figure, it didn't give the best result and also higher values of n gave poorer result.

5. [Early stopping](https://github.com/locuslab/robust_overfitting) did give good results some time. But perhaps more experiments should be done to check whether this works (This is not related to batchout)

I used to think batchout or Mixup fail with PGD can perhaps be expalined by a paper Tramer et al(2017). I have written more about this later in other file. The gist is that Tramer et al finds around 25 directions in which Adversarial examples exist for MNIST. This number I think will be larger as dataset size increases. But  Batchout or Mixup can at max consider 9 direction (each direction in the direction of other class) for CIFAR10. Perhaps this is the main reason why it is giving good (and sometimes better than PGD training) on FGSM but poor on PGD.



### Papers and Resources Followed

https://adversarial-ml-tutorial.org/  for hands-on Introduction

#### Papers

[Deep neural networks are easily fooled: High confidence predictions for unrecognizable images](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.html) This paper uses Genetic algorithm to create images that fool deep nets to 99% accuracy. Nothing important but gives an idea that deep nets can be fooled to high accuracy. 

[Intriguing properties of neural networks](https://arxiv.org/pdf/1312.6199.pdf)  The first paper to introduce Adversarial examples. Generates AE using some Newton Optimizer method

[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572) Introduces FGSM and FGSM based AT

[Universal adversarial perturbations](https://arxiv.org/abs/1610.08401) 

The papers from [Madry's Lab](http://madry-lab.ml/) and [Zico Kolter's Lab](http://zicokolter.com/publications/) are excellent and top rated. 

Papers such as "[Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175)", "[PGD AT](https://arxiv.org/pdf/1706.06083.pdf)" from Madry's Lab are important, they also maintain a blog https://gradientscience.org/. "[Fast is better than Free](https://arxiv.org/pdf/2001.03994.pdf)", "[Certified Defence via Randomized Smoothing](https://arxiv.org/pdf/1902.02918.pdf)", "Overfitting in Adversarial Domain" papers from zico's lab are necessary. [This paper](https://proceedings.neurips.cc/paper/2020/file/b8ce47761ed7b3b6f48b583350b7f9e4-Paper.pdf) however claims there are some shortcomings and other advantages of Fast is better than Free paper.

[Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420)

[Adversarial Logit Pairing](https://arxiv.org/abs/1803.06373) This paper from Goodfellow is very good but it's submission was retracted from NeurIPS (not sure why). The paper proposes a new method to make models robust. However a subsequent paper was able to break the defence.

[TRADES](https://arxiv.org/pdf/1901.08573.pdf) One of the most important papers that breaks away from min-max optimisation technique. The paper is also very interesting to read.

[Unlabeled Data improves AR](https://arxiv.org/pdf/1905.13736.pdf) showed great improvement in robustness.

[The space of adversarial transferability](https://arxiv.org/pdf/1704.03453.pdf) I never actually read the paper completely because it is mathematically heavy. I feel something very important related to adversarial directions is discussed. 

[Auto Attack](https://github.com/fra31/auto-attack) (ensemble of 4 attacks) is the most commonly used benchmark

[SOTA](https://arxiv.org/pdf/2010.03593.pdf) no new algorithm is proposed. Just extensive experiments are performed and new best is obtained.

#### Youtube Videos:

https://www.youtube.com/watch?v=aK7DJ8XCWCI&t=447s Konda Reddy Sir's Introduction

https://www.youtube.com/watch?v=4rFOkpI0Lcg Intro to Adversarial Examples by Arxiv Insights

https://www.youtube.com/watch?v=df_NZyGeVXg&t=1s Lipschitz constant explaination from Intriguing properties of Neural Networks paper.

https://www.youtube.com/watch?v=CIfsB_EYsVI&t=1883s GoodFellow's Stanford Lecture wrote about it 

https://www.youtube.com/watch?v=UHs2mGBH0Fg Zico Kolter's Lecture on Randomized Smoothing

https://www.youtube.com/watch?v=fzusr-VdPxw&t=385s Talk from author on Bugs not Feature paper.

https://www.youtube.com/watch?v=hMO6rbMAPew Yannic Kilcher on Features not Bugs  Paper



