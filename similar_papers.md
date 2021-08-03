																																						17/3

## Few Papers related to Batchout

### Mixup: BEYOND EMPIRICAL RISK MINIMIZATION (ICLR - 2018)

Mixup is quite similar to batchout. It is a form of regularization (data augmentation. Mixup trains
a neural network on convex combinations of pairs of examples and their labels. The paper claims to improve generalization, AR, GAN training stability and reduces the memorization of corrupt labels.
$$
x̃ = λ{x}_i + (1 − λ)x_j  \\

ỹ = λ{y}_i + (1 − λ){y}_j
$$
The major difference is that even the linear combination of 'y' is also done and this is only done in the input space. The following figure is related to a toy example. We see a **smooth transition**, light blue indicates lesser probability. 

![3](/home/reshikesh/hrushikesh/diary/pics_paper/1.png)

The following figures were obtained on Cifar10. The figure on left shows an improvement in prediction(prediction refers to the predicted class). As lambda moves from 0(original image) to 1(selected image), the %miss (where a "miss" refers to a prediction when the class does not belong to both the original or selected image) decreases. For me, the presences of a miss itself is surprising because how can other classes come in between the convex combination of two classes? It's possible nonetheless because deep-nets are piecewise linear (with relatively few pieces).

![4](/home/reshikesh/hrushikesh/diary/pics_paper/2.png)

The figure on right talks about the gradient magnitude w.r.t image for differnet lambda. The smaller the norms, the better it is for adversarial robustness.



### ManiFold Mixup Better Representations by interpolating Hidden States(ICML 2019)

This paper is very much similar to the above paper, except that it also considers assigning mixup in the feature space. Paper claims: Manifold Mixup improves the hidden representations and decision boundaries of neural networks at multiple layers,  improves strong baselines in supervised learning, robustness
to **single-step adversarial attacks**.  Here they have specifically mentioned robustness to single-step AA such as FGSM, I-FGSM 

*They apply mixup to only one layer and this layer is choosen randomly (includes input layer as well).*

On a toy dataset, manifold mixup does the following:

![4](/home/reshikesh/hrushikesh/diary/pics_paper/4.png)

Paper claims that Manifold Mixup Flattens Representations. More specifically, this flattening reduces the number of directions with significant variance (akin to reducing their number of principal components). A lot of Maths and proofs are shown regarding flattening. The concept of Flattening was actually not explained properly in the paper, this was also pointed out in OpenReview.

![5](/home/reshikesh/hrushikesh/diary/pics_paper/5.png)

![3](/home/reshikesh/hrushikesh/diary/pics_paper/3.png)

It is seen that by perturbing in the feature space, there is some improvement. Also surprising is that it has surpassed Madry et Al(PGD training) on FGSM accuracy.  This is what is written in paper:

*While we found that using Manifold Mixup improves the robustness to single-step FGSM attack (especially over Input Mixup), we found that Manifold Mixup did not significantly improve robustness against stronger,
multi-step attacks such as PGD* 

I have seen in various places that some models which have very good accuracy on FGSM might fail in PGD, for example a model had 90% FGSM accuracy but 0 % PGD whereas in contrast Madry et Al model has 56% FGSM accuracy and around 44% PGD.  Another thing to note from here is, in my batchout experiments I achieved accuracy of 52% on FGSM and around 2-3% on PGD (standard model-0%). 

A lot of questions where asked in [openreview](https://openreview.net/forum?id=rJlRKjActQ&noteId=Bkg5lrwV0X) and might give better explaination of what is flattening, manifolds etc. 



### MIXUP INFERENCE (ICLR 2020)

This paper applies mixup in inference phase (Called MI) and claims that MI can further improve the adversarial robustness for the models trained by mixup and its variants. If we want to classify a image x on classifier f, we do f(x) but in this method we need to do f(x, a) where 'a' is a batch of images and 'b' corresponding class. Then mixup is applied between x and 'a' with a suitable value of 'alpha' the convex combination parameter. Average of the results are taken as prediction. The paper is short but the paper was slighlty difficult to read because of excessive use of symbols. The good thing about the method is that it can be used on any model.

![6](/home/reshikesh/hrushikesh/diary/pics_paper/6.png)

[*Interpolated*](https://arxiv.org/pdf/1906.06784.pdf) here refers to a method that was proposed by the authors Manifold Mixup. I haven't read it but it looks similar to mixup. Also this [link](https://openreview.net/forum?id=ByxtC2VtPB&noteId=SkeFK-RuYS) should be checked that gives a list of interpolation related papers. There are various other papers on mixup as well that I found by checking the result of mixup in OpenReview.



###  Batchout

Batchout is slightly different with mixup. We only take convex combination w.r.t images only. Intuitively in my opinion both are acceptable. Mixup tries to make smoother transition. Batchout tries to push the decision boundary away. I am confident that by training on ResNet, the accuracy reported in Mixup(71%) and and perhaps Manifold Mixup(77%) can be achieved (Currently I have 53% with VGG). 

#### Experiements Conducted

1. Usually value of n choosen was from 0.2 or 0.3. This time I applied negative values of -0.2 and -0.3 and the results were surprisingly good but that as good as 0.2 or 0.3. Why did I apply negative values? It was because of some misunderstanding why I was going through TRADES paper. Below we can see alt_-2 and  alt_-3  gave 27% and 19% respectively, better than the standard model 6.5% at epsilon 8/255. The values in "[]" refers to accuracies from 0 to 8/255 epsilon and value in "()" is accuracy  at l2 FGSM attack. 

   ![7]( /home/reshikesh/diary/pics_paper/7.png)

   I tried a random combination of +3 and -2, but the results were somewhere in between as can be seen in random_3_-2 and random_3_-2_0. But all these models fail with PGD.

2.  I applied the batchout attack specifically on a pair of classes and implemented PGD targeted attack. The accuracy was poor here as well perhaps because our method is not strong with PGD.the 

3. The original implementation of batchout did not ensure that the two images that will be perturbed will be of same class. I applied batchout in both the cases, when it is ensured both classes are of same class and when it is not ensured. I see that strangely when it is not ensured, I achieve better accuracy(6% better accuracy). This can give some insight such as: **The linear space between images/features of same class is not necessarily classified as the class?** I tried to check whether mixup ensures both classes are different, but I didn't get any info, will need to look into it.

4. Batchout was applied to all layers after a specific layer. As we can see in above figure, it didn't give the best result and also higher values of n gave poorer result.

5. [Early stopping](https://github.com/locuslab/robust_overfitting) did give good results some time. But perhaps more experiments should be done to check whether this works (This is not related to batchout)



Why I think batchout or Mixup fail with PGD can perhaps be expalined by a paper Tramer et al(2017). I have written more about this later in other file. The gist is that Tramer et al finds around 25 directions in which Adversarial examples exist for MNIST. This number I think will be larger as dataset size increases. But  Batchout or Mixup can at max consider 9 direction (each direction in the direction of other class) for CIFAR10. Perhaps this is the main reason why it is giving good (and sometimes better than PGD training) on FGSM but poor on PGD.



### Layer-wise Adversarial Training Approach to Improve Adversarial Robustness

AT is designed to only manipulate input. Since intermediate layers also play an important role, layer perturbation is introduced by this method, i.e for each layer there will be a separate perturbation to the input it recieves from the previous layer.  

![10](/home/reshikesh/diary/pics_paper/10.png)

R~conv~  will be slightly different because of the presence of the convolution operator.

Few pictures from the paper:

![8](/home/reshikesh/diary/pics_paper/8.png)

This is for a FC layer. For convolution layer it will be slightly different because we take the convolution operator instead of simple dot product. 

![9](/home/reshikesh/diary/pics_paper/9.png)

Note that for the 1st layer equation (8) in the above figure is nothing but the perturbation due to FGSM adversary. 

**Training procedure**:

1. A batch of images is sent and forward pass is done. In the forward pass the values R~fc~  and R~conv~ is added for each FC layer and conv layer respectively. 
2. Backward pass is done. Here the values of R~fc~ and R~conv~ are updated as well.
3. Process repeats

The results show positive result. The best positive result is when the perturbations are added to all the layers (including the starting layer).

Few points:

1. This is like FGSM training done cumulatively on each layer. Few of the methods that improved FGSM training in the past are Random initializations and methods used in Fast Adversarial training paper. 

2. So we are perturbing the features at each layer, this is somewhat similar to batchout. Batchout gave better results when applied to alternative layers, perhaps this method may also give better results when applied at alternative layers.

3. Eventually this method tries to perturb the value that would be added to a layer in AT. However, the value added to one layer will be taken as input to the other layer and this results in a difference in the perturbation. This difference, however gave better results.

4. 

   