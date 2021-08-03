

10^th^ May:

So far various ablation studies were done with VGG19, applying batchout in different ways. Three major types of Models trained with batchout were:

#### 1. Batchout on Alternative Layers (Alt):

<img src="/home/reshikesh/Pictures/Screenshot from 2021-05-10 20-42-15.png" alt="Screenshot from 2021-05-10 20-42-15" style="zoom: 80%;" />

Batchout is not applied initially for 1st 12 feature spaces. After 12 features are learnt, batchout is applied on alternative layers. In above pic we can see it is applied after every odd # feature.

#### 2. Batchout on All layers (All):

![Screenshot from 2021-05-10 20-42-36](/home/reshikesh/Pictures/Screenshot from 2021-05-10 20-42-36.png)

Here batchout is applied after every layer but not in the initial layers. In above pic, batchout can be seen applied after #12 or #14 which was absent in the 1st type. 

#### 3. Batchout on Alternative layers with random value of convex parameter(Ran)

Same as 1 but the value of convex parameter is randomly changed for a batch.    



Let us consider a simple scenario where there are three images: Cat, Dog and Frog. Assume due to randomness the cat image is perturbed with dog, the dog with frog and frog with cat. In the 1st layer of batchout, we see normal convex combination. But in the 2nd layer, the peruturbed image of cat also has some perturbations from frog class. This appears to be against intuition.   

![WhatsApp Image 2021-05-10 at 9.03.00 PM](/home/reshikesh/diary/pics/b.jpeg)

So the solution to this was thought to perturb the image of cat with dog and the dog with cat. The image of the frog will be perturbed with other image. (Batchsize is generally even numbers: 32, 64 etc, so every image will undergo perturbation.) This was realized this week, let models trained this way be denoted as **C**.



The results obtained are:

| #     | Model             | Test Acc          | FGSM Acc  | PGD(10)   | PGD(50)  | PGD(200)        |
| ----- | ----------------- | ----------------- | --------- | --------- | -------- | --------------- |
| 1     | Alt_3             | 86.73             | **52.50** | 54.8      | 19.21    | 4.91            |
| 2     | Alt_2             | 87.91             | 36.19     | 38.49     | 17.37    | 8.84            |
| 3     | Alt_2 (*C*)       | 90.25             | 32.77     | 40.1      | 23.58    | 21.18           |
| 4     | Alt_2.5 (*C*)     | 86.25             | 31.89     | 45.58     | 27.27    | 15.21           |
| 5     | All_middle_2      | 90.21             | 42.40     | 43.97     | 30.79    | 25.82           |
| 6     | All_middle_3      | 86.29             | 29.21     |           | 12.07    | 10.82           |
| **7** | alt_random_3_-2_0 | 82.88 (very less) | 37.62     | **55.06** | **44.7** | **32.37(best)** |
| 8     | alt_random_3_-2   | 87.66             | 39.44     |           | 32.54    | 21.94           |

In the above table, _3 refers to n=0.3 and similarly 2/2.5

1. The best accuracy against PGD(200 steps) is 32.37 % by a alternative model trained with n=[0.3, -0.2, 0] with probability of selecting n [0.6, 0.3, 0.1]. Interestingly this model gave very less test accuracy. original_feature
2. Alternate trained model with n=0.3 gave highest accuracy on FGSM but failed against 200 steps PGD.
3. Only two models were trained with a change **C** and though they didn't give good FGSM accuracy but there was improvement in PGD (200 steps). Madry et al AT model actually had higher PGD accuracy than FGSM accuracy.
4. Applying batchout on all layers also gave very good accuracy.
5. n=0.2 looks to give better PGD accuracy than n=0.3



Considering above points the model all_middle[2_-2_0] with **C** with p[0.6, 0.2, 0.2] was trained and gave 93.92% error on 200 steps. However, the model was trained only once. Maybe because of some mistakes in training, I might have got less accuracy.

Autoattack gives 0.80% acc on alt_random_3_-2 but that is for only 1000 examples of test set. What is suprising is it is saying 250 out of 250 successfully perturbed but then says robust acc is 0.8%.

#### Conclusions

1. A small change in the algorithm gave good improvement. 
2. Initially I thought the models gave only good accuracy on FGSM but I see good accuracies on PGD as well.
3. Training a model on CIFAR10 would take close to 90 minutes-2:30 hours. So should I perform more experiments with current model trying to improve accuracy or shift to ResNet or perform experiments with other datasets: CIFAR100 and MNIST. If proper results are obtained, maybe a paper is possible to submit to [ICML](https://advml-workshop.github.io/icml2021/) which has a deadline of 5^th^ June.



15^th^ may

### ResNet and DenseNet experiments

1. Resnet18 alternate middle with n=0.2. PGD(50 steps) 0% accuracy, FGSM acc is 5.7%.

2. All middle, n=0.2. PGD(50) is giving 0% accuracy and FGSM acc is 11.5%.

3. All middle and alt middle with random [0.3, -0.2, 0], p=[0.6, 0.3, 0.1]) gave 0% acc on PGD(50). Both gave very good test acc. But this random is different from the batchout random, the value of n is not same throughtout the layers for the same class.

   In both of the above, the training happened very quickly/smoothly, so maybe we need to increase strength of adversaries? Things to try:

   1. batchout on all layers. Think that the model's initial layers are learning a lot unlike VGG or the perturbations resulting in 1st layer are continuing in the network due to the addition?
   
   
   
   
   
   | #    | Model                            | Test Acc  | FGSM Acc | PGD(50) Acc |
   | ---- | -------------------------------- | --------- | -------- | ----------- |
   | 1    | Alt_middle n=0.2                 | 87+       | 5.7      | 0           |
   | 2    | All_middle n=0.2                 | 87+       | 11.5     | 0           |
   | 3    | All_middle rand                  | 87.75     | 7        | 0           |
   | 4    | Alt_middle rand                  | 92        | 8.3      | 0           |
   | 5    | All n=0.1                        | 90.4      | 0.4      | 0           |
   | 6    | All n=0.2 (with and without reg) | 81.3/84.6 | 2.7      | 0           |
   
   

Densenet also similary failed. Giving around 10% FGSM acc and 0% PGD acc. But standard model gives 88.31 acc.



### MNIST experiments

##### Architecture of model

![image-20210518203811649](/home/reshikesh/.config/Typora/typora-user-images/image-20210518203811649.png)

Standard model gave 97.99% acc but PGD error(50) is 0% and FGSM error is 0.5%.

n =0.1, 1% acc on PGD and 6.5% on FGSM

Training doesn't converge for 0.2, 0.3. If I remove the 1st batchout and train with 

n=0.1, I get FGSM acc 10.7% and 1% PGD

n=0.15, I get FGSM acc 4.2% and 0% PGD.

n=0.2 with 91.1% acc FGSM and PGD both 1%

Now I apply batchout everywhere except the last layer. Again 1% on PGD, 10% on FGSM. No improvement at all.

I think the model is too shallow, let's just train it on bigger model

**Conclusions**:

Densenet and ResNet both of them failed to give any significant robust accuracies when batchout was applied on CIFAR10. 

When batchout was applied to MNIST, no model gave more than 5% robust accuracies on e=0.3, including the architecture present in the original paper.

So the only positive from experiments is VGG19 gives 32.37% acc on PGD(200). Out of all the various papers, only one paper contains robust accuracy of VGG16 achieving 30.9% acc on PGD.

However, the SOTA paper says that PGD(200) is not a good way to confirm robustness of a model because of gradient masking. So auto attack was performed on them which gave 0% accuracy.

Conclusion is even though we achieved 32.37% acc on VGG19 which I think is more than what would be achieved by A.T on VGG19, it is not robust, as AutoAttack gives 0.80% accuracy. 

AutoAttack performs PGD attack but with no hyper-parameters. Usually it performs around 1000 iterations with different step sizes, following a exploration and exploitation strategy. The step sizes are usually big in the beginning(exploration) and then becomes small (exploitation).