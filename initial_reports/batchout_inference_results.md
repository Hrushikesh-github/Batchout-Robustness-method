### Batchout in Inference Phase

##### Algorithm

1. Load the model 

2. Choose a perturbation set with equal number of images of each class. For example if 30 images were choosen, then there would be 3 images of cat, 3 of dog etc. 

   Perturbation set is images that are sampled either from test dataset or train dataset. They will be perturbed(batchout perturbation) with the image that we want to predict. 

3. Pass the image along with the perturbation set. We apply batchout on the image with each of images in perturbation set.

   The code gives result in the following way:

   ​	Suppose perturbation set contains 30 images. The output of model when we pass image and 			   	perturbation set will be tensor of shape [31, 10]. Where each row is a prediction. The 1^st^ row contains 	the prediction when image is passed unperturbed to the model. The other 30 rows are predictions 	    	when we perturb the model.

4. We take two types of accuracies/predictions.

   	1. Count_error: Take the class which is most commonly predicted.
   	2. Mean_error: Take the mean of the 31 predictions and then take the prediction. (This was done in mixup Inference paper)

5. If the class obtained from above method is same as given class, the image correctly predicted, otherwise not. 



Batchout trained models are giving 20% accuracy on PGD(50 steps). But give only 5% on PGD(200 steps).  



13.2 13.19

30 samples from test set as perturbation images. n=0.3 model with same value for inference. count_error and mean_error are 13.22 and 13.21%. 13.27% was normal test accuracy without any perturbation. FGSM errors are 47.35, 47.31% and PGD errors are 76.03, 76.04 

When 30 samples from train set was taken:  Normal: 13.24 13.21, PGD: 75.91 75.99, FGSM:47.36 47.32 

When 10 samples from test set: Normal: 13.2 13.19 PGD 75.27 75.36 FGSM: 47.37 47.31. This is very suprising as we get very much same results

When 80 samples from test set: Normal: 13.22 13.21 PGD 75.86 75.99 FGSM:47.36 47.31

30 samples from test but with n=0.2: Normal: (Will remain same) PGD 76.94 76.98 FGSM: 47.43 47.42

![a](/home/reshikesh/diary/pics/a.jpeg)

PGD(10 steps) error 45.2. Inference gives 44.36 44.38 error %

PGD(50 steps) error is 80.79. Inference gives 79.69 79.9 error % 

Note that the above results are for alt_3 model



********************************************************************************************************************************



Model trained with same target images, 

113.pt: (n=0.20)

```
PGD error (10, 50, 200) is 0.599, 0.7642, 0.7882
FGSM error is:  0.6723
```

146.pt: (n=0.25)

```
PGD error(10, 50, 200) is:  0.5442,0.7273, 0.8479, 
FGSM error is:  0.6811
```

I have confirmed for alt_3, PGD(200 steps) gives 0.9509 error and 113.pt is giving 78.82 error as mentioned above.

alt_2 FGSM, PGD(10, 50, 200) error rate: 0.6381, 0.6151, 0.8263, 0.9116

all_middle_2 PGD(10, 50, 200) error rate: 0.5603, 0.6921, **0.7418**

all_middle_3 test_acc = 86.43 and FGSM error is 70.79 PGD(50, 200): 0.8793, 0.8918

random_3_-2_0 test, FGSM, PGD(10, 50, 200): 82.88%, 54.31%, 44.94%, 0.553, **0.6807**



| #     | Model              | Test Acc          | FGSM Acc | PGD(10) | PGD(50) | PGD(200)     |
| ----- | ------------------ | ----------------- | -------- | ------- | ------- | ------------ |
| 1     | Alt_3              | 86.73             | 52.50    | -45.2   | -80.79  | -95.09       |
| 2     | Alt_2              | 87.91             | 36.19    | -61.51  | -82.63  | -91.16       |
| 3     | Alt_2 (*Change*)   | 90.25             | 32.77    | -59.9   | -76.42  | -78.82       |
| 4     | Alt_2.5 (*Change*) | 86.25             | 31.89    | -54.42  | -72.73  | -84.79       |
| 5     | All_middle_2       | 90.21             | 42.40    | -56.03  | -69.21  | -74.18       |
| 6     | All_middle_3       | 86.29             | 29.21    |         | -87.93  | -89.18       |
| **7** | alt_random_3_-2_0  | 82.88 (very less) | 37.62    | -44.94  | -55.3   | -67.63(best) |
| 8     | alt_random_3_-2    | 87.66             | 39.44    |         | -67.46  | -78.06       |

In the above table ''-'' refers to error.

all_middle[2_-2_0] with change with p[0.6, 0.2, 0.2] gave 93.92% error on 200 steps. A failure case.

1. Random value of n is giving very good improvement. 
2. Having batchout for all the features for n=.2 is giving very good results. So small perturbations for all layers are giving good results.
3. The value of n=.3 is getting less acc than n=.2