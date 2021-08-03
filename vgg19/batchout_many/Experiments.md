## Experiments when applying batchout to various layers

*Date: 23/2/2021*

In the below experiments, there is a slight difference from the actual paper implementation. Here I ensure that the features of the same datapoint from the batch are added in subsequent batchout layers. However this datapoint is choosen randomly. 

#### 1. Batchout at alternative layers after a specific layer

```python
   def forward(self, x, y):
        x = self.features1(x) #0
        x = self.features2(x) #1
        x = self.features3(x) #2 R
        x = self.features4(x) #3 0
        x = self.features5(x) #4 1
        x = self.features6(x) #5 R
        x = self.features7(x) #6 2
        x = self.features8(x) #7 3
        x = self.features8(x) #8 4
        x = self.features8(x) #9 5
        x = self.features9(x) #10 R

        x = self.features10(x) #11 6

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features11(x) #12
        x = self.features11(x) #13

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #14
        x = self.features12(x) #15 R

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #16
        x = self.features11(x) #17

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #18
        x = self.features11(x) #19

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features12(x) #20 R
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

```



* For `n=0.1`, the model is trained similar to standard model. 89.7% test accuracy. {initial lr=0.1, decay=5e-4, momentum=0.9}. Robust error = 59.10% {e=8/255 - FGSM} which is actually less. But the training was completed easily just like standard training.



 ![progress_reg (5)](/home/hrushikesh/robust/vgg19/batchout_many/n_1/progress_reg (5).png)

* For `n=0.2` the model couldn't be trained. There was clear overfitting and the best accuracy I achieved was 60%. But then I realised that I was decreasing the learning rates too quickly. (This was probably the same reason why I didn't get the required results when implementing the batchout paper. ) ![progress_reg (5)](/home/hrushikesh/robust/vgg19/batchout_many/n_2/progress_reg (4).png)

  {initial lr=0.1 (droped only till 1e-3 but don't remember exactly when), decay=5e-4, momentum=0.9}. Accuracy=87.91%. Robust error =45.75%. This is a very good improvement. But the training required almost 3x more epochs 

* For `n=0.3` initially about 2-3 times the training process didn't give proper results. Observing high overfitting, I increased the regularization from 5e-4 to 1e-3. Also the momentum was changed from 0.9 to 0.5 and later when learning rate was changed to 1e-4, it was increased to 0.9. 

  ![progress_reg (5)](/home/hrushikesh/robust/vgg19/batchout_many/n_3/progress_reg (5).png)

  Accuracy: 86.73%, **Robust error=30.89%. This is a very good improvement.** For comparison of above results and the best result obtained by me in past, the following figure will be helpful:

![alt](/home/hrushikesh/robust/vgg19/batchout_many/comparision.png)



- [ ] Keep a link to the previous results below this picture
- [ ] merge both the docs
- [ ] Modify the AutoAttack para (after merging), sir clearly mentioned about it in the docs.
- [ ] Attack the model using AutoAttack after 11:20 tomorrow
- [ ] Create the contents at the start of the document
- [ ] Mention that other expts that can be done are applying batchout at various layers



| Model - VGG19bn                      | Test Accuracy | FGSM Accuracy (8/255) |
| ------------------------------------ | ------------- | --------------------- |
| Standard model                       | 89.19         | 32.09                 |
| Batchout at last layer               | 86.98         | 34                    |
| Batchout at middle layer (n_1)       | 86.79         | 37.81                 |
| Batchout at middle layer (n_2)       | 89.89         | 45.48                 |
| Batchout at middle layer (n_3)       | 89.52         | 48.22                 |
| Batchout at alternative layers (n_1) | 89.69         | 40.90                 |
| Batchout at alternative layers (n_2) | 87.91         | 54.25                 |
| Batchout at alternative layers (n_3) | 86.73         | 69.09                 |

