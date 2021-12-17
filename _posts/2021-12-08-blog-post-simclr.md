---
title: 'Reproducing SimCLR'
date: 2021-12-08
permalink: /posts/2021/12/simclr/
tags:
  - Self supervised learning 
  - Constrastive Learning
  - SimCLR 
---

In this blog post I will document my implementation, experiments and findings obtained from the reproduction of the 2020 paper by Chen et al. entitled "A Simple Framework for Contrastive Learning of Visual Representations" or in fewer words [SimCLR](https://arxiv.org/pdf/2002.05709.pdf).


1.Background to self-supervised contrastive learning 
======

**1.1 What is contrastive learning?**
	Constrative learning is a representation learning paradigm where the similarities and differences of samples in a dataset are exploited in order to learn more robust represenations of the data. This is done by learning a transformation which maps similar samples close to each other while repelling different samples from one and other. In the past this was often done in a supervised manner using triplet-losses and/or [Siamese networks](https://arxiv.org/pdf/1412.6622.pdf). One of the long standing problems with constrative learning is the requirement for negative samples, as to find differences between samples we need to know what "difference" is.

**1.2 What is self-supervised learning?**
	Self-supervised learning is an unsupervised machine learning framework in which pretext tasks are used as indirect training objectives such that the trained models can be used for downstream tasks. Examples of pretext tasks are [context prediction](https://arxiv.org/pdf/1505.05192.pdf) and [rotation prediction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf). In this paradigm, rather than training networks with explicit categorical labels, such as done when training classifiers, we exploit implicitly known similaries in the (augmented) data, such as predicting whether a particular image is rotated or not. 


**1.3 Why then do we care about SimCLR?**
	SimCLR offers a simple constrative learning framework, that doesnt explicitly need negative samples nor complex augmentation routines. Amazingly, this self-supervised method has been shown to offer comparable performance (Top-1 accuracy on ImageNet) to supervised methods when a linear classifer is applied on top of the learnt represenations. 


**1.4 What does this blog post bring to the table?**
	In this work I give a brief introduction into my pytorch implementation of simclr and I try reproduce the CIFAR-10 results documented in the paper. This being said, the original publication uses transfer learning to adapt the ImageNet trainined model to CIFAR-10, however I plan to train on these lower resolution datasets to see if equal performance can be obtained.


2.Simple contrastive learning 
======
SimCLR, as suggested in the name is simple and is formulated as follows. Given a batch, $$x$$, of $$N$$ samples and two augmentations $$\tau_0,\tau_1$$ drawn from some set of augmentations $$ {\rm T}$$, then 

$$\hat{x_i} = t_0(x) \quad \text{and} \quad \hat{x_j} = t_1(x) \quad \text{where} \quad  \tau_0, \tau_1 \sim {\rm T}$$

such that we obtain 1 positively augmented sample and $$2(N-1)$$ negatively augmented samples. Then we project the both sets of augmented samples to their latent representation using encoder $$f()$$. In the case of this blog post we limit ourselves to a ResNet-18, such that

$$ h_i = f(\hat{x_i}) \quad  and \quad h_j = f(\hat{x_j}) $$

It must be noted that these representations $$h_i$$ and $$h_j$$ are used for the downstream classification tasks, but for training purposes we still need to apply a non-linear MLP, $$g$$. Such that 

$$ z_i = g(h_i) \quad  and \quad z_j = g(h_j) $$

The final part of the SimCLR formulation is the loss function. Here a we use a cosine-similarity latent projections $$z_i$$ and $$z_j$$. We then weight it using a modified cross-entropy based soft-max function,

$$ \mathcal{L}_{\text{SimCLR}}^{(i,j)} = -\log \big( \dfrac{exp(sim(z_i,z_j)/t)}{\sum_{k=1}^{2N} exp(sim(z_i, z_k)/t)} \big), \quad i \neq k$$

where $$t$$ is the temperature variable and $$sim()$$ is the cosine-similarity function. 


3.Experimental setup 
======
In this section I define my pytorch based implementation of SimCLR. For a full overview of the pytorch based implementation please see [the full Github repository](https://github.com/mesarcik/simclr).
 
**3.1 Data Augmentation**
		The data augmentations are implemented using `torchvision.transforms`. Here we define the augmentations as specified by the appendices of the SimCLR paper:

- Randomly cropping and re-sizing to original dimensions
- Randomly flipping horizontally
- Applying a colour jitter, this is applied randomly with a probability of 0.8 
- Finally we change some images to gray scale with a probability of 0.2 and apply a Gaussian blur to 50% of the data.

The implementation of this can be seen below: 

```python
from torchvision import transforms

color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
blur = transforms.GaussianBlur((k_size, k_size), sigma=(0.1, 2.0))

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
								  transforms.RandomHorizontalFlip(),
								  transforms.RandomApply([color_jitter], p=0.8),
								  transforms.RandomGrayscale(p=0.2),
								  transforms.RandomApply([blur], p=0.5),
								  transforms.ToTensor()])
```

**3.2 Model Specifcation**
		The key difference in the SimCLR model and the original ResNet is that a non-linear projection head is required as specified in Equation 3. In this case we remove the final fully connected layer of the ResNet to obtain the $$h()$$ projection and then replace it with a fully connected layer and a ReLU activation. This is implemented in pytorch as follows: 
  
```python
# adapted from https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py

import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, out_dim=128):
        super(Resnet, self).__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=out_dim)
        dim_mlp = self.resnet.fc.in_features
    
        self.resnet.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.resnet.fc)


    def forward(self, x): 
        return self.resnet(x)

```

**3.3 SimCLR Loss**
The next part of the implementation is the SimCLR loss. Here we implement the loss function in the most readable, but probably the least efficient way. The softmax-based loss is defined below: 

```python
from torch.nn import CosineSimilarity
_sim = CosineSimilarity(dim=1, eps=1e-6)

def simclr_loss(negative_z_0, negative_z_1, positive_z_0, positive_z_1):
    positive = _sim(positive_z_0, positive_z_1)
    negative = 0
    for i in range(_batch_size-1):
        negative += torch.exp(_sim(negative_z_0[i:i+1,:],
                                   negative_z_1[i:i+1,:])/_t).item()

    loss = -torch.log(torch.exp(positive/_t)/negative)
    return loss

```
Here we calculate the cosine similarity of the positive samples and divide this through by the temperature weighted sum of the negative samples. Then using this we index each sub-batch of positive and negative pairs using the `cat` operator, and pass the pairs to the similarity function defined above.  We then iterate over each batch, calculate the similarities and update the weights of the model defined below. 


```python
  	z_0 = model(x_augmented_0.to(_device))
	z_1 = model(x_augmented_1.to(_device))
	for i in range(batch_size):
			# the positive pair of augmented 
			positive_z_0 = z_0[i:i+1,...]
			positive_z_1 = z_1[i:i+1,...]

			# the negative pair of augmented 
			negative_z_0 = torch.cat((z_0[:i,...],z_0[i+1:,...]), axis=0)
			negative_z_1 = torch.cat((z_1[:i,...],z_1[i+1:,...]), axis=0)

			if i ==0: loss = simclr_loss(negative_z_0, negative_z_1, positive_z_0, positive_z_1)
			else: loss+= simclr_loss(negative_z_0, negative_z_1, positive_z_0, positive_z_1)
   	loss = loss/2*_batch_size
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```


**3.3 Linear Classifier**
The linear classifier is implemented in the sklearn framework. It receives the embedding generated by the SimCLR model and uses a multi-class SVM to classify the test data. It must be noted that the SVM is applied to the embeddings, $$h$$, before the non-linear MLP is applied. The code is shown below:

```python
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(z_train, z_train_labels)
y = svclassifier.predict(z_test)   

``` 

4.Intermediate results
======
Unfortunately due to other work, full testing and validation of this SimCLR implementation was not possible. As of writing, the model has been trained for only 17 epochs, whereas the paper recommends training for at least 100 epochs. For this reason we believe that the model has not yet converged and therefore the results cannot be considered a full reproduction. Nonetheless the results are documented below. In the first figure (shown below) we can see the loss function over 17 epochs, here we use a batch size of 1024 on the CIFAR-10 dataset. It can be seen that the training process is noisey and the model has not converged.    

![Loss](/assets/images/los6.png)

Using this partially trained model we apply a linear classifier, and obtain the following accuracy report: 

```
              precision    recall  f1-score   support

           0       0.35      0.34      0.34      1000
           1       0.35      0.19      0.25      1000
           2       0.20      0.19      0.20      1000
           3       0.22      0.12      0.16      1000
           4       0.24      0.37      0.29      1000
           5       0.35      0.20      0.26      1000
           6       0.26      0.40      0.32      1000
           7       0.32      0.13      0.18      1000
           8       0.35      0.57      0.43      1000
           9       0.33      0.43      0.37      1000

    accuracy                           0.29     10000
   macro avg       0.30      0.29      0.28     10000
weighted avg       0.30      0.29      0.28     10000
``` 

Here it is clear that our embedding is not fuly trained as we are only obtaining a Top-1 accuracy of 0.29. We investigate this result further by inspecting the 2-dimensional t-SNE embedding of the first 2000 samples of our learnt representations, as can be seen the plot below:

![TSNE](/assets/images/tsne.png)

Here it is clear that our model has not learnt enough about the training data to distinguish between classes.  

5.Conclusions and final remarks
======

In this blog post we have shown an implementation of the SimCLR paper, unfortunately due to time constraints a full replication of the original paper's results was not possible. We note that it is possible with more time (training) the model should converge thus improving performance. Nonetheless, I hope you found this post insightful and easy to follow. 


