---
title: 'SimCLR Blog Post'
date: 2021-12-08
permalink: /posts/2021/12/simclr/
tags:
  - Self supervised learning 
  - Constrastive Learning
  - SimCLR 
---

EDL winter school assignment: reproducing SimCLR 
======
In this blog post I will document my implementation, experiments and finds of the 2020 paper by Chen et al. called "A Simple Framework for Contrastive Learning of Visual Representations" or in fewer words [SimCLR](https://arxiv.org/pdf/2002.05709.pdf).


1.Background to self-supervised contrastive learning 
------

**1.1 What is contrastive learning?**
	Constrative learning is a representation learning paradigm where the similarities and differences of samples in a dataset are exploited in order to learn more robust represenations of the data. This is done by learning a transformation which maps similar samples close to each other while repelling different samples from one and other. In the past this was often done in a supervised manner using triplet-losses and/or Siamese networks. One of the long standing problems with constrative learning is the requirement for negative samples. 

**1.2 What is self-supervised learning?**
	Self-supervised learning is an unsupervised machine learning framwork in which pretext tasks are used as indirect training objectives such that the trained models can be used for downstream tasks. Examples of pretext tasks are [context prediction](https://arxiv.org/pdf/1505.05192.pdf) and [rotation prediction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf).  


**1.3 Why then do we care about SimCLR?**
	SimCLR offers a simple constrative learning framework, that doesnt explicitly need negative samples nor complex augmentation routines. Amazingly, this self-supervised method has been shown to offer comparable performance (Top-1 accuracy on ImageNet) to supervised methods when a linear classifer is applied on top of the learn represenations. 


**1.4 What does this blog post bring to the table?**
	In this work I give a brief introduction into my pytorch implementation of simclr and I try reproduce the Fashion-MNIST (FMNIST) and CIFAR-10 results documented in the paper. This being said, the original publication uses transfer learning to adapt the ImageNet trainined model to CIFAR-10 and FMNIST, however I plan to train on these lower resolution datasets to see if equal performance can be obtained.


2.Simple Contrastive Learning 
------
SimCLR, as suggested in the name is simple and is formulated as follows. Given a batch of samples $$\vb{x}$$
$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$
