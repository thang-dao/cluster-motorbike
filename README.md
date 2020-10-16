# Cluster Motorbike 
The project implements to pre-process data for [Zalo Challenge 2019](https://challenge.zalo.ai/#challenge).
First, we use pre-trained [mobilenet_v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) to extract features from images, 
then use the features for cluster by [KMeans Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) 

### Prerequisites  
 1. torch
 2. numpy
 3. opencv
 4. scikit-learn
 5. torchvision

 