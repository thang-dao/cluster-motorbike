import torch
import numpy as np
from sklearn.cluster import KMeans
from Dataloader import *
from torchvision.models.resnet import resnet18
from torchvision.models import mobilenet_v2, vgg16

data_path = '/home/vietthangtik15/zalo/motobike'
batch_size = 1
# model = resnet18(pretrained=True)
model = mobilenet_v2(pretrained=True)
model.cuda()
print(model)
# X = torch.tensor()
X = []
data_loader = get_dataloader(root_dir=data_path, batch_size=batch_size)
for i_batch, sample_batched in enumerate(data_loader):
    sample_batched = sample_batched.cuda()
    resnet_feature = model(sample_batched)
    resnet_feature_np = resnet_feature.detach().cpu().numpy()
    X.append(resnet_feature_np.flatten())
    print(resnet_feature.shape)
print(len(X))
X = np.asarray(X)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
print(kmeans.labels_)