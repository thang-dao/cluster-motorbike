import torch
from sklearn.cluster import KMeans
from Dataloader import *
from torchvision.models.resnet import resnet50

data_path = '/home/run/ai_challange/ai_challange/data/moto_data/motobike/1'
batch_size = 64
model = resnet50()
# model.cuda()
print(model)
# X = torch.tensor()
data_loader = get_dataloader(root_dir=data_path, batch_size=batch_size)
for i_batch, sample_batched in enumerate(data_loader):
    # sample_batched = sample_batched.cuda()
    if i_batch == 0:
        output = model(sample_batched)
        X = output
    else:
        output = model(sample_batched)
        X = torch.cat((X, output), 0)
    print(X.shape)
print(X.shape)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
print(kmeans.labels_)