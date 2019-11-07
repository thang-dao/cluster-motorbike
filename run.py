import torch
import numpy as np
from pandas import DataFrame 
from sklearn.cluster import KMeans
from Dataloader import *
from torchvision.models.resnet import resnet18
from torchvision.models import mobilenet_v2, vgg16


def clusers_motor(path_data, model, batch_size=1):

    model.cuda()
    print(model)
    X = []
    image_ = []
    data_loader = get_dataloader(root_dir=data_path, batch_size=batch_size)
    for sample_batched, image_name in data_loader:
        sample_batched = sample_batched.cuda()
        resnet_feature = model(sample_batched)
        resnet_feature_np = resnet_feature.detach().cpu().numpy()
        X.append(resnet_feature_np.flatten())
        image_.append(image_name[0])
        print(resnet_feature.shape)
    X = np.array(X)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    motorbike = {'Name': image_, 'Class': kmeans.labels_}
    df = DataFrame(motorbike, columns=['Name', 'Class'])
    export_csv = df.to_csv('./motorbike.csv', index=None)
    print(df)
    return export_csv
    
if __name__ == "__main__":
    data_path = '/home/vietthangtik15/zalo/motobike'
    model = mobilenet_v2(pretrained=True)
    _csv = clusers_motor(data_path, model)