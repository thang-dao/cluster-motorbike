import os
import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class   Motorbike(Dataset):
    def __init__(self, root_dir, images, transform=None):
        self.images = images
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
               
        return img, self.images[idx]

def get_dataloader(root_dir, batch_size):
    images = os.listdir(root_dir)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    motorbike_dataset = Motorbike(root_dir, images, transform=transform)
    dataloader = DataLoader(motorbike_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    return dataloader

if __name__ == "__main__":
    root_dir = '/home/run/ai_challange/ai_challange/data/moto_data/motobike/1'
    images = os.listdir(root_dir)
    dataloader = get_dataloader(root_dir)
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched.shape)