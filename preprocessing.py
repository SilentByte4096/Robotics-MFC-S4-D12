import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label, category in enumerate(["no_pothole", "pothole"]):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                self.images.append(os.path.join(category_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

dataset_path = ""  
dataset = PotholeDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

sample_img, sample_label = dataset[0]
print(f"Sample image shape: {sample_img.shape}, Label: {sample_label}")

cv2.imshow("Sample Image", sample_img.numpy().transpose(1, 2, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()