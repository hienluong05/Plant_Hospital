import torch
from torchvision import transforms, datasets, models
import numpy as np
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import os

weights = ResNet50_Weights.IMAGENET1K_V2

# Import dataset and transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    weights.transforms()  # KHÔNG unpack, chỉ dùng như một transform cuối
])

dataset = datasets.ImageFolder(root=r'C:\PlantVillage', transform=transform)

# Print the number of images, original position and transforms applied
dataset 

# Split the dataset
indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))  # 85% of all for training and validation
train = int(np.floor(0.70 * split))   # 70% of split for training, the rest for validation

# Check number of images in each set
print("Number of images in training set:", train)
print("Number of images in validation set:", split - train)    
print("Number of images in test set:", len(dataset) - split)

np.random.shuffle(indices)

train_indices, validation_indices, test_indices = (
    indices[:train], 
    indices[train:split], 
    indices[split:]
)

# Create samplers
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

for idx in train_sampler.indices:
    img_path, _ = dataset.samples[idx]
    if not os.path.exists(img_path):
        print("NOT FOUND:", img_path)

# Get number of classes
targets_size = len(dataset.class_to_idx)
print('Number of classes:', targets_size)