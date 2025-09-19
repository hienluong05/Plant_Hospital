import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import datasets, transforms, models  # datsets  , transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torchvision.models import resnet50, ResNet50_Weights 

import dataProcessor 

# Định nghĩa lại class CNN giống lúc train ban đầu
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)

num_classes = 15  # 
model = CNN(num_classes)
model.load_state_dict(torch.load(r"G:\My Drive\Documents\Plant_Disease_Detection_Copy\MyAI\App\flaskr\plant_disease_detection_model.pt"))

# Select device: cuda if GPU is available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# create DataLoader for training, validation and test sets
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset, 
    batch_size=batch_size, 
    sampler=dataProcessor.train_sampler
)

test_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset,      
    batch_size=batch_size,
    sampler=dataProcessor.test_sampler
)

validation_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset,
    batch_size = batch_size,
    sampler = dataProcessor.validation_sampler
)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

# Train using batch
def train_model(model, criterion, train_loader, validation_loader, epochs):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    validation_accuracies = np.zeros(epochs)
    
    for epoch in range(epochs):
        t0 = datetime.now()
        train_loss = []
        n_correct = 0
        n_total = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            _, preds = torch.max(output, 1)
            n_correct += (preds == targets).sum().item()
            n_total += targets.size(0)
        train_acc = n_correct / n_total
        train_loss = np.mean(train_loss)
        
        model.eval()
        validation_loss = []
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                validation_loss.append(loss.item())
                _, preds = torch.max(output, 1)
                n_correct += (preds == targets).sum().item()
                n_total += targets.size(0)
        val_acc = n_correct / n_total
        validation_loss = np.mean(validation_loss)
        
        train_losses[epoch] = train_loss
        validation_losses[epoch] = validation_loss
        train_accuracies[epoch] = train_acc
        validation_accuracies[epoch] = val_acc
        
        dt = datetime.now() - t0
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {dt}")
        
    return train_losses, validation_losses, train_accuracies, validation_accuracies

# Resume training
print("Resume Training...")
train_losses, validation_losses, train_accuracies, validation_accuracies = train_model(model, criterion, train_loader, validation_loader, epochs=5)  # train tiếp 10 epochs nữa

# Hoặc ghi đè file cũ
torch.save(model.state_dict(), r"G:\My Drive\Documents\Plant_Disease_Detection_Copy\MyAI\App\flaskr\plant_disease_detection_model.pt")
print("Original model updated!")

# Plot the losses
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.show()

# Plot the accuracies
plt.figure(figsize=(8,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.show()