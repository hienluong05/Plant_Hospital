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


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # Load pre-trained ResNet-50 model
        self.model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        output = self.model(x) # Forward pass through ResNet-50
        return output
    
# Select device: cuda if GPU is available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
model = CNN(dataProcessor.targets_size) # Create model with number of classes
model.to(device) # Take model to device

# Loss function and optimizer
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters())

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

device = "cpu"

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

# train the model in 5 epochs
print("Starting training...")
train_losses, validation_losses, train_accuracies, validation_accuracies = train_model(model, criterion, train_loader, validation_loader, 5)

# Save the model
print("Done training.")
torch.save(model.state_dict(), "plant_disease_detection_model.pt")

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

# # Calculate accuracy 
# def accuracy(loader):
#     n_correct = 0
#     n_total = 0
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             _, predictions = torch.max(outputs, 1)
#             n_correct += (predictions == targets).sum().item()
#             n_total += targets.shape[0]
#     model.train()
#     return n_correct / n_total

# print("Train Acc:", accuracy(train_loader))
# print("Val Acc:", accuracy(validation_loader))
# print("Test Acc:", accuracy(test_loader))

