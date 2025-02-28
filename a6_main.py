#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from a6_model_tasneem import CNNModel  # Import your model

# Custom Dataset Class for Tiny ImageNet Test Data
class TinyImageNetTestDataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = self._load_annotations(annotations_file)
        self.class_to_idx = self._create_class_to_idx()

    def _load_annotations(self, annotations_file):
        """Load annotations from val_annotations.txt."""
        annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:  # Ensure the line is properly formatted
                    image_name, class_name, _, _, _, _ = parts
                    annotations.append((image_name, class_name))
        return annotations

    def _create_class_to_idx(self):
        """Create a mapping from class names to indices."""
        classes = sorted(set([ann[1] for ann in self.annotations]))
        return {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name, class_name = self.annotations[idx]
        img_path = os.path.join(self.root_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]  # Convert class name to index
        if self.transform:
            image = self.transform(image)
        return image, label

# Training setup
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Model evaluation function
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    # Define dataset and data loader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Load training dataset
    train_dataset = datasets.ImageFolder(
        root='C:\\Users\\tasne\\Desktop\\DL\\Assigment 3\\tinyImageNet\\tiny-imagenet-200\\train',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Load test dataset using custom dataset class
    test_dataset = TinyImageNetTestDataset(
        root_dir='C:\\Users\\tasne\\Desktop\\DL\\Assigment 3\\tinyImageNet\\tiny-imagenet-200\\test\\images',
        annotations_file='C:\\Users\\tasne\\Desktop\\DL\\Assigment 3\\tinyImageNet\\tiny-imagenet-200\\val\\val_annotations.txt',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Instantiate model
    model = CNNModel()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    
    # Save model architecture
    model_config = {
        "model_name": "CNNModel",
        "architecture": "Custom CNN",
        "accuracy": accuracy
    }
    with open("config.json", "w") as f:
        json.dump(model_config, f)
    
    # Save model parameters
    torch.save(model.state_dict(), "model.pth")
    
    print("Model and configuration saved successfully!")


# In[ ]:




