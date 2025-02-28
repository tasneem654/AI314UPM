#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Define max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Define adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Define fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)  
        
        # Define dropout layers
        self.dropout = nn.Dropout(p=0.5)
        
        # Initialize weights and biases
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Initialize weights from a normal distribution with mean=0 and variance=0.01
                nn.init.normal_(layer.weight, mean=0, std=0.1)  
                # Initialize biases to ones
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  
        return x


model = CNNModel()
print(model)


# In[ ]:




