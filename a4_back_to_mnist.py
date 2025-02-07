#!/usr/bin/env python
# coding: utf-8

# In[37]:


import time
import random
import numpy as np
import torch as tr
import torchvision as tv


# Make sure you have a unique seed (different from other students)
seed = 38559076
random.seed(seed)
np.random.seed(seed)
tr.manual_seed(seed)

# Getting MNIST dataset
trn_dataset = tv.datasets.MNIST(root='data', train=True, download=True)
evl_dataset = tv.datasets.MNIST(root='data', train=False, download=False)


# Manual transformations

## on training data
trn_images, trn_labels = zip(*trn_dataset)
trn_images = np.array(list(map(lambda x: 2.0 * np.asarray(x).flatten().astype(np.float32) / 255.0 - 1.0, trn_images)))
trn_images = tr.from_numpy(trn_images)
trn_labels = tr.from_numpy(np.array(trn_labels).astype(np.int64))

## on training data
evl_images, evl_labels = zip(*evl_dataset)
evl_images = np.array(list(map(lambda x: 2.0 * np.asarray(x).flatten().astype(np.float32) / 255.0 - 1.0, evl_images)))
evl_images = tr.from_numpy(evl_images)
evl_labels = tr.from_numpy(np.array(evl_labels).astype(np.int64))

# Quick sanity check
print(trn_images.dtype, trn_images.shape, trn_images.min(), trn_images.max())
print(trn_labels.shape, trn_labels.min(), trn_labels.max())
print(evl_images.dtype, evl_images.shape, evl_images.min(), evl_images.max())
print(evl_labels.shape, evl_labels.min(), evl_labels.max())


# ///////////////////////////////////////////////////
# >>>>>>>> START OF YOUR PLAYGROUND

batch_size = 1024

# Simple Feedforward Neural Network
class SimpleFFNet(tr.nn.Module):
  def __init__(self):
    super().__init__()
    # 784 -> 16 - Layer 1 -- Affine Transformation
    self.linear1 = tr.nn.Linear(28 * 28, 16, bias=True)
    # 16 -> 16  - Layer 2 -- Linear Transformation
    self.linear2 = tr.nn.Linear(16, 16, bias=False)
    #self.linear3 = tr.nn.Linear(16, 16, bias=False)
    # 16 -> 10  - Layer 3 -- Affine Transformation
    self.linear3 = tr.nn.Linear(16, 10, bias=True)         

  def init_weights(self):
    #tr.nn.init.xavier_uniform_(self.linear1.weight)
    #tr.nn.init.xavier_uniform_(self.linear2.weight)
    #tr.nn.init.xavier_uniform_(self.linear3.weight)
    #tr.nn.init.uniform_(self.linear1.weight, a=-0.1, b=0.1)
    #tr.nn.init.uniform_(self.linear2.weight, a=-0.1, b=0.1)
    #tr.nn.init.uniform_(self.linear3.weight, a=-0.1, b=0.1)
      
    #tr.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    #tr.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
    #tr.nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='relu')
      
    tr.nn.init.normal_(self.linear1.weight, mean=0.0, std=0.01)
    tr.nn.init.normal_(self.linear2.weight, mean=0.0, std=0.01)
    tr.nn.init.normal_(self.linear3.weight, mean=0.0, std=0.01)
    #tr.nn.init.normal_(self.linear4.weight, mean=0.0, std=0.01)
  
  def forward(self, x):
    x = tr.tanh(self.linear1(x))
    x = tr.tanh(self.linear2(x))

    #x = tr.sigmoid(self.linear1(x))
    #x = tr.sigmoid(self.linear2(x))

    #x = tr.relu(self.linear1(x))
    #x = tr.relu(self.linear2(x))
    

    #x = self.linear1(x)
    # x = tr.relu(x)  or  x = tr.sigmoid(x)  or YOUR OWN FUNCTION
    #x = self.linear2(x)
    # x = tr.relu(x)  or  x = tr.sigmoid(x)  or YOUR OWN FUNCTION
    x = self.linear3(x)
    #x = self.linear4(x)
    return x

# >>>>>>>> END OF YOUR PLAYGROUND
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# Create the model, define the loss function, and the optimizer
model = SimpleFFNet()
model.init_weights()
loss_fn = tr.nn.CrossEntropyLoss()
optimizer = tr.optim.Adam(model.parameters(), lr=0.0001)

# Training
start_time = time.time()
nbatch = np.ceil(len(trn_images) / batch_size)
for epoch in range(1, (epochs := 1000) + 1):
  for idx in np.array_split(np.arange(len(trn_images)), nbatch):
    optimizer.zero_grad()
    model.train()
    output = model(trn_images[idx])
    loss = loss_fn(output, trn_labels[idx])
    loss.backward()
    optimizer.step()


  if epoch % 10 == 0:
    model.eval()
    with tr.no_grad():
      output = model(evl_images)
      loss = loss_fn(output, evl_labels)
      accuracy = tr.sum(tr.argmax(output, dim=1) == evl_labels).item() / len(evl_labels)
      print(f'Epoch {epoch:3d}/{epochs} -- '
            f'Loss: {loss.item():.5f} -- '
            f'Eval Accuracy: {accuracy*100:.2f}% -- '
            f'Time: {time.time() - start_time:.1f} seconds')


end_time = time.time()
print(f'You achieved a loss of: {loss.item():.5f} -- '
      f'Eval Accuracy: {accuracy*100:.2f}% -- '
      f'in: {end_time - start_time:.2f} seconds')


# In[41]:


import matplotlib.pyplot as plt

# Evaluate Model and Collect Misclassified Samples
model.eval()
misclassified = []

with tr.no_grad():
    output = model(evl_images)
    predicted_labels = tr.argmax(output, dim=1)
    
    # Find misclassified indices
    misclassified_indices = (predicted_labels != evl_labels).nonzero(as_tuple=True)[0]

    # Collect first 10 misclassified samples
    for i in range(min(10, len(misclassified_indices))):
        idx = misclassified_indices[i].item()
        misclassified.append((evl_images[idx].reshape(28, 28), evl_labels[idx].item(), predicted_labels[idx].item()))

# Plot the misclassified images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Misclassified Images (True Label → Predicted Label)", fontsize=14)

for ax, (image, true_label, pred_label) in zip(axes.flat, misclassified):
    ax.imshow(image, cmap='gray')
    ax.set_title(f"{true_label} → {pred_label}", fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




