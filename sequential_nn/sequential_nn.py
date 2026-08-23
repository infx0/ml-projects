"""Train and analyze a feed-forward PyTorch network on the Iris dataset.

The script loads and standardizes the data, creates train and test tensors and data
loaders, defines a sequential classification model, trains it while recording timing
and loss information, and reports and visualizes its predictive performance.
"""

# Setup and Imports
# Core Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn for data and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Utilities
import time
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
import pandas as pd

# load the dataset
iris = load_iris()
# create a DataFrame with the feature data
# Data Loading and Processing
# X - data
feature_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# add the target labels
# y - target
label_df = pd.DataFrame(iris.target, columns=["label"])
print("RAW DATA")
print(feature_df.head())
print(label_df.head())

# Reproducibility
rand_seed = 42
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)
batch_size = 32

# Scale, Fit
scaler = StandardScaler()
scaled_feature_df = pd.DataFrame(
    scaler.fit_transform(feature_df), columns=iris.feature_names
)
print("STANDARD SCALED DATA")
print(scaled_feature_df.head())
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_feature_df, label_df, test_size=0.2, random_state=rand_seed
)

# Pipeline

# convert training and test data into correct types
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long).squeeze()
y_test = torch.tensor(y_test.values, dtype=torch.long).squeeze()

# define the datasets so we can make data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# keep shuffle false for the test set for reproducibility
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your Sequential Neural Network
input_dim = X_train.shape[1]
output_dim = 3

model = nn.Sequential(
    nn.Linear(input_dim, 8),
    nn.ReLU(),
    nn.Linear(8, 6),
    nn.ReLU(),
    nn.Linear(6, output_dim),
)

# model = nn.Sequential(
#     nn.Linear(input_dim, 6),
#     nn.ReLU(),
#     nn.Linear(6, output_dim)
# )
print(model)

# Loss Function, Optimizer, and Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 75
train_losses = []

start_time = time.time()
for epoch in range(n_epochs):
    running_loss = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

plt.plot(train_losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Evaluation metrics

model.eval()
correct, total = 0, 0
predictions = []

start_time = time.time()
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = model(Xb)
        _, predicted = torch.max(preds.data, 1)
        predictions.extend(predicted.tolist())
        total += yb.size(0)
        print("total", total)
        correct += (predicted == yb).sum().item()
        print("correct", correct)

accuracy = 100 * correct / total
end_time = time.time()
print(f"Testing completed in {end_time - start_time:.8f} seconds.")
print(f"Test Accuracy: {accuracy:.2f}%")

# Complexity Analysis

# Example: re-train one epoch to capture runtime cost
start_time = time.time()
for Xb, yb in train_loader:
    optimizer.zero_grad()
    outputs = model(Xb)
    loss = criterion(outputs, yb)
    loss.backward()
    optimizer.step()
end_time = time.time()
print(f"One epoch completed in {end_time - start_time:.8f} seconds.")

# Experimentation Framework
# Try at least two modifications to a baseline and record
# See the markdown cell below for experiments and notes
