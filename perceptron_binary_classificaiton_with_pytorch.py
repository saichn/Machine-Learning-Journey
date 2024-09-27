"""
This module demonstrates a simple binary classification task using a perceptron implemented with PyTorch. It includes the following steps:

1. **Data Preparation**: Generates synthetic data using `sklearn.datasets.make_blobs` and splits it into training and testing sets.
2. **Model Definition**: Defines a neural network model (perceptron) with a single linear layer and a sigmoid activation function for binary classification.
3. **Helper Functions**: Includes functions to extract model parameters, plot the decision boundary, and visualize training losses.
4. **Training and Evaluation**: Trains the model using stochastic gradient descent and binary cross-entropy loss, and evaluates its performance on the test set.

The module provides a clear and concise example of how to implement and train a simple perceptron for binary classification tasks using PyTorch. It also includes visualizations to help understand the model's performance and decision boundaries.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
seed = 7777
torch.manual_seed(seed)
np.random.seed(seed)

# Data Preparation
# Generate sample data
X, y = datasets.make_blobs(n_samples=1000, centers=[(-1, 1), (1, -1)], n_features=2, cluster_std=0.8, random_state=seed)

# Convert numpy arrays to PyTorch tensors
x_data = torch.Tensor(X)
y_data = torch.Tensor(y).view(-1, 1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=seed)

# Model Definition
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        pred = self.linear(x)
        prob = self.sigmoid(pred)
        return prob

    def predict(self, x):
        pred = self.forward(x)
        return np.where(pred > 0.5, 1, 0)

# Helper Functions
def get_params(model):
    w = model.linear.weight.detach().numpy().flatten()
    b = model.linear.bias.detach().numpy().flatten()
    return w, b

def plot_fit(model, X, y, plot_title):
    plt.figure()
    [[w1, w2], b1] = get_params(model)
    x1 = np.array([X.min().item(), X.max().item()])
    x2 = -(x1 * w1 + b1) / w2
    plt.plot(x1, x2, 'r')
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(plot_title)
    plt.grid()
    plt.show()

def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch vs Training Loss')
    plt.grid()
    plt.show()

# Training and Evaluation
model = Model(2, 1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Plot initial decision boundary
plot_fit(model, x_train, y_train, 'Initial Fit')

# Training loop
epochs = 300
losses = []
for i in range(1, epochs + 1):
    prob = model(x_data)  # Forward pass
    loss = criterion(prob, y_data)  # Compute loss
    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters
    losses.append(loss.item())
    if i % 10 == 0 or i == 1:
        print(f'Epoch {i}, loss {loss:.4f}')

# Plot final decision boundary and training losses
plot_fit(model, x_train, y_train, 'Final Fit')
plot_losses(losses)

# Make predictions on the test set
y_test_pred = model.predict(x_test)

# Plot test set predictions
plot_fit(model, x_test, y_test_pred, 'Test Prediction')
