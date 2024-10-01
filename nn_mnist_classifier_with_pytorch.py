"""
### Name: MNIST Neural Network Classifier

### Description:
This Python script implements a neural network classifier for the MNIST dataset using PyTorch. The code includes data preprocessing, model definition, training, and evaluation. Key features include:

- **Data Transformation**: Resizes images to 28x28 pixels, converts them to tensors, and normalizes them to the range [-1, 1].
- **Model Architecture**: Defines a neural network with two hidden layers using ReLU activation functions.
- **Training and Evaluation**: Includes functions to train the model for multiple epochs and evaluate its performance on the test set.
- **Visualization**: Provides helper functions to display images and plot training/test loss and accuracy curves.
- **Error Analysis**: Displays incorrectly predicted images after each epoch to help diagnose model performance.

This script is a comprehensive example of building, training, and evaluating a neural network for image classification tasks.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define transformations for the input data
# Resize images to 28x28, convert to Tensor, and normalize to range [-1, 1]
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load and transform the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# Load and transform the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Helper function to display images
def show_images(images, labels, predictions=None, color=None):
    fig = plt.figure(figsize=(25, 4))
    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        plt.imshow(img.transpose(1, 2, 0))
        title = f'Label: {label.item()}'
        if predictions is not None:
            title += f' Pred: {predictions[idx].item()}'
        ax.set_title(title, color=color if color else 'black')
    plt.show()

# Helper function to unnormalize images
def unnormalize(img, mean, std):
    return img.detach().numpy() * std + mean

# Helper function to plot training and test curves
def plot_curves(train_data, test_data, xlabel, ylabel, title):
    plt.figure()
    plt.plot(train_data, label='Train')
    plt.plot(test_data, label='Test')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

# Display a batch of unnormalized images
data_iter = iter(train_loader)
images, labels = next(data_iter)
unnormalized_images = [unnormalize(img, 0.5, 0.5) for img in images]
show_images(unnormalized_images[:20], labels[:20])

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here; CrossEntropyLoss applies SoftMax internally
        return x

# Initialize model, loss function, and optimizer
input_size = 784  # 28x28
output_size = 10
model = NeuralNetwork(input_size, 128, 64, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_corrects = 0.0, 0.0
    test_loss, test_corrects = 0.0, 0.0

    # Training phase
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)  # Flatten images
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_corrects += (preds == labels).sum().item()

    # Testing phase
    incorrect_images, incorrect_preds, correct_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_corrects += (preds == labels).sum().item()

            # Collect incorrectly predicted images
            incorrect_images.extend(images[preds != labels])
            incorrect_preds.extend(preds[preds != labels])
            correct_labels.extend(labels[preds != labels])

    # Unnormalize incorrectly predicted images
    unnormalized_incorrect_images = [unnormalize(img, 0.5, 0.5) for img in incorrect_images]

    # Calculate and store average losses and accuracies
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_corrects / len(train_loader.dataset))
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_corrects / len(test_loader.dataset))

    # Print epoch results and display incorrectly predicted images
    print(f"Epoch {epoch}:")
    show_images(unnormalized_incorrect_images[:20], correct_labels[:20], incorrect_preds[:20], color='red')
    print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

# Plot training and test curves
plot_curves(train_losses, test_losses, 'Epochs', 'Loss', 'Epochs vs Loss')
plot_curves(train_accuracies, test_accuracies, 'Epochs', 'Accuracy', 'Epochs vs Accuracy')
