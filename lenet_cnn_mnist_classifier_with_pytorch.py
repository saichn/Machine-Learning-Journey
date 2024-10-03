"""
### Project Description: MNIST Digit Classification with LeNet

This project involves building and training a convolutional neural network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model architecture is inspired by the LeNet-5 network, which is well-suited for image classification tasks.

#### Key Components:

1. **Data Preparation**:
   - **Transformations**: Images are resized to 28x28 pixels, converted to tensors, and normalized to the range [-1, 1].
   - **Datasets**: The MNIST training and test datasets are loaded and transformed using `torchvision.datasets` and `torchvision.transforms`.
   - **Data Loaders**: Data loaders are created for efficient batch processing during training and testing.

2. **Model Architecture**:
   - **Convolutional Layers**: Two convolutional layers with ReLU activation and max pooling.
     - `conv1`: 1 input channel, 30 output channels, 5x5 kernel.
     - `conv2`: 30 input channels, 50 output channels, 5x5 kernel.
   - **Fully Connected Layers**: Two fully connected layers with dropout for regularization.
     - `fc1`: 50*4*4 input features, 128 output features.
     - `fc2`: 128 input features, 10 output features (one for each digit class).

3. **Training and Evaluation**:
   - **Loss Function**: Cross-entropy loss is used to measure the model's performance.
   - **Optimizer**: Adam optimizer is employed for updating the model parameters.
   - **Training Loop**: The model is trained for 10 epochs, with training and testing phases in each epoch.
     - **Training Phase**: The model learns from the training data, and the loss and accuracy are recorded.
     - **Testing Phase**: The model's performance is evaluated on the test data, and incorrectly predicted images are collected for analysis.

4. **Visualization**:
   - **Image Display**: Helper functions are provided to display images, including incorrectly predicted ones.
   - **Training Curves**: Loss and accuracy curves for both training and testing phases are plotted to visualize the model's learning progress.

5. **Device Utilization**:
   - The model and data are moved to a GPU if available, ensuring faster computation.

This project demonstrates the application of CNNs in image classification, providing a comprehensive workflow from data preprocessing to model evaluation and visualization.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST greyscale images has 1 channel with size 28x28. 1x28x28
        # conv2d arguments - input channels, no.of filters/output_channels, kernel_size, stride
        # Larger stride length might result in less effective feature extraction
        self.conv1 = nn.Conv2d(1, 30, 5, 1)   # 1x28x28 --> (30 1x5x5 Kernels) --> 30x24x24 --> (2x2 max pool) --> 30x12x12
        self.conv2 = nn.Conv2d(30, 50, 5, 1)  # 30x12x12 --> (50 30x5x5 Kernels) --> 50x8x8 --> (2x2 max pooling) --> 50x4x4
        self.maxpool2d = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(50*4*4, 128)
        self.dropout1 = nn.Dropout(0.5)  # Helps avoid overfitting and uniform learning in model. Recommended to use in the layers with high no.of parameters
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool2d(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x) # Return value shouldn't apply any activation fucntions on it, as CrossEntropyLoss funciton applies SoftMax internally
        return x

model = LeNet().to(device)
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
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
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
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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
