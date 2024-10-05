"""
### Project Description: Image Classification with LeNet

This project involves building and training convolutional neural networks (CNNs) using PyTorch to classify images from the CIFAR10 and MNIST datasets. The model architecture is inspired by the LeNet-5 network, which is well-suited for image classification tasks.

#### Key Components:

1. **Data Preparation**:
   - **Transformations**: 
     - For CIFAR10: Images are resized to 32x32 pixels, with additional data augmentation techniques such as random cropping, horizontal flipping, rotation, affine transformations, and color jittering to improve model robustness.
     - For MNIST: Images are resized to 28x28 pixels, converted to tensors, and normalized to the range [-1, 1].
   - **Datasets**: The CIFAR10 and MNIST training and test datasets are loaded and transformed using `torchvision.datasets` and `torchvision.transforms`.
   - **Data Loaders**: Data loaders are created for efficient batch processing during training and testing.

2. **Model Architecture**:
   - **Convolutional Layers**: 
     - For CIFAR10: Four convolutional layers with ReLU activation, batch normalization, and max pooling.
       - `conv1`: 3 input channels, 32 output channels, 3x3 kernel.
       - `conv2`: 32 input channels, 64 output channels, 3x3 kernel.
       - `conv3`: 64 input channels, 128 output channels, 3x3 kernel.
       - `conv4`: 128 input channels, 256 output channels, 3x3 kernel.
     - For MNIST: Two convolutional layers with ReLU activation and max pooling.
       - `conv1`: 1 input channel, 30 output channels, 5x5 kernel.
       - `conv2`: 30 input channels, 50 output channels, 5x5 kernel.
   - **Fully Connected Layers**: Two fully connected layers with dropout for regularization.
     - For CIFAR10: 
       - `fc1`: 256*2*2 input features, 512 output features.
       - `fc2`: 512 input features, 10 output features (one for each class).
     - For MNIST:
       - `fc1`: 50*4*4 input features, 128 output features.
       - `fc2`: 128 input features, 10 output features (one for each digit class).

3. **Training and Evaluation**:
   - **Loss Function**: Cross-entropy loss is used to measure the model's performance.
   - **Optimizer**: 
     - For CIFAR10: SGD optimizer with momentum and weight decay is employed.
     - For MNIST: Adam optimizer is employed.
   - **Learning Rate Scheduler**: A StepLR scheduler is used to adjust the learning rate during training.
   - **Training Loop**: The model is trained for a specified number of epochs, with training and testing phases in each epoch.
     - **Training Phase**: The model learns from the training data, and the loss and accuracy are recorded.
     - **Testing Phase**: The model's performance is evaluated on the test data, and incorrectly predicted images are collected for analysis.

4. **Visualization**:
   - **Image Display**: Helper functions are provided to display images, including incorrectly predicted ones.
   - **Training Curves**: Loss and accuracy curves for both training and testing phases are plotted to visualize the model's learning progress.

5. **Device Utilization**:
   - The model and data are moved to a GPU if available, ensuring faster computation.

This project demonstrates the application of CNNs in image classification, providing a comprehensive workflow from data preprocessing to model evaluation and visualization. It includes advanced techniques such as data augmentation, regularization, and learning rate scheduling to improve model performance and generalization.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(7777)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enhanced data augmentation for training data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),  # Randomly rotate the image
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Apply random affine transformations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change brightness, contrast, and saturation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize the image
])

# Transform for test data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize the image
])

# Load and transform the CIFAR10 training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# Load and transform the CIFAR10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Class labels for CIFAR10
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_images(images, labels, predictions=None, color=None):
    """
    Display a batch of images with their labels and predictions.
    """
    fig = plt.figure(figsize=(25, 4))
    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        plt.imshow(img.transpose(1, 2, 0))
        title = f'Label: {classes[label.item()]}'
        if predictions is not None:
            title += f' Pred: {classes[predictions[idx].item()]}'
        ax.set_title(title, color=color if color else 'black')
    plt.show()

def unnormalize(img, mean, std):
    """
    Unnormalize an image.
    """
    return img.detach().cpu().numpy() * std + mean

def plot_curves(train_data, test_data, xlabel, ylabel, title):
    """
    Plot training and test curves.
    """
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

class LeNet(nn.Module):
    """
    Define the LeNet neural network model.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool2d = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256*2*2, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool2d(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2d(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool2d(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

epochs = 100
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

best_accuracy = 0.0
for epoch in range(1, epochs + 1):
    train_loss, train_corrects = 0.0, 0.0
    test_loss, test_corrects = 0.0, 0.0

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

    model.eval()
    incorrect_images, incorrect_preds, correct_labels = [], [], []
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

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_corrects / len(train_loader.dataset))
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_corrects / len(test_loader.dataset))

    if test_accuracies[-1] > best_accuracy:
        best_accuracy = test_accuracies[-1]
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch}:")
    show_images(unnormalized_incorrect_images[:20], correct_labels[:20], incorrect_preds[:20], color='red')
    print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    scheduler.step()

plot_curves(train_losses, test_losses, 'Epochs', 'Loss', 'Epochs vs Loss')
plot_curves(train_accuracies, test_accuracies, 'Epochs', 'Accuracy', 'Epochs vs Accuracy')
