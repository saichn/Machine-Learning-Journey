import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(777)

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Function to extract model parameters
def get_params(model):
    w = model.linear.weight.detach().numpy().flatten()[0]
    b = model.linear.bias.detach().numpy().flatten()[0]
    return w, b

# Function to plot the linear fit
def plot_fit(model, X, Y, plot_title):
    plt.figure()
    w1, b1 = get_params(model)  # Get model parameters

    # Use any two points to draw a line connecting them using pyplot.
    x1 = np.array([X.min().item(), X.max().item()])
    y1 = x1 * w1 + b1  # Predictions made by the model

    plt.plot(x1, y1, 'r')  # Linear regression line
    plt.scatter(X.numpy(), Y.numpy())  # Input data
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(plot_title)
    plt.text(0, X.max().item()-1, f'Y = {w1:.4f} * X + {b1:.4f}' if b1 > 0 else f'Y = {w1:.4f} * X {b1:.4f}', color='Green')
    plt.grid()
    plt.show()

# Function to plot training losses
def plot_losses(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.show()

# Generate synthetic data
X = torch.randn(100, 1) * 10
Y = X + 3 * torch.randn(100, 1)

# Initialize model, loss function, and optimizer
model = LinearRegression(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Plot initial fit
plot_fit(model, X, Y, 'Initial Fit')

# Training loop
epochs = 10
losses = []

for i in range(1, epochs + 1):
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    losses.append(loss.item())
    
    optimizer.zero_grad()  # Zero out gradients from previous step
    loss.backward()  # Calculate gradients
    optimizer.step()  # Update the parameters

    # Plot fit at epoch 3
    if i == 3:
        print(f"Epoch: {i}, Loss: {loss.item():.4f}")
        plot_fit(model, X, Y, f'Fit at Epoch {i}')

# Plot final fit and training losses
plot_fit(model, X, Y, 'Final Fit')
plot_losses(losses)
