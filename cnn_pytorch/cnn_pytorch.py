import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

"""
Use the following code block to get MNIST_MEAN and MNIST_STD used below.
# Accumulate sum and squared sum
mean = 0.0
std = 0.0
num_samples = 0

for images, _ in train_loader:
    batch_samples = images.size(0)  # batch size (1000)
    images = images.view(batch_samples, -1)  # flatten the 28x28 images
    mean += images.mean(1).sum()
    std += images.std(1).sum()
    num_samples += batch_samples
 
mean /= num_samples
std /= num_samples

print(f"Mean: {mean.item():.4f}, Std: {std.item():.4f}")
"""

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MNIST_IMAGE_LEN = 28
MNIST_NUM_CLASSES = 10
NUM_EPOCHS = 14


class NeuralNetwork(nn.Module):
    """
    A class for a generic neural network.
    """

    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # 3136 = 64*7*7
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, MNIST_NUM_CLASSES),
        )

    def forward(self, x):
        logits = self.conv_stack(x)
        return logits


# get the appropriate device, just set to cpu for this exercise
device = torch.device("cpu")
print(f"Using {device} device")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
)

# get the MNIST training and test sets
train_dataset = torchvision.datasets.MNIST(
    root="./mnist-train", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./mnist-test", train=False, download=True, transform=transform
)

# set up the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = NeuralNetwork().to(device)

# set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# train the network
epochs = NUM_EPOCHS
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# evaluate the model
model.eval()
test_loss = 0.0
correct = 0
total = 0

# disable gradient calculations
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # get predicted class
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}%")
# I was able to get an accuracy of 99.15% within 5 minutes of wallclock
# time with this script on an iMac M3 with 8GB of RAM.
