import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

def train_model(model, num_epochs, train_dataset, learning_rate):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            images = images.permute(0, 3, 1, 2)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
class CNNModel(nn.Module):
    def __init__(self, input_channels=4, input_height=158, input_width=300):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically calculate flattened size
        self.flatten_size = self._get_flatten_size(input_channels, input_height, input_width)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def _get_flatten_size(self, channels, height, width):
        # Simulate forward pass to calculate the size
        x = torch.zeros(1, channels, height, width)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        return x.numel()

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Sigmoid activation for binary classification
        x = torch.sigmoid(x)
        return x