# CNN model definition
import torch
import torch.nn as nn
import torch.nn.functional as F




class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=0.5):
        """
        A simple CNN architecture.

        Args:
            input_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR).
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14] (or [B, 32, 16, 16])
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]  (or [B, 64, 8, 8])
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
