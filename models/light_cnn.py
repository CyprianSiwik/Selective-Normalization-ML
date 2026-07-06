# Lightweight CNN model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.norm_utils import make_norm, apply_dropout_norm


class LightCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=0.5, norm=None, norm_order='after_dropout'):
        """
        A lightweight CNN architecture with ~50-70% fewer parameters than the standard CNN.

        Args:
            input_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR).
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            norm (str or None): 'batch', 'layer', 'group', 'selective', or None.
            norm_order (str): 'before_dropout' or 'after_dropout'.
        """
        super(LightCNN, self).__init__()
        self.norm_type = norm
        self.norm_order = norm_order

        # Reduced channel sizes: 32->16, 64->32
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = make_norm(norm, 32, spatial=True, dropout_rate=dropout)

        # Smaller fully connected layer: 128->64
        self.fc1 = nn.Linear(32 * 7 * 7 if input_channels == 1 else 32 * 8 * 8, 64)
        self.norm2 = make_norm(norm, 64, spatial=False, dropout_rate=dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 14, 14] (or [B, 16, 16, 16])
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 7, 7]  (or [B, 32, 8, 8])
        x = apply_dropout_norm(x, self.dropout, self.norm1, self.norm_type, self.norm_order)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = apply_dropout_norm(x, self.dropout, self.norm2, self.norm_type, self.norm_order)
        x = self.fc2(x)
        return x
