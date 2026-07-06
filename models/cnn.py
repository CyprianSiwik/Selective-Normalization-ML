# CNN model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.norm_utils import make_norm, apply_dropout_norm


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=0.5, norm=None, norm_order='after_dropout'):
        """
        A simple CNN architecture.

        Args:
            input_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR).
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            norm (str or None): 'batch', 'layer', 'group', 'selective', or None.
            norm_order (str): 'before_dropout' or 'after_dropout' - relative order of
                normalization and dropout at each site (ignored when norm='selective',
                since that layer performs both together).
        """
        super(CNN, self).__init__()
        self.norm_type = norm
        self.norm_order = norm_order

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = make_norm(norm, 64, spatial=True, dropout_rate=dropout)

        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.norm2 = make_norm(norm, 128, spatial=False, dropout_rate=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14] (or [B, 32, 16, 16])
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]  (or [B, 64, 8, 8])
        x = apply_dropout_norm(x, self.dropout, self.norm1, self.norm_type, self.norm_order)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = apply_dropout_norm(x, self.dropout, self.norm2, self.norm_type, self.norm_order)
        x = self.fc2(x)
        return x
