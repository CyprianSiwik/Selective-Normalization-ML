# MLP model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.norm_utils import make_norm, apply_dropout_norm


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], num_classes=10, dropout=0.5, norm=None,
                 norm_order='after_dropout'):
        """
        Simple MLP model.

        Args:
            input_size (int): Number of input features (e.g. 28*28=784 for MNIST).
            hidden_sizes (list): List of hidden layer sizes.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            norm (str or None): 'batch', 'layer', 'group', 'selective', or None.
            norm_order (str): 'before_dropout' or 'after_dropout'.
        """
        super(MLP, self).__init__()
        self.norm_type = norm
        self.norm_order = norm_order

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = input_size

        for hidden_dim in hidden_sizes:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append(make_norm(norm, hidden_dim, spatial=False, dropout_rate=dropout))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(in_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if necessary

        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(layer(x))
            x = apply_dropout_norm(x, self.dropout, norm, self.norm_type, self.norm_order)

        return self.output_layer(x)
