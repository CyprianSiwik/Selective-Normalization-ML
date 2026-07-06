# SelectiveNormalization - normalizes only active (non-dropped) neurons.
#
# Standard dropout-then-normalize pipelines compute normalization statistics
# over the post-dropout tensor, treating the zeroed-out entries as real
# values. This layer instead computes per-feature mean/var only from the
# entries that survive dropout, then normalizes (and re-masks) the result.
import torch
import torch.nn as nn


class SelectiveNormalization(nn.Module):
    """
    Applies dropout and then normalizes only the surviving (non-dropped)
    activations, using per-feature statistics computed from those survivors.

    Args:
        num_features (int): Number of channels (spatial=True) or features (spatial=False).
        dropout_rate (float): Dropout probability.
        spatial (bool): True for 4D conv activations (B, C, H, W), False for 2D (B, F).
        eps (float): Numerical stability constant.
        momentum (float): Running-stat update rate, used for eval-mode normalization.
    """

    def __init__(self, num_features, dropout_rate=0.5, spatial=False, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        self.spatial = spatial
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def _reduce_dims(self):
        return (0, 2, 3) if self.spatial else (0,)

    def _broadcast_shape(self):
        return (1, self.num_features, 1, 1) if self.spatial else (1, self.num_features)

    def forward(self, x):
        shape = self._broadcast_shape()

        if not self.training:
            mean = self.running_mean.view(shape)
            var = self.running_var.view(shape)
            normalized = (x - mean) / torch.sqrt(var + self.eps)
            return normalized * self.weight.view(shape) + self.bias.view(shape)

        # Dropout mask, applied elementwise (same granularity as nn.Dropout).
        mask = (torch.rand_like(x) >= self.dropout_rate).float()
        x_dropped = x * mask

        # Per-feature statistics computed only from surviving activations.
        dims = self._reduce_dims()
        counts = mask.sum(dim=dims).clamp(min=1)
        mean = x_dropped.sum(dim=dims) / counts
        var = (x_dropped ** 2).sum(dim=dims) / counts - mean ** 2
        var = var.clamp(min=0)

        with torch.no_grad():
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)

        mean = mean.view(shape)
        var = var.view(shape)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        # Re-mask so dropped entries stay zero, matching standard dropout semantics.
        return mask * (normalized * self.weight.view(shape) + self.bias.view(shape))
