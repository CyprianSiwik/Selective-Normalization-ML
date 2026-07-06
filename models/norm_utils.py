# Shared helpers for building/applying normalization layers across models.
import torch.nn as nn

from models.selective_norm import SelectiveNormalization


def make_norm(norm_type, num_features, spatial=False, dropout_rate=0.5):
    """
    Build a normalization layer for a given site in a model.

    Args:
        norm_type (str or None): one of None, 'batch', 'layer', 'group', 'selective'.
        num_features (int): channels (spatial=True) or features (spatial=False).
        spatial (bool): True for 4D conv activations, False for 2D (B, F).
        dropout_rate (float): only used when norm_type == 'selective', since that
            layer performs dropout and normalization together.
    """
    if norm_type is None or norm_type == 'none':
        return nn.Identity()
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_features) if spatial else nn.BatchNorm1d(num_features)
    if norm_type == 'layer':
        return nn.GroupNorm(1, num_features) if spatial else nn.LayerNorm(num_features)
    if norm_type == 'group':
        groups = min(8, num_features)
        while groups > 1 and num_features % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_features)
    if norm_type == 'selective':
        return SelectiveNormalization(num_features, dropout_rate=dropout_rate, spatial=spatial)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def apply_dropout_norm(x, dropout, norm, norm_type, norm_order):
    """
    Apply dropout and normalization at a site, in the configured order.

    'selective' norm performs dropout internally (it needs the dropout mask
    to know which activations to normalize over), so the standalone dropout
    module is skipped in that case.
    """
    if norm_type == 'selective':
        return norm(x)
    if norm_order == 'before_dropout':
        return dropout(norm(x))
    return norm(dropout(x))
