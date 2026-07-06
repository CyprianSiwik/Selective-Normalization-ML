#Utils

import torch
from torch.nn.utils.rnn import pad_sequence

def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
    else:
        raise ValueError("Unexpected batch format.")

    if isinstance(x, list):
        x = pad_sequence(x, batch_first=True)

    return x, y
