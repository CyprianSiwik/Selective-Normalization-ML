# Lightweight RNN model definition
import torch
import torch.nn as nn

class LightRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=64,
                 hidden_dim=64,
                 num_layers=1,
                 num_classes=2,
                 rnn_type='lstm',
                 bidirectional=False,
                 dropout=0.5,
                 pad_idx=0):
        """
        Lightweight RNN-based classifier for text with reduced dimensions.

        Args:
            vocab_size (int): Vocabulary size.
            embed_dim (int): Embedding dimension (reduced from 128 to 64).
            hidden_dim (int): RNN hidden state size (reduced from 128 to 64).
            num_layers (int): Number of RNN layers.
            num_classes (int): Number of output classes.
            rnn_type (str): One of ['rnn', 'lstm', 'gru'].
            bidirectional (bool): Whether to use bidirectional RNN.
            dropout (float): Dropout rate.
            pad_idx (int): Index for padding token in embedding.
        """
        super(LightRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type.lower()

        rnn_class = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }[self.rnn_type]

        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]

        if self.rnn_type == 'lstm':
            output, (hidden, _) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            if isinstance(hidden, tuple):  # LSTM
                hidden = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
            else:
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.fc(self.dropout(hidden))