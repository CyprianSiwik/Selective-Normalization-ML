# Selective normalization method: dropout + normalization computed only over
# the surviving (non-dropped) activations. See models/selective_norm.py for
# the SelectiveNormalization layer itself.
from models.cnn import CNN
from models.mlp import MLP
from models.rnn import RNNClassifier
from models.light_cnn import LightCNN
from models.light_mlp import LightMLP
from models.light_rnn import LightRNN

from train import train
import torch


def run_selective_norm(model_type='cnn', dataset='cifar10', dropout_rate=0.3, lightweight=False,
                        epochs=10, lr=0.001, log_file='training_log.csv', plot_dir='plots', seed=None):
    """
    Run training with dropout + selective normalization (norm only on active neurons).

    Args:
        model_type (str): 'cnn', 'mlp', or 'rnn'
        dataset (str): dataset name
        dropout_rate (float): dropout probability
        lightweight (bool): Use lightweight model variants if True
    """
    if seed is not None:
        torch.manual_seed(seed)

    input_channels = None
    num_classes = None
    input_size = None
    vocab_size = 0

    # === Dataset loading ===
    if dataset == 'cifar10':
        from data import cifar
        train_loader = cifar.get_cifar_dataset('cifar10', train=True)
        test_loader = cifar.get_cifar_dataset('cifar10', train=False)
        input_channels = 3
        num_classes = 10

    elif dataset == 'cifar100':
        from data import cifar
        train_loader = cifar.get_cifar_dataset('cifar100', train=True)
        test_loader = cifar.get_cifar_dataset('cifar100', train=False)
        input_channels = 3
        num_classes = 100

    elif dataset == 'mnist':
        from data import mnist
        train_loader = mnist.get_mnist_dataset(train=True)
        test_loader = mnist.get_mnist_dataset(train=False)
        input_channels = 1
        input_size = 784
        num_classes = 10

    elif dataset == 'imdb':
        from data import imdb
        train_loader, test_loader, vocab = imdb.get_imdb_dataset()
        vocab_size = len(vocab)
        num_classes = 2

    elif dataset == 'uci_adult':
        from data import uci_adult
        train_loader, test_loader, _, _ = uci_adult.get_adult_dataloaders()
        input_size = 105
        num_classes = 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # === Model Selection with Selective Normalization ===
    if model_type == 'cnn':
        if lightweight:
            model = LightCNN(input_channels=input_channels, num_classes=num_classes,
                              dropout=dropout_rate, norm='selective')
        else:
            model = CNN(input_channels=input_channels, num_classes=num_classes,
                         dropout=dropout_rate, norm='selective')

    elif model_type == 'mlp':
        if lightweight:
            model = LightMLP(input_size=input_size, hidden_sizes=[128, 64], num_classes=num_classes,
                              dropout=dropout_rate, norm='selective')
        else:
            model = MLP(input_size=input_size, hidden_sizes=[512, 256], num_classes=num_classes,
                         dropout=dropout_rate, norm='selective')

    elif model_type == 'rnn':
        if lightweight:
            model = LightRNN(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=64,
                num_classes=num_classes,
                dropout=dropout_rate,
                norm='selective',
                bidirectional=False
            )
        else:
            model = RNNClassifier(
                vocab_size=vocab_size,
                embed_dim=128,
                hidden_dim=256,
                num_classes=num_classes,
                dropout=dropout_rate,
                norm='selective',
                bidirectional=False
            )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # === Train ===
    return train(model, train_loader, test_loader, epochs=epochs, lr=lr, log_file=log_file, plot_dir=plot_dir)


if __name__ == '__main__':
    # Test standard models with selective normalization
    run_selective_norm(model_type='cnn', dataset='cifar10', dropout_rate=0.3, lightweight=False)

    # Test lightweight models with selective normalization
    run_selective_norm(model_type='mlp', dataset='mnist', dropout_rate=0.5, lightweight=True)
