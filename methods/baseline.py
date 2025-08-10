# Baseline method (no dropout, no norm)
from models.cnn import CNN
from models.mlp import MLP
from models.rnn import RNNClassifier
from models.light_cnn import LightCNN
from models.light_mlp import LightMLP
from models.light_rnn import LightRNN

from data import cifar
from data import mnist
from data import imdb
from data import uci_adult

from train import train


def run_baseline(model_type='cnn', dataset='cifar10', lightweight=False, batch_size=64):
    """
    Run baseline method (no dropout, no normalization).

    Args:
        model_type (str): 'cnn', 'mlp', or 'rnn'
        dataset (str): Dataset name
        lightweight (bool): Use lightweight model variants if True
    """
    input_channels = None
    input_size = None
    num_classes = None
    vocab_size = None
    train_loader, test_loader = None, None

    # === Dataset Loading Only ===
    if dataset == 'cifar10':
        train_loader = cifar.get_cifar_dataset('cifar10', train=True)
        test_loader = cifar.get_cifar_dataset('cifar10', train=False)
        input_channels = 3
        num_classes = 10

    elif dataset == 'cifar100':
        train_loader = cifar.get_cifar_dataset('cifar100', train=True)
        test_loader = cifar.get_cifar_dataset('cifar100', train=False)
        input_channels = 3
        num_classes = 100

    elif dataset == 'mnist':
        train_loader = mnist.get_mnist_dataset(train=True)
        test_loader = mnist.get_mnist_dataset(train=False)
        input_size = 784
        num_classes = 10

    elif dataset == 'imdb':
        train_loader, test_loader, vocab = imdb.get_imdb_dataset()
        vocab_size = len(vocab)
        num_classes = 2

    elif dataset == 'uci_adult':
        train_loader, test_loader, _, _ = uci_adult.get_adult_dataloaders()
        input_size = 105
        num_classes = 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # === Model Selection ===
    if model_type == 'cnn':
        if input_channels is None or num_classes is None:
            raise ValueError("CNN model requires image dataset with input_channels and num_classes.")

        if lightweight:
            model = LightCNN(input_channels=input_channels, num_classes=num_classes, dropout=0.0)
        else:
            model = CNN(input_channels=input_channels, num_classes=num_classes, dropout=0.0)

    elif model_type == 'mlp':
        if input_size is None or num_classes is None:
            raise ValueError("MLP model requires input_size and num_classes.")

        if lightweight:
            model = LightMLP(input_size=input_size, hidden_sizes=[128, 64], num_classes=num_classes, dropout=0.0)
        else:
            model = MLP(input_size=input_size, hidden_sizes=[512, 256], num_classes=num_classes, dropout=0.0)

    elif model_type == 'rnn':
        if vocab_size is None or num_classes is None:
            raise ValueError("RNN model requires text dataset with vocab_size and num_classes.")

        if lightweight:
            model = LightRNN(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=64,
                num_classes=num_classes,
                dropout=0.0,
                bidirectional=False
            )
        else:
            model = RNNClassifier(
                vocab_size=vocab_size,
                embed_dim=128,
                hidden_dim=256,
                num_classes=num_classes,
                dropout=0.0,
                bidirectional=False
            )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # === Train the model ===
    train(model, train_loader, test_loader, epochs=10)


if __name__ == '__main__':
    # Test standard models
    run_baseline(model_type='cnn', dataset='cifar10', lightweight=False)

    # Test lightweight models
    run_baseline(model_type='cnn', dataset='cifar10', lightweight=True)