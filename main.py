# Entry point for running experiments
import argparse
from methods.baseline import run_baseline
from methods.dropout import run_dropout
from methods.norm import run_norm
from methods.standard_combo import run_standard_combo
from methods.selective_norm import run_selective_norm


def main():
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'dropout', 'norm', 'standard_combo', 'selective'],
                        help='Method to run: baseline, dropout, norm, standard_combo, selective')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'imdb', 'uci_adult'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'rnn'],
                        help='Model architecture to use')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight model variants')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (used by dropout, standard_combo, selective methods)')
    parser.add_argument('--normalization', type=str, default='batch',
                        choices=['batch', 'layer', 'group'],
                        help='Normalization type (used by norm, standard_combo methods)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_file', type=str, default='training_log.csv', help='Path to CSV log file')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory for saving plots')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    common_kwargs = {
        'model_type': args.model,
        'dataset': args.dataset,
        'lightweight': args.lightweight,
        'epochs': args.epochs,
        'lr': args.lr,
        'log_file': args.log_file,
        'plot_dir': args.plot_dir,
        'seed': args.seed,
    }

    if args.method == 'baseline':
        run_baseline(**common_kwargs)
    elif args.method == 'dropout':
        run_dropout(dropout_rate=args.dropout_rate, **common_kwargs)
    elif args.method == 'norm':
        run_norm(normalization=args.normalization, **common_kwargs)
    elif args.method == 'standard_combo':
        run_standard_combo(dropout_rate=args.dropout_rate, normalization=args.normalization, **common_kwargs)
    elif args.method == 'selective':
        run_selective_norm(dropout_rate=args.dropout_rate, **common_kwargs)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == '__main__':
    main()
