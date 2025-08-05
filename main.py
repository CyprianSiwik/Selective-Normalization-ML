# Entry point for running experiments
import argparse
from methods.baseline import run_baseline
from methods.dropout import run_dropout
from methods import selective_norm

def main():
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'dropout', 'selective'],
                        help='Method to run: baseline, dropout, selective')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'imdb', 'uci_adult'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'rnn'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_file', type=str, default='training_log.csv', help='Path to CSV log file')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory for saving plots')

    args = parser.parse_args()

    kwargs = {
        'model_type': args.model,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'log_file': args.log_file,
        'plot_dir': args.plot_dir
    }

    if args.method == 'baseline':
        run_baseline(**kwargs)
    elif args.method == 'dropout':
        run_dropout(**kwargs)
    elif args.method == 'selective':
        run_selective(**kwargs)
    elif args.method == 'lightweight':
        run_lightweight(**kwargs)
    elif args.method == 'norm':
        run_norm(**kwargs)
    elif args.method == 'standard_combo':
        run_standard_combo(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == '__main__':
    main()
