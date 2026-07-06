# run_experiments.py — Sweep all methods across a model/dataset matrix and
# organize results so they can be compared (see compare_results.py).
#
# Valid (model, dataset) pairs, since not every architecture applies to every
# dataset: CNN needs images (mnist/cifar10/cifar100), RNN needs text (imdb),
# MLP works on anything with a fixed-size flattened input.
import argparse
import json
import os
import time
import traceback

from methods.baseline import run_baseline
from methods.dropout import run_dropout
from methods.norm import run_norm
from methods.standard_combo import run_standard_combo
from methods.selective_norm import run_selective_norm

METHODS = ['baseline', 'dropout', 'norm', 'standard_combo', 'selective']

VALID_PAIRS = {
    'cnn': ['mnist', 'cifar10', 'cifar100'],
    'mlp': ['mnist', 'uci_adult', 'cifar10', 'cifar100'],
    'rnn': ['imdb'],
}


def run_one(method, model_type, dataset, lightweight, epochs, lr, dropout_rate, normalization, seed, results_dir):
    run_name = f"{method}/{model_type}_{dataset}{'_light' if lightweight else ''}_seed{seed}"
    run_dir = os.path.join(results_dir, method, f"{model_type}_{dataset}{'_light' if lightweight else ''}_seed{seed}")
    log_file = os.path.join(run_dir, 'log.csv')
    plot_dir = os.path.join(run_dir, 'plots')

    common = dict(model_type=model_type, dataset=dataset, lightweight=lightweight,
                  epochs=epochs, lr=lr, log_file=log_file, plot_dir=plot_dir, seed=seed)

    if method == 'baseline':
        run_baseline(**common)
    elif method == 'dropout':
        run_dropout(dropout_rate=dropout_rate, **common)
    elif method == 'norm':
        run_norm(normalization=normalization, **common)
    elif method == 'standard_combo':
        run_standard_combo(dropout_rate=dropout_rate, normalization=normalization, **common)
    elif method == 'selective':
        run_selective_norm(dropout_rate=dropout_rate, **common)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return run_name, log_file


def main():
    parser = argparse.ArgumentParser(description='Sweep all methods across a model/dataset matrix.')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'uci_adult'],
                        choices=['mnist', 'uci_adult', 'cifar10', 'cifar100', 'imdb'],
                        help='Datasets to include in the sweep.')
    parser.add_argument('--models', nargs='+', default=['cnn', 'mlp', 'rnn'],
                        choices=['cnn', 'mlp', 'rnn'], help='Model architectures to include.')
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS,
                        help='Methods to include.')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model variants.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per run.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for dropout/standard_combo/selective.')
    parser.add_argument('--normalization', type=str, default='batch', choices=['batch', 'layer', 'group'],
                        help='Normalization type for norm/standard_combo.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Seeds to repeat each run with.')
    parser.add_argument('--results_dir', type=str, default='results', help='Root directory for sweep output.')

    args = parser.parse_args()

    manifest = {'runs': [], 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}

    for model_type in args.models:
        valid_datasets = [d for d in args.datasets if d in VALID_PAIRS.get(model_type, [])]
        skipped = set(args.datasets) - set(valid_datasets)
        for dataset in skipped:
            print(f"Skipping {model_type}/{dataset}: not a valid model/dataset pair.")

        for dataset in valid_datasets:
            for method in args.methods:
                for seed in args.seeds:
                    print(f"\n=== {method} | {model_type} | {dataset} | seed={seed} ===")
                    entry = {
                        'method': method, 'model': model_type, 'dataset': dataset,
                        'lightweight': args.lightweight, 'seed': seed,
                    }
                    start = time.time()
                    try:
                        run_name, log_file = run_one(
                            method, model_type, dataset, args.lightweight, args.epochs, args.lr,
                            args.dropout_rate, args.normalization, seed, args.results_dir,
                        )
                        entry['status'] = 'ok'
                        entry['log_file'] = log_file
                        entry['duration_s'] = time.time() - start
                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        entry['status'] = 'failed'
                        entry['error'] = str(exc)
                        entry['traceback'] = traceback.format_exc()
                        entry['duration_s'] = time.time() - start

                    manifest['runs'].append(entry)

    manifest['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(args.results_dir, exist_ok=True)
    manifest_path = os.path.join(args.results_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    n_ok = sum(1 for r in manifest['runs'] if r['status'] == 'ok')
    n_failed = len(manifest['runs']) - n_ok
    print(f"\nSweep complete: {n_ok} succeeded, {n_failed} failed. Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
