# compare_results.py — Aggregate a run_experiments.py sweep into a summary
# table and comparison plots, answering the README's question: does
# selective normalization actually help relative to the other four methods?
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def epochs_to_threshold(test_accs, fraction=0.95):
    """Epoch (1-indexed) at which test accuracy first reaches `fraction` of
    the run's own best accuracy - a simple proxy for convergence speed."""
    if not test_accs:
        return None
    target = fraction * max(test_accs)
    for i, acc in enumerate(test_accs, start=1):
        if acc >= target:
            return i
    return len(test_accs)


def summarize_run(entry):
    df = pd.read_csv(entry['log_file'])
    test_accs = df['Test Acc (%)'].tolist()
    return {
        'method': entry['method'],
        'model': entry['model'],
        'dataset': entry['dataset'],
        'seed': entry['seed'],
        'final_test_acc': test_accs[-1],
        'best_test_acc': max(test_accs),
        'epochs_to_95pct_best': epochs_to_threshold(test_accs, 0.95),
        'avg_epoch_time_s': df['Epoch Time (s)'].mean(),
        'avg_grad_norm': df['Grad Norm'].mean(),
        'avg_activation_std': df['Activation Std'].mean(),
        'peak_memory_mb': df['Peak Memory (MB)'].max(),
        'avg_inference_time_s': df['Inference Time (s/batch)'].mean(),
        'log_file': entry['log_file'],
    }


def plot_comparison(df, model, dataset, plot_dir):
    subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
    if subset.empty:
        return

    plt.figure()
    for method in subset['method'].unique():
        method_runs = subset[subset['method'] == method]
        for _, row in method_runs.iterrows():
            run_df = pd.read_csv(row['log_file'])
            label = method if len(method_runs) == 1 else f"{method} (seed={row['seed']})"
            plt.plot(run_df['Epoch'], run_df['Test Acc (%)'], label=label)

    plt.title(f'Test Accuracy by Method — {model}/{dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'compare_{model}_{dataset}.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Aggregate a run_experiments.py sweep into a comparison report.')
    parser.add_argument('--results_dir', type=str, default='results', help='Root directory produced by run_experiments.py.')
    args = parser.parse_args()

    manifest_path = os.path.join(args.results_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    ok_runs = [r for r in manifest['runs'] if r['status'] == 'ok']
    failed_runs = [r for r in manifest['runs'] if r['status'] != 'ok']
    if failed_runs:
        print(f"{len(failed_runs)} run(s) failed and are excluded from the comparison:")
        for r in failed_runs:
            print(f"  - {r['method']}/{r['model']}/{r['dataset']} seed={r['seed']}: {r.get('error')}")

    if not ok_runs:
        print("No successful runs to summarize.")
        return

    summary_rows = [summarize_run(entry) for entry in ok_runs]
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(args.results_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote summary table: {summary_csv}\n")

    display_cols = ['method', 'model', 'dataset', 'seed', 'final_test_acc', 'best_test_acc',
                     'epochs_to_95pct_best', 'avg_epoch_time_s', 'avg_grad_norm', 'peak_memory_mb']
    print(summary_df[display_cols].to_string(index=False))

    # Average across seeds for a per-(method, model, dataset) ranking view.
    grouped = summary_df.groupby(['model', 'dataset', 'method'], as_index=False).agg(
        mean_final_test_acc=('final_test_acc', 'mean'),
        mean_best_test_acc=('best_test_acc', 'mean'),
        mean_epochs_to_95pct_best=('epochs_to_95pct_best', 'mean'),
        n_seeds=('seed', 'count'),
    )
    grouped_csv = os.path.join(args.results_dir, 'summary_by_method.csv')
    grouped.to_csv(grouped_csv, index=False)
    print(f"\nWrote per-method summary (averaged across seeds): {grouped_csv}\n")
    print(grouped.sort_values(['model', 'dataset', 'mean_best_test_acc'], ascending=[True, True, False]).to_string(index=False))

    plot_dir = os.path.join(args.results_dir, 'comparison')
    for model, dataset in summary_df[['model', 'dataset']].drop_duplicates().itertuples(index=False):
        plot_comparison(summary_df, model, dataset, plot_dir)
    print(f"\nWrote comparison plots to: {plot_dir}")


if __name__ == '__main__':
    main()
