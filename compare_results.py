# compare_results.py — Aggregate a run_experiments.py sweep into a summary
# table and comparison plots, answering the README's question: does
# selective normalization actually help relative to the other four methods?
#
# The comparison that actually tests the hypothesis is selective vs.
# standard_combo (both use dropout + normalization; they differ only in
# whether normalization stats are computed before or after excluding
# dropped neurons), and specifically how their accuracy gap moves as
# dropout_rate increases — see plot_dropout_gap() and
# paired_significance_tests().
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


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
        'dropout_rate': entry.get('dropout_rate'),
        'normalization': entry.get('normalization'),
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


def plot_comparison(df, model, dataset, dropout_rate, plot_dir):
    """Overlay test-accuracy-vs-epoch curves for every method, at a fixed
    dropout_rate. Methods that don't use dropout_rate (baseline, norm) are
    included in every dropout_rate's plot as a constant reference."""
    subset = df[(df['model'] == model) & (df['dataset'] == dataset) &
                ((df['dropout_rate'] == dropout_rate) | df['dropout_rate'].isna())]
    if subset.empty:
        return

    plt.figure()
    for method in subset['method'].unique():
        method_runs = subset[subset['method'] == method]
        for _, row in method_runs.iterrows():
            run_df = pd.read_csv(row['log_file'])
            label = method
            if len(method_runs) > 1:
                extra = [f"seed={row['seed']}"]
                if pd.notna(row['normalization']):
                    extra.append(f"norm={row['normalization']}")
                label = f"{method} ({', '.join(extra)})"
            plt.plot(run_df['Epoch'], run_df['Test Acc (%)'], label=label)

    dr_suffix = f"_dr{dropout_rate}" if dropout_rate is not None else ""
    title_suffix = f" (dropout_rate={dropout_rate})" if dropout_rate is not None else ""
    plt.title(f'Test Accuracy by Method — {model}/{dataset}{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'compare_{model}_{dataset}{dr_suffix}.png'))
    plt.close()


def plot_dropout_gap(summary_df, model, dataset, plot_dir):
    """
    The key hypothesis-testing plot: (selective − standard_combo) best test
    accuracy vs. dropout_rate, averaged across seeds with error bars. The
    README's claimed failure mode (norm stats distorted by post-dropout
    zeros) should get worse as dropout_rate increases, so a gap that grows
    with dropout_rate is the clearest evidence *for* the hypothesis; a flat
    or near-zero gap across the whole sweep is evidence *against* it.
    """
    subset = summary_df[(summary_df['model'] == model) & (summary_df['dataset'] == dataset)]
    sel = subset[(subset['method'] == 'selective') & subset['dropout_rate'].notna()]
    std = subset[(subset['method'] == 'standard_combo') & subset['dropout_rate'].notna()]
    if sel.empty or std.empty:
        return

    merged = pd.merge(
        sel[['dropout_rate', 'seed', 'best_test_acc']],
        std[['dropout_rate', 'seed', 'best_test_acc']],
        on=['dropout_rate', 'seed'], suffixes=('_selective', '_standard_combo'),
    )
    if merged.empty:
        return
    merged['gap'] = merged['best_test_acc_selective'] - merged['best_test_acc_standard_combo']

    grouped = merged.groupby('dropout_rate')['gap'].agg(['mean', 'std', 'count']).reset_index()
    if len(grouped) < 2:
        print(f"  [{model}/{dataset}] only one dropout_rate present — need >=2 "
              f"(via --dropout_rates) to show a gap-vs-dropout-rate trend. Skipping plot.")
        return

    plt.figure()
    plt.errorbar(grouped['dropout_rate'], grouped['mean'], yerr=grouped['std'].fillna(0),
                 marker='o', capsize=4)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f'Selective − Standard-Combo Accuracy Gap vs Dropout Rate — {model}/{dataset}')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Best Test Acc Gap (percentage points)')
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'dropout_gap_{model}_{dataset}.png'))
    plt.close()


def paired_significance_tests(summary_df):
    """
    Paired t-test (selective vs. standard_combo, paired by seed so each pair
    shares the same init/data order) at each (model, dataset, dropout_rate).
    Needs >=2 seeds in common to produce a p-value.
    """
    rows = []
    has_dr = summary_df['dropout_rate'].notna()
    for (model, dataset, dr), group in summary_df[has_dr].groupby(['model', 'dataset', 'dropout_rate']):
        sel = group[group['method'] == 'selective'].set_index('seed')['best_test_acc']
        std = group[group['method'] == 'standard_combo'].set_index('seed')['best_test_acc']
        common_seeds = sel.index.intersection(std.index)
        if len(common_seeds) < 2:
            continue
        sel_vals = sel.loc[common_seeds].values
        std_vals = std.loc[common_seeds].values
        t_stat, p_value = stats.ttest_rel(sel_vals, std_vals)
        rows.append({
            'model': model, 'dataset': dataset, 'dropout_rate': dr, 'n_seeds': len(common_seeds),
            'mean_selective_acc': sel_vals.mean(), 'mean_standard_combo_acc': std_vals.mean(),
            'mean_gap': (sel_vals - std_vals).mean(), 't_stat': t_stat, 'p_value': p_value,
        })
    return pd.DataFrame(rows)


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

    display_cols = ['method', 'model', 'dataset', 'dropout_rate', 'normalization', 'seed',
                     'final_test_acc', 'best_test_acc', 'epochs_to_95pct_best',
                     'avg_epoch_time_s', 'avg_grad_norm', 'peak_memory_mb']
    print(summary_df[display_cols].to_string(index=False))

    # Average across seeds for a per-(method, model, dataset, dropout_rate) ranking view.
    grouped = summary_df.groupby(['model', 'dataset', 'method', 'dropout_rate'], dropna=False, as_index=False).agg(
        mean_final_test_acc=('final_test_acc', 'mean'),
        mean_best_test_acc=('best_test_acc', 'mean'),
        std_best_test_acc=('best_test_acc', 'std'),
        mean_epochs_to_95pct_best=('epochs_to_95pct_best', 'mean'),
        n_seeds=('seed', 'count'),
    )
    grouped_csv = os.path.join(args.results_dir, 'summary_by_method.csv')
    grouped.to_csv(grouped_csv, index=False)
    print(f"\nWrote per-method summary (averaged across seeds): {grouped_csv}\n")
    print(grouped.sort_values(['model', 'dataset', 'mean_best_test_acc'], ascending=[True, True, False]).to_string(index=False))

    # The hypothesis test: paired significance + accuracy-gap-vs-dropout-rate.
    sig_df = paired_significance_tests(summary_df)
    if sig_df.empty:
        print("\nNo paired significance tests run — need >=2 seeds with both 'selective' and "
              "'standard_combo' at a shared dropout_rate (use --seeds 0 1 2 ...).")
    else:
        sig_csv = os.path.join(args.results_dir, 'significance_selective_vs_standard_combo.csv')
        sig_df.to_csv(sig_csv, index=False)
        print(f"\nPaired significance tests (selective vs. standard_combo), wrote: {sig_csv}\n")
        print(sig_df.to_string(index=False))

    plot_dir = os.path.join(args.results_dir, 'comparison')
    for model, dataset in summary_df[['model', 'dataset']].drop_duplicates().itertuples(index=False):
        dr_values = summary_df.loc[
            (summary_df['model'] == model) & (summary_df['dataset'] == dataset), 'dropout_rate'
        ].dropna().unique()
        for dr in (dr_values if len(dr_values) else [None]):
            plot_comparison(summary_df, model, dataset, dr, plot_dir)
        print(f"\n[{model}/{dataset}] dropout-rate gap analysis:")
        plot_dropout_gap(summary_df, model, dataset, plot_dir)

    print(f"\nWrote comparison plots to: {plot_dir}")


if __name__ == '__main__':
    main()
