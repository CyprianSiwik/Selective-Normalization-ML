# compare_results.py — Aggregate a run_experiments.py sweep into a summary
# table and comparison plots, answering the README's question: does
# selective normalization actually help relative to the other four methods?
#
# The comparison that actually tests the hypothesis is selective vs.
# standard_combo (both use dropout + normalization; they differ only in
# whether normalization stats are computed before or after excluding
# dropped neurons), and specifically how their accuracy gap moves as
# dropout_rate increases — see plot_dropout_gap() and
# paired_significance_tests(). Beyond that headline number this module also
# tracks: training stability (grad-norm volatility), generalization
# (train/test accuracy gap), selective normalization's own train/eval
# activation mismatch, the compute/memory cost of selective relative to
# standard_combo, and multiple-comparison-corrected significance with an
# effect size.
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


def _col(df, name):
    """df[name] if the column exists, else a same-length all-NaN series.
    Keeps summarize_run() working on logs written before a metric was added."""
    if name in df.columns:
        return df[name]
    return pd.Series([float('nan')] * len(df))


def summarize_run(entry):
    df = pd.read_csv(entry['log_file'])
    test_accs = df['Test Acc (%)'].tolist()
    train_test_gap = _col(df, 'Train Acc (%)') - df['Test Acc (%)']
    activation_std = _col(df, 'Activation Std')
    eval_activation_std = _col(df, 'Eval Activation Std')
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
        # Training stability: epoch-to-epoch grad-norm volatility, and (if
        # train.py logged it) batch-to-batch volatility within an epoch.
        # A method that reaches the same accuracy with a wilder gradient
        # signal is still less stable, which is part of what the README's
        # hypothesis predicts for standard_combo at high dropout rates.
        'grad_norm_std_across_epochs': df['Grad Norm'].std(),
        'avg_grad_norm_std_within_epoch': _col(df, 'Grad Norm Std').mean(),
        'avg_activation_std': activation_std.mean(),
        'avg_eval_activation_std': eval_activation_std.mean(),
        # Selective normalization accumulates running stats only from
        # dropout survivors during training, but eval mode sees every
        # activation — this is the resulting train/eval mismatch, trackable
        # for every method so selective's own value can be compared to the
        # others' instead of assumed to be zero.
        'train_eval_activation_std_gap': (activation_std - eval_activation_std).abs().mean(),
        # Generalization: does the dropout/normalization interaction change
        # how much the model overfits, independent of raw test accuracy?
        'avg_train_test_acc_gap': train_test_gap.mean(),
        'final_train_test_acc_gap': train_test_gap.iloc[-1],
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

    standard_combo can be swept across several normalization types (batch/
    layer/group); each is compared against selective separately rather than
    pooled, since batch norm's cross-sample statistics make it the most
    exposed to the claimed distortion mechanism while layer/group norm
    (computed per-sample) may not show it at all — averaging them together
    would hide that difference.
    """
    subset = summary_df[(summary_df['model'] == model) & (summary_df['dataset'] == dataset)]
    sel = subset[(subset['method'] == 'selective') & subset['dropout_rate'].notna()]
    std = subset[(subset['method'] == 'standard_combo') & subset['dropout_rate'].notna()]
    if sel.empty or std.empty:
        return

    for normalization in sorted(std['normalization'].dropna().unique()):
        std_norm = std[std['normalization'] == normalization]
        merged = pd.merge(
            sel[['dropout_rate', 'seed', 'best_test_acc']],
            std_norm[['dropout_rate', 'seed', 'best_test_acc']],
            on=['dropout_rate', 'seed'], suffixes=('_selective', '_standard_combo'),
        )
        if merged.empty:
            continue
        merged['gap'] = merged['best_test_acc_selective'] - merged['best_test_acc_standard_combo']

        grouped = merged.groupby('dropout_rate')['gap'].agg(['mean', 'std', 'count']).reset_index()
        if len(grouped) < 2:
            print(f"  [{model}/{dataset}/norm={normalization}] only one dropout_rate present — need >=2 "
                  f"(via --dropout_rates) to show a gap-vs-dropout-rate trend. Skipping plot.")
            continue

        plt.figure()
        plt.errorbar(grouped['dropout_rate'], grouped['mean'], yerr=grouped['std'].fillna(0),
                     marker='o', capsize=4)
        plt.axhline(0, linestyle='--', color='gray')
        plt.title(f'Selective − Standard-Combo Accuracy Gap vs Dropout Rate\n{model}/{dataset} (norm={normalization})')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Best Test Acc Gap (percentage points)')
        plt.grid(True)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'dropout_gap_{model}_{dataset}_norm-{normalization}.png'))
        plt.close()


def _holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the same order as the input, each capped at 1.0. With e.g. 3 models x 5
    datasets x 5 dropout rates x 3 normalizations, uncorrected p-values will
    turn up "significant" results by chance alone; this controls the
    family-wise error rate across every test the sweep produced."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [None] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (n - rank) * p_values[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted


def paired_significance_tests(summary_df):
    """
    Paired t-test (selective vs. standard_combo, paired by seed so each pair
    shares the same init/data order) at each (model, dataset, dropout_rate,
    normalization). Needs >=2 seeds in common to produce a p-value. Also
    reports Cohen's d (paired) alongside the p-value, since with only a
    handful of seeds a "significant" result can still be a negligible
    effect size, and applies a Holm-Bonferroni correction across every test
    in the sweep so the significance claims survive multiple comparisons.
    """
    rows = []
    has_dr = summary_df['dropout_rate'].notna()
    for (model, dataset, dr), group in summary_df[has_dr].groupby(['model', 'dataset', 'dropout_rate']):
        sel = group[group['method'] == 'selective'].set_index('seed')['best_test_acc']
        std_group = group[group['method'] == 'standard_combo']
        for normalization in std_group['normalization'].dropna().unique():
            std = std_group[std_group['normalization'] == normalization].set_index('seed')['best_test_acc']
            common_seeds = sel.index.intersection(std.index)
            if len(common_seeds) < 2:
                continue
            sel_vals = sel.loc[common_seeds].values
            std_vals = std.loc[common_seeds].values
            diff = sel_vals - std_vals
            t_stat, p_value = stats.ttest_rel(sel_vals, std_vals)
            diff_std = diff.std(ddof=1)
            cohens_d = diff.mean() / diff_std if diff_std > 0 else float('nan')
            rows.append({
                'model': model, 'dataset': dataset, 'dropout_rate': dr, 'normalization': normalization,
                'n_seeds': len(common_seeds),
                'mean_selective_acc': sel_vals.mean(), 'mean_standard_combo_acc': std_vals.mean(),
                'mean_gap': diff.mean(), 't_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['p_value_holm'] = _holm_bonferroni(df['p_value'].tolist())
    return df


def cost_benefit_report(summary_df):
    """
    Cost side of the ledger: selective normalization's per-epoch time,
    per-batch inference time, and peak memory relative to standard_combo,
    reported alongside the accuracy gap at the same (model, dataset,
    dropout_rate, normalization). A gap that only shows up alongside a
    large compute/memory overhead is a different conclusion than a gap
    that comes for free.
    """
    has_dr = summary_df['dropout_rate'].notna()
    grouped = summary_df[has_dr].groupby(
        ['model', 'dataset', 'dropout_rate', 'method', 'normalization'], dropna=False
    ).agg(
        best_test_acc=('best_test_acc', 'mean'),
        avg_epoch_time_s=('avg_epoch_time_s', 'mean'),
        avg_inference_time_s=('avg_inference_time_s', 'mean'),
        peak_memory_mb=('peak_memory_mb', 'mean'),
    ).reset_index()

    sel = grouped[grouped['method'] == 'selective'].set_index(['model', 'dataset', 'dropout_rate'])
    std = grouped[grouped['method'] == 'standard_combo']

    rows = []
    for _, std_row in std.iterrows():
        key = (std_row['model'], std_row['dataset'], std_row['dropout_rate'])
        if key not in sel.index:
            continue
        sel_row = sel.loc[key]
        rows.append({
            'model': key[0], 'dataset': key[1], 'dropout_rate': key[2],
            'normalization': std_row['normalization'],
            'accuracy_gap_pp': sel_row['best_test_acc'] - std_row['best_test_acc'],
            'epoch_time_overhead_pct': 100 * (sel_row['avg_epoch_time_s'] / std_row['avg_epoch_time_s'] - 1),
            'inference_time_overhead_pct': 100 * (sel_row['avg_inference_time_s'] / std_row['avg_inference_time_s'] - 1),
            'peak_memory_overhead_pct': 100 * (sel_row['peak_memory_mb'] / std_row['peak_memory_mb'] - 1),
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
                     'avg_train_test_acc_gap', 'grad_norm_std_across_epochs',
                     'train_eval_activation_std_gap', 'avg_epoch_time_s', 'peak_memory_mb']
    print(summary_df[display_cols].to_string(index=False))

    # Average across seeds for a per-(method, model, dataset, dropout_rate,
    # normalization) ranking view. normalization is part of the grouping
    # key so multiple normalization types swept for norm/standard_combo
    # don't get silently averaged together.
    grouped = summary_df.groupby(
        ['model', 'dataset', 'method', 'dropout_rate', 'normalization'], dropna=False, as_index=False
    ).agg(
        mean_final_test_acc=('final_test_acc', 'mean'),
        mean_best_test_acc=('best_test_acc', 'mean'),
        std_best_test_acc=('best_test_acc', 'std'),
        mean_epochs_to_95pct_best=('epochs_to_95pct_best', 'mean'),
        mean_train_test_acc_gap=('avg_train_test_acc_gap', 'mean'),
        mean_grad_norm_std_across_epochs=('grad_norm_std_across_epochs', 'mean'),
        mean_train_eval_activation_std_gap=('train_eval_activation_std_gap', 'mean'),
        n_seeds=('seed', 'count'),
    )
    grouped_csv = os.path.join(args.results_dir, 'summary_by_method.csv')
    grouped.to_csv(grouped_csv, index=False)
    print(f"\nWrote per-method summary (averaged across seeds): {grouped_csv}\n")
    print(grouped.sort_values(['model', 'dataset', 'mean_best_test_acc'], ascending=[True, True, False]).to_string(index=False))

    # The hypothesis test: paired significance (with effect size and
    # multiple-comparison correction) + accuracy-gap-vs-dropout-rate.
    sig_df = paired_significance_tests(summary_df)
    if sig_df.empty:
        print("\nNo paired significance tests run — need >=2 seeds with both 'selective' and "
              "'standard_combo' at a shared dropout_rate (use --seeds 0 1 2 ...).")
    else:
        sig_csv = os.path.join(args.results_dir, 'significance_selective_vs_standard_combo.csv')
        sig_df.to_csv(sig_csv, index=False)
        print(f"\nPaired significance tests (selective vs. standard_combo), wrote: {sig_csv}\n")
        print(sig_df.to_string(index=False))

    # Cost side of the ledger: is any accuracy gap worth its compute/memory overhead?
    cost_df = cost_benefit_report(summary_df)
    if cost_df.empty:
        print("\nNo selective/standard_combo pairs available for a cost-benefit comparison.")
    else:
        cost_csv = os.path.join(args.results_dir, 'cost_benefit_selective_vs_standard_combo.csv')
        cost_df.to_csv(cost_csv, index=False)
        print(f"\nWrote cost-benefit comparison (selective vs. standard_combo), wrote: {cost_csv}\n")
        print(cost_df.to_string(index=False))

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
