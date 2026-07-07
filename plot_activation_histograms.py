# plot_activation_histograms.py — Direct mechanistic evidence for or against
# the selective-normalization hypothesis: overlay the actual distribution of
# activations at the normalization study site for each method, at a given
# dropout rate, and quantify the selective-vs-standard_combo distortion as a
# scalar rather than something only eyeballed off a histogram.
#
# The README's claimed failure mode is that standard_combo normalizes over a
# tensor where dropped entries have been hard-zeroed, distorting the
# statistics. If that's really happening, standard_combo's histogram should
# show a distortion around the value where zeros land after normalization
# that selective's histogram (which excludes those entries from the
# normalization stats) doesn't show. If the histograms look similar, that's
# evidence the distortion isn't practically significant.
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def load_sample(log_file):
    sample_path = os.path.join(os.path.dirname(log_file), 'activation_sample.npy')
    if not os.path.exists(sample_path):
        return None
    return np.load(sample_path)


def _zero_frac(sample, eps=1e-6):
    return float(np.mean(np.abs(sample) < eps))


def _js_distance(a, b, bins=60):
    """Jensen-Shannon distance (base 2, so it's bounded in [0, 1]) between
    the empirical distributions of two activation samples, via shared
    histogram binning. 0 = identical distributions, 1 = disjoint support."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 0.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi))
    hist_b, _ = np.histogram(b, bins=edges)
    p = hist_a.astype(float) + 1e-12
    q = hist_b.astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2))


def distortion_metrics(standard_combo_sample, selective_sample):
    """
    Scalar distortion metrics between standard_combo's and selective's
    post-normalization activation distributions, so "does standard_combo
    distort the distribution relative to selective" is a number that can be
    plotted against dropout_rate and correlated with the accuracy gap,
    rather than only judged by eye from overlaid histograms.
    """
    return {
        'wasserstein_distance': float(wasserstein_distance(standard_combo_sample, selective_sample)),
        'js_distance': _js_distance(standard_combo_sample, selective_sample),
        'zero_frac_standard_combo': _zero_frac(standard_combo_sample),
        'zero_frac_selective': _zero_frac(selective_sample),
    }


def plot_group(runs, model, dataset, dropout_rate, normalization, plot_dir):
    plt.figure()
    plotted = False
    for method, entry in runs.items():
        sample = load_sample(entry['log_file'])
        if sample is None:
            continue
        plt.hist(sample, bins=60, density=True, histtype='step', linewidth=1.5, label=method)
        plotted = True

    if not plotted:
        plt.close()
        return False

    dr_suffix = f"_dr{dropout_rate}" if dropout_rate is not None else ""
    norm_suffix = f"_norm-{normalization}" if normalization is not None else ""
    title_bits = []
    if dropout_rate is not None:
        title_bits.append(f"dropout_rate={dropout_rate}")
    if normalization is not None:
        title_bits.append(f"normalization={normalization}")
    title_suffix = f" ({', '.join(title_bits)})" if title_bits else ""
    plt.title(f'Study-Site Activation Distribution — {model}/{dataset}{title_suffix}')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'activation_hist_{model}_{dataset}{dr_suffix}{norm_suffix}.png'))
    plt.close()
    return True


def plot_distortion_vs_dropout(distortion_df, model, dataset, normalization, plot_dir):
    """Companion to compare_results.py's dropout_gap plot: does the
    selective-vs-standard_combo activation-distribution distortion itself
    (not just the downstream accuracy gap) grow with dropout rate?"""
    subset = distortion_df[
        (distortion_df['model'] == model) & (distortion_df['dataset'] == dataset) &
        (distortion_df['normalization'] == normalization)
    ].sort_values('dropout_rate')
    if len(subset) < 2:
        return

    fig, ax1 = plt.subplots()
    ax1.plot(subset['dropout_rate'], subset['wasserstein_distance'], marker='o', color='tab:blue',
              label='Wasserstein distance')
    ax1.set_xlabel('Dropout Rate')
    ax1.set_ylabel('Wasserstein distance', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(subset['dropout_rate'], subset['js_distance'], marker='s', color='tab:red', label='JS distance')
    ax2.set_ylabel('JS distance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Standard-Combo vs Selective Activation Distortion vs Dropout Rate\n{model}/{dataset} (norm={normalization})')
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'distortion_vs_dropout_{model}_{dataset}_norm-{normalization}.png'))
    plt.close()


def _index_runs(ok_runs):
    """
    Index runs by the key each method actually varies over:
      baseline           -> single reference run per (model, dataset)
      norm               -> keyed by normalization
      dropout, selective -> keyed by dropout_rate
      standard_combo     -> keyed by (dropout_rate, normalization)
    Grouping everything by only (model, dataset, dropout_rate) - as if
    normalization didn't exist - would silently keep just the first
    normalization type seen whenever several are swept for norm/
    standard_combo. Returns the four lookup dicts plus the sets of
    dropout_rate / normalization values actually present.
    """
    baseline, by_norm, by_dr, by_dr_norm = {}, {}, {}, {}
    dropout_rates, normalizations = set(), set()

    for r in ok_runs:
        key_md = (r['model'], r['dataset'])
        if r['method'] == 'baseline':
            baseline.setdefault(key_md, r)
        elif r['method'] == 'norm':
            by_norm.setdefault((*key_md, r['normalization']), r)
            normalizations.add(r['normalization'])
        elif r['method'] in ('dropout', 'selective'):
            by_dr.setdefault((*key_md, r['method'], r['dropout_rate']), r)
            dropout_rates.add(r['dropout_rate'])
        elif r['method'] == 'standard_combo':
            by_dr_norm.setdefault((*key_md, r['dropout_rate'], r['normalization']), r)
            dropout_rates.add(r['dropout_rate'])
            normalizations.add(r['normalization'])

    return baseline, by_norm, by_dr, by_dr_norm, dropout_rates, normalizations


def main():
    parser = argparse.ArgumentParser(description='Overlay activation-distribution histograms across methods, and '
                                                   'quantify selective-vs-standard_combo distortion.')
    parser.add_argument('--results_dir', type=str, default='results', help='Root directory produced by run_experiments.py.')
    args = parser.parse_args()

    manifest_path = os.path.join(args.results_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    ok_runs = [r for r in manifest['runs'] if r['status'] == 'ok']
    baseline, by_norm, by_dr, by_dr_norm, dropout_rates, normalizations = _index_runs(ok_runs)

    plot_dir = os.path.join(args.results_dir, 'activation_histograms')
    comparison_dir = os.path.join(args.results_dir, 'comparison')
    written = 0
    distortion_rows = []

    model_datasets = sorted({(r['model'], r['dataset']) for r in ok_runs})
    for model, dataset in model_datasets:
        for dr in sorted(d for d in dropout_rates if d is not None):
            for norm in sorted(n for n in normalizations if n is not None):
                runs = {}
                if (model, dataset) in baseline:
                    runs['baseline'] = baseline[(model, dataset)]
                if (model, dataset, norm) in by_norm:
                    runs['norm'] = by_norm[(model, dataset, norm)]
                if (model, dataset, 'dropout', dr) in by_dr:
                    runs['dropout'] = by_dr[(model, dataset, 'dropout', dr)]
                if (model, dataset, 'selective', dr) in by_dr:
                    runs['selective'] = by_dr[(model, dataset, 'selective', dr)]
                if (model, dataset, dr, norm) in by_dr_norm:
                    runs['standard_combo'] = by_dr_norm[(model, dataset, dr, norm)]

                if plot_group(runs, model, dataset, dr, norm, plot_dir):
                    written += 1

                if 'selective' in runs and 'standard_combo' in runs:
                    sel_sample = load_sample(runs['selective']['log_file'])
                    std_sample = load_sample(runs['standard_combo']['log_file'])
                    if sel_sample is not None and std_sample is not None:
                        metrics = distortion_metrics(std_sample, sel_sample)
                        distortion_rows.append({
                            'model': model, 'dataset': dataset, 'dropout_rate': dr, 'normalization': norm,
                            **metrics,
                        })

    if written == 0:
        print("No activation_sample.npy files found (or no dropout_rate-varying runs). "
              "Run run_experiments.py first with a method that uses dropout_rate.")
    else:
        print(f"Wrote {written} activation histogram(s) to: {plot_dir}")

    if not distortion_rows:
        print("No selective/standard_combo pairs available — skipping distortion analysis.")
        return

    distortion_df = pd.DataFrame(distortion_rows)
    distortion_csv = os.path.join(args.results_dir, 'activation_distortion.csv')
    distortion_df.to_csv(distortion_csv, index=False)
    print(f"\nWrote selective-vs-standard_combo distortion metrics: {distortion_csv}\n")
    print(distortion_df.to_string(index=False))

    for (model, dataset, normalization), _ in distortion_df.groupby(['model', 'dataset', 'normalization']):
        plot_distortion_vs_dropout(distortion_df, model, dataset, normalization, comparison_dir)

    # If compare_results.py has already been run, correlate the mechanistic
    # distortion with the downstream accuracy gap - the direct test of
    # whether the claimed mechanism actually predicts the accuracy effect,
    # rather than the two just happening to move independently.
    summary_path = os.path.join(args.results_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        print("\n(no summary.csv found — run compare_results.py too to correlate distortion with the accuracy gap)")
        return

    summary_df = pd.read_csv(summary_path)
    sel_acc = summary_df[summary_df['method'] == 'selective'].groupby(
        ['model', 'dataset', 'dropout_rate'])['best_test_acc'].mean()
    std_acc = summary_df[summary_df['method'] == 'standard_combo'].groupby(
        ['model', 'dataset', 'dropout_rate', 'normalization'])['best_test_acc'].mean()

    corr_rows = []
    for _, row in distortion_df.iterrows():
        sel_key = (row['model'], row['dataset'], row['dropout_rate'])
        std_key = (row['model'], row['dataset'], row['dropout_rate'], row['normalization'])
        if sel_key in sel_acc.index and std_key in std_acc.index:
            corr_rows.append({**row.to_dict(), 'accuracy_gap': sel_acc.loc[sel_key] - std_acc.loc[std_key]})

    if len(corr_rows) < 2:
        print("\nNot enough matched (distortion, accuracy-gap) points to correlate.")
        return

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = os.path.join(args.results_dir, 'distortion_vs_accuracy_gap.csv')
    corr_df.to_csv(corr_csv, index=False)
    r_wasserstein = corr_df['wasserstein_distance'].corr(corr_df['accuracy_gap'])
    r_js = corr_df['js_distance'].corr(corr_df['accuracy_gap'])
    print(f"\nWrote distortion-vs-accuracy-gap table: {corr_csv}")
    print(f"Correlation(wasserstein_distance, accuracy_gap) = {r_wasserstein:.3f}")
    print(f"Correlation(js_distance, accuracy_gap) = {r_js:.3f}")


if __name__ == '__main__':
    main()
