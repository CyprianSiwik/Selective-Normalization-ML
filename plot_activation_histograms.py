# plot_activation_histograms.py — Direct mechanistic evidence for or against
# the selective-normalization hypothesis: overlay the actual distribution of
# activations at the normalization study site for each method, at a given
# dropout rate.
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


def load_sample(log_file):
    sample_path = os.path.join(os.path.dirname(log_file), 'activation_sample.npy')
    if not os.path.exists(sample_path):
        return None
    return np.load(sample_path)


def plot_group(runs, model, dataset, dropout_rate, plot_dir):
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
    title_suffix = f" (dropout_rate={dropout_rate})" if dropout_rate is not None else ""
    plt.title(f'Study-Site Activation Distribution — {model}/{dataset}{title_suffix}')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'activation_hist_{model}_{dataset}{dr_suffix}.png'))
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='Overlay activation-distribution histograms across methods.')
    parser.add_argument('--results_dir', type=str, default='results', help='Root directory produced by run_experiments.py.')
    args = parser.parse_args()

    manifest_path = os.path.join(args.results_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    ok_runs = [r for r in manifest['runs'] if r['status'] == 'ok']

    # Group by (model, dataset, dropout_rate); within a group, keep the
    # first seed seen per method (one representative run per method).
    groups = {}
    for r in ok_runs:
        key = (r['model'], r['dataset'], r.get('dropout_rate'))
        groups.setdefault(key, {})
        groups[key].setdefault(r['method'], r)

    # Methods with no dropout_rate (baseline, norm) apply as a reference at
    # every dropout_rate group for the same (model, dataset).
    no_dr_methods = {}
    for (model, dataset, dr), runs in groups.items():
        if dr is None:
            no_dr_methods.setdefault((model, dataset), {}).update(runs)

    plot_dir = os.path.join(args.results_dir, 'activation_histograms')
    written = 0
    for (model, dataset, dr), runs in groups.items():
        if dr is None:
            continue
        combined = dict(no_dr_methods.get((model, dataset), {}))
        combined.update(runs)
        if plot_group(combined, model, dataset, dr, plot_dir):
            written += 1

    if written == 0:
        print("No activation_sample.npy files found (or no dropout_rate-varying runs). "
              "Run run_experiments.py first with a method that uses dropout_rate.")
    else:
        print(f"Wrote {written} activation histogram(s) to: {plot_dir}")


if __name__ == '__main__':
    main()
