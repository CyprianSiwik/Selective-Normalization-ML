# Selective Normalization Research

This research project investigates a novel approach called selective normalization, which modifies the interaction between dropout and normalization layers during neural network training. In conventional pipelines, batch normalization or similar methods operate on all neuron activations, including those that will later be deactivated by dropout. This mismatch can distort the statistical assumptions underlying normalization, potentially introducing instability or inefficiency during training.

Selective normalization addresses this by normalizing only the active (non-dropped) neurons during each forward pass. The goal is to make the network's normalization dynamics more consistent with its actual structure at each moment during training. By adapting the normalization step to exclude dropped neurons, we aim to improve convergence speed, reduce training instability, and enhance generalization.

This repository provides a complete experimental framework to rigorously evaluate selective normalization across a wide range of conditions. The implementation includes five core methods for comparison: a pure baseline without dropout or normalization, dropout only, normalization only, a standard dropout-then-normalization configuration, and the proposed selective normalization method. The experiments are structured to identify not only raw performance differences but also how these methods affect training dynamics, convergence behavior, and inference reliability.

The framework spans multiple architectures — multilayer perceptrons, convolutional networks, and recurrent networks — each with a standard and a lightweight (reduced-parameter) variant. It covers MNIST, CIFAR-10, CIFAR-100, IMDB, and UCI Adult. (Transformers and TinyImageNet are not yet implemented; they're a natural next step but out of scope for the current codebase.) Every run tracks both primary and secondary metrics: test accuracy, training/test loss, per-epoch gradient norm (training stability), activation mean/std at the normalization site under study, inference time, and peak memory usage. `run_experiments.py` and `compare_results.py` provide the organized logging and cross-method comparison needed to actually evaluate the hypothesis, rather than just running one configuration at a time.

Selective normalization is still an open question, but this research provides the tools and design needed to determine whether the idea leads to practical improvements over existing methods. Whether or not the method succeeds, this study aims to clarify the effects of dropout-normalization interaction and contribute a reproducible, well-analyzed body of work to the community.

## Setup

```
pip install -r requirements.txt
```

`torchtext` (used only by the IMDB loader) is an unmaintained package that can be finicky to install; skip `--dataset imdb` / `--model rnn` if it gives you trouble — the other four datasets and both other architectures don't depend on it.

## Running a single experiment

```
python main.py --method <baseline|dropout|norm|standard_combo|selective> \
                --model <cnn|mlp|rnn> --dataset <mnist|cifar10|cifar100|uci_adult|imdb> \
                --epochs 10 --lightweight --seed 0
```

`--dropout_rate` and `--normalization` apply to whichever methods use them (dropout/standard_combo/selective, and norm/standard_combo, respectively). Logs are written as a CSV to `--log_file`, and loss/accuracy/gradient-norm/activation-std plots to `--plot_dir`.

## Running the full comparison (the actual hypothesis test)

A single run, or even five runs at one dropout rate, doesn't tell you much — the comparison that actually matters is `selective` vs. `standard_combo` (both use dropout + normalization; they differ only in whether normalization stats are computed before or after excluding dropped neurons), and specifically whether their gap *grows with dropout rate*, since that's the mechanism the hypothesis claims. `run_experiments.py` sweeps the method × model × dataset × dropout-rate matrix and writes each run to its own directory under `results/` so nothing gets overwritten:

```
python run_experiments.py --datasets mnist uci_adult cifar10 --models cnn mlp rnn \
                           --epochs 20 --seeds 0 1 2 \
                           --dropout_rates 0.1 0.3 0.5 0.7 0.9 \
                           --normalizations batch layer group \
                           --results_dir results
```

Sweeping `--normalizations` matters, not just `--dropout_rates`: batch norm's cross-sample statistics make it the most exposed to the claimed distortion mechanism (dropped zeros pollute a whole batch's stats), while layer/group norm are computed per-sample and may not show the effect at all. Every downstream comparison below treats each normalization type separately rather than averaging them together, so `batch` vs `layer` vs `group` is itself part of the result, not noise to average out.

Invalid (model, dataset) pairs (e.g. `cnn`+`uci_adult`) are skipped automatically, and a failed run (e.g. a flaky download) doesn't abort the sweep — it's recorded in `results/manifest.json` and excluded from the comparison. Then aggregate:

```
python compare_results.py --results_dir results
```

Every run also logs, per epoch: batch-to-batch gradient-norm std (training-stability, not just its mean), train- and eval-mode activation mean/std at the study site (so selective normalization's own train/eval mismatch — its running stats are accumulated only from dropout survivors, but eval mode sees every activation — is visible instead of assumed away), and the train/test accuracy gap (generalization, independent of raw accuracy).

`compare_results.py` writes:
- `results/summary.csv` — per-run metrics, including the above.
- `results/summary_by_method.csv` — averaged across seeds, ranked by best test accuracy per model/dataset/dropout_rate/normalization.
- `results/significance_selective_vs_standard_combo.csv` — a paired t-test (paired by seed) at each dropout rate x normalization, with Cohen's d (paired effect size) alongside the p-value and a Holm-Bonferroni-corrected p-value (`p_value_holm`) across every test in the sweep, since a handful of significant-looking p-values is expected by chance once you run dozens of them. Needs `--seeds` with at least 2 values to produce a p-value.
- `results/cost_benefit_selective_vs_standard_combo.csv` — selective's per-epoch time, per-batch inference time, and peak memory overhead (%) relative to standard_combo, next to the accuracy gap at the same setting — so a gap that only shows up alongside a large compute/memory cost reads differently than one that's free.
- `results/comparison/compare_<model>_<dataset>[_dr<rate>].png` — test-accuracy-vs-epoch curves, one figure per dropout rate.
- `results/comparison/dropout_gap_<model>_<dataset>_norm-<normalization>.png` — **the key plot**: (selective − standard_combo) best-test-accuracy gap vs. dropout rate, with error bars across seeds, one figure per normalization type. A gap that grows with dropout rate is evidence *for* the hypothesis; flat/near-zero across the whole sweep is evidence *against* it. Needs at least two `--dropout_rates` values to render.

For the mechanistic (not just accuracy) picture, overlay the actual activation distributions at the normalization study site:

```
python plot_activation_histograms.py --results_dir results
```

This writes:
- `results/activation_histograms/activation_hist_<model>_<dataset>_dr<rate>_norm-<normalization>.png` — all five methods' post-normalization activation distributions at a given dropout rate and normalization type — a direct look at whether `standard_combo` actually distorts the distribution relative to `selective`, independent of downstream accuracy.
- `results/activation_distortion.csv` and `results/comparison/distortion_vs_dropout_<model>_<dataset>_norm-<normalization>.png` — the same standard_combo-vs-selective distortion quantified as a scalar (Wasserstein distance and Jensen-Shannon distance between the two distributions, plus each one's fraction of near-zero activations) and plotted against dropout rate, so the distortion claim doesn't rest on eyeballing histogram overlays.
- `results/distortion_vs_accuracy_gap.csv` (only if `compare_results.py` has already been run) — merges the distortion metrics with the accuracy gap and reports their correlation, the direct test of whether the claimed mechanism actually predicts the downstream accuracy effect rather than the two just moving independently.
