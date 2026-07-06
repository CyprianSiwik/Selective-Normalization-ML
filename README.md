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

A single run tells you nothing about whether selective normalization helps — you need all five methods run under matched conditions. `run_experiments.py` sweeps the method × model × dataset matrix and writes each run to its own directory under `results/` so nothing gets overwritten:

```
python run_experiments.py --datasets mnist uci_adult cifar10 --models cnn mlp rnn \
                           --epochs 20 --seeds 0 1 2 --results_dir results
```

Invalid (model, dataset) pairs (e.g. `cnn`+`uci_adult`) are skipped automatically, and a failed run (e.g. a flaky download) doesn't abort the sweep — it's recorded in `results/manifest.json` and excluded from the comparison. Then aggregate:

```
python compare_results.py --results_dir results
```

This writes `results/summary.csv` (per-run metrics), `results/summary_by_method.csv` (averaged across seeds, ranked by best test accuracy per model/dataset), and overlay plots per model/dataset combo in `results/comparison/` showing all five methods' test-accuracy curves on the same axes — the figure that actually answers the README's question.
