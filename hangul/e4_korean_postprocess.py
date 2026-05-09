"""
Postprocess saved Hangul correlation matrices into mean/std/sem error bars.

This does not run the model again. It only reads existing raw files such as:
    results/hangul_results_dprime/<model>/<layer>/13-13.csv
    results/hangul_results_dprime/<model>/<layer>/13-52.csv
    results/hangul_results_dprime/<model>/<layer>/13-130.csv

For each condition, it recreates the 1000 random thresholding splits used in
hangul/e1_korean_imagenet.py and reports:
    - mean accuracy
    - std accuracy
    - sem accuracy

The five reported conditions are:
    13-13, 13-52, 52-13, 13-130, 130-13
"""

import argparse
import json
import math
import os
import random

import numpy as np


PAIR_LIST = [(13, 13), (13, 52), (13, 130)]


def pair_key(size_1, size_2):
    return f"{size_1}-{size_2}"


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sem = float(std / math.sqrt(len(values))) if len(values) > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
    }


def extract_scores(correlations):
    """
    Match the original Hangul scoring convention:
    - correct scores come from [i, i]
    - distractor scores come from [i, i+1]
    for even i in 0..52
    """
    all_correct = []
    all_distractor = []
    for idx in range(0, 53, 2):
        all_correct.append(float(correlations[idx][idx]))
        all_distractor.append(float(correlations[idx][idx + 1]))
    return all_correct, all_distractor


def compute_dprime(correct_correlations, distractor_correlations, threshold):
    """
    This matches the repository's existing custom d-prime function.
    """
    target_distractor_pairs = [1 if value > threshold else 0 for value in correct_correlations]
    target_target_pairs = [1 if value <= threshold else 0 for value in distractor_correlations]

    target_distractor_correct = np.mean(target_distractor_pairs)
    target_target_incorrect = 1.0 - np.mean(target_target_pairs)
    return float(target_distractor_correct - target_target_incorrect)


def resample_condition(correlations, num_samples=1000, split_seed=1):
    """
    Recreate the 1000 random splits from e1_korean_imagenet.py using only the
    saved correlation matrix.
    """
    all_correct, all_distractor = extract_scores(correlations)
    rng = random.Random(split_seed)

    accuracy_samples = []
    dprime_samples = []

    for _ in range(num_samples):
        randidxs = sorted(rng.sample(range(54), k=41))

        thresh_correct = []
        test_correct = []
        thresh_distractor = []
        test_distractor = []

        for idx, value in enumerate(all_correct):
            if idx in randidxs:
                thresh_correct.append(value)
            else:
                test_correct.append(value)

        for idx, value in enumerate(all_distractor):
            if idx + 27 in randidxs:
                thresh_distractor.append(value)
            else:
                test_distractor.append(value)

        correct = thresh_correct
        distractor = thresh_distractor

        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in correct + distractor:
            correctly_above_threshold = sum(value > threshold for value in correct)
            incorrectly_above_threshold = sum(value > threshold for value in distractor)
            correctly_below_threshold = 27 - incorrectly_above_threshold
            accuracy = (correctly_above_threshold + correctly_below_threshold) / 54
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        test_correctly_above_threshold = sum(
            value > best_threshold for value in correct + test_correct
        )
        test_incorrectly_above_threshold = sum(
            value > best_threshold for value in distractor + test_distractor
        )
        test_correctly_below_threshold = 27 - test_incorrectly_above_threshold
        test_accuracy = (
            test_correctly_above_threshold + test_correctly_below_threshold
        ) / 54

        accuracy_samples.append(float(test_accuracy))
        dprime_samples.append(
            compute_dprime(all_correct, all_distractor, best_threshold)
        )

    return {
        "accuracy": summarize(accuracy_samples),
    }


def layer_dir_from_args(base, model, layer):
    model_dir = os.path.join(base, model)
    if layer == "":
        return model_dir
    return os.path.join(model_dir, layer)


def compute_layer_stats(layer_dir, num_samples=1000, split_seed=1):
    """
    Compute stats for the five bar-graph conditions from one saved layer.
    """
    results = {}

    for size_1, size_2 in PAIR_LIST:
        path = os.path.join(layer_dir, f"{size_1}-{size_2}.csv")
        correlations = np.loadtxt(path, delimiter=",")

        forward = resample_condition(
            correlations,
            num_samples=num_samples,
            split_seed=split_seed,
        )
        results[pair_key(size_1, size_2)] = forward

        if size_1 != size_2:
            reverse = resample_condition(
                correlations.transpose(),
                num_samples=num_samples,
                split_seed=split_seed,
            )
            results[pair_key(size_2, size_1)] = reverse

    return results


def main():
    parser = argparse.ArgumentParser(description="Postprocess Hangul raw correlation matrices into 1000-split error bars.")
    parser.add_argument("--base", default="results/hangul_results_dprime", help="Base Hangul results directory.")
    parser.add_argument("--model", required=True, help="Model directory name under the base directory.")
    parser.add_argument("--layer", default="", help="Layer subdirectory. Use empty string for the root layer.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random splits.")
    parser.add_argument("--split-seed", type=int, default=1, help="Seed for reproducible resampling.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    layer_dir = layer_dir_from_args(args.base, args.model, args.layer)
    stats = compute_layer_stats(
        layer_dir,
        num_samples=args.num_samples,
        split_seed=args.split_seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
