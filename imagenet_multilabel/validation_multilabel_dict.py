#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict


def build_multilabel_dict(csv_path: str):
    rows_by_model = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "scale", "multi_label_acc"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            model = row["model"]
            scale = float(row["scale"])
            multi_label_acc = float(row["multi_label_acc"])
            rows_by_model[model].append((scale, multi_label_acc))

    out = {}
    for model, pairs in rows_by_model.items():
        pairs.sort(key=lambda x: x[0])  # sort by scale
        out[model] = [multi_label_acc for _, multi_label_acc in pairs]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build {model: [multi_label_acc,...]} from hmax_multi_label_results.csv sorted by scale."
    )
    parser.add_argument(
        "--csv",
        default="results/hmax_multi_label_results.csv",
        help="Path to multi-label results CSV (default: results/hmax_multi_label_results.csv)",
    )
    args = parser.parse_args()

    result = build_multilabel_dict(args.csv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
