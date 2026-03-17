#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict


def build_top1_dict(csv_path: str):
    rows_by_model = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "scale_invariance", "top1"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            model = row["model"]
            scale = float(row["scale_invariance"])
            top1 = float(row["top1"])
            rows_by_model[model].append((scale, top1))

    out = {}
    for model, pairs in rows_by_model.items():
        pairs.sort(key=lambda x: x[0])  # sort by scale_invariance
        out[model] = [top1 for _, top1 in pairs]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build {model: [top1,...]} from validation_imagenet.csv sorted by scale_invariance."
    )
    parser.add_argument(
        "--csv",
        default="validation_imagenet.csv",
        help="Path to validation CSV (default: validation_imagenet.csv)",
    )
    args = parser.parse_args()

    result = build_top1_dict(args.csv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
