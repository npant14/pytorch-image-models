import json
import os
import re

import pandas as pd


ORDER = ["13-13", "13-52", "52-13", "13-130", "130-13"]

ALL_LAYERS_FILES = {
    "vit_base": "results/hangul_results_dprime/vit_base_all_layers.csv",
    "resnet18": "results/hangul_results_dprime/resnet18_timm_all_layers.csv",
    "alexnet": "results/hangul_results_dprime/alexnet_timm_all_layers.csv",
    "hmax_v3_adj": "results/hangul_results_dprime/hmax_v3_adj_all_layers.csv",
}

STD_JSON_FILES = {
    "vit_base": "results/hangul_postprocess/vit_base_blocks.7.norm2.json",
    "resnet18": "results/hangul_postprocess/resnet18_timm_layer3.1.conv1.json",
    "alexnet": "results/hangul_postprocess/alexnet_timm_features.5.json",
    "hmax_v3_adj": "results/hangul_postprocess/hmax_v3_adj_model_backbone.s2.json",
}

# Edit this if you want the helper to include human values in hangul_accs.
HUMAN_ACCS = [0.875, 0.85, 0.91, 0.9, 0.83]


def parse_npfloat64_dict(cell):
    """
    Parse a string like:
      "{('13', '13'): np.float64(0.99), ('13', '52'): np.float64(0.62)}"
    into:
      {"13-13": 0.99, "13-52": 0.62}
    """
    s = str(cell).strip().strip('"').strip("'").rstrip(",")
    pairs = re.findall(
        r"\('([^']+)'\s*,\s*'([^']+)'\):\s*(?:np\.float64\()?"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)?",
        s,
    )
    return {f"{a}-{b}": float(value) for a, b, value in pairs}


def avg_npfloat64_dict(cell):
    values = parse_npfloat64_dict(cell)
    return sum(values.values()) / len(values)


def ordered_values(values_dict):
    return [values_dict[key] for key in ORDER]


def best_row_for_model(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df["accuracies_avg"] = df["accuracies"].apply(avg_npfloat64_dict)
    highest_acc_row = df.loc[df["accuracies_avg"].idxmax()]
    return highest_acc_row


def build_hangul_accs():
    hangul_accs = {}
    for model, csv_path in ALL_LAYERS_FILES.items():
        best_row = best_row_for_model(csv_path)
        hangul_accs[model] = ordered_values(parse_npfloat64_dict(best_row["accuracies"]))
    if HUMAN_ACCS is not None:
        hangul_accs["human"] = HUMAN_ACCS
    return hangul_accs


def build_hangul_stds():
    hangul_stds = {}
    for model, json_path in STD_JSON_FILES.items():
        with open(json_path, "r") as f:
            data = json.load(f)
        hangul_stds[model] = [data[key]["accuracy"]["std"] for key in ORDER]
    return hangul_stds


def build_hangul_dprimes():
    hangul_dprimes = {}
    for model, csv_path in ALL_LAYERS_FILES.items():
        best_row = best_row_for_model(csv_path)
        hangul_dprimes[model] = ordered_values(parse_npfloat64_dict(best_row["d_primes"]))
    return hangul_dprimes


def print_dict(name, values):
    print(f"{name} = {{")
    for model, row in values.items():
        print(f'  "{model}": [')
        for value in row:
            print(f"    {value},")
        print("  ],")
    print("}")


if __name__ == "__main__":
    if not all(os.path.exists(path) for path in ALL_LAYERS_FILES.values()):
        raise FileNotFoundError("Missing one or more *_all_layers.csv files.")
    if not all(os.path.exists(path) for path in STD_JSON_FILES.values()):
        raise FileNotFoundError("Missing one or more postprocessed std JSON files.")

    print_dict("hangul_accs", build_hangul_accs())
    print()
    print_dict("hangul_stds", build_hangul_stds())
    print()
    print_dict("hangul_dprimes", build_hangul_dprimes())
