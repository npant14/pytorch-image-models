import os
import re
import pandas as pd
from korean import Korean

BASE = './korean_results/korean_results_dprime'
MODEL_NAME_LIST = ['alexnet_all_layers.csv',
                   'chresmax_v3_2_all_layers.csv',
                   'resnet18_all_layers.csv']

def avg_npfloat64_dict(cell: str,
                       return_list=False) -> float:
    """
    Parse a string like:
      "{('13','13'): np.float64(0.99), ('13','52'): np.float64(0.62)}"
    and return the average of the numeric values.
    """
    # Clean common CSV artifacts
    s = cell.strip().strip('"').strip("'").rstrip(',')

    # Fast path: extract numbers inside np.float64(...)
    nums = [float(x) for x in re.findall(
        r"np\.float64\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", s
    )]
    if nums:
        if return_list:
            return nums
        return sum(nums) / len(nums)


for model_name in MODEL_NAME_LIST:
    df = pd.read_csv(os.path.join(BASE, model_name), index_col=0)
    
    df['accuracies_avg'] = df['accuracies'].apply(avg_npfloat64_dict)
    df['d_primes_avg'] = df['d_primes'].apply(avg_npfloat64_dict)

    highest_acc_row = df.loc[df['accuracies_avg'].idxmax()]
    print(f"Highest accuracy row for {model_name}:")
    print(avg_npfloat64_dict(highest_acc_row['accuracies'], return_list=True))
    print(avg_npfloat64_dict(highest_acc_row['d_primes'], return_list=True))
    print("\n")
    
    