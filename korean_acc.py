import os
import pandas as pd
from korean import Korean

def get_all_acc(model='', modelname=None, layer='', base_filepath=None):
    k = Korean(model,
                os.path.join('/oscar/data/tserre/xyu110/pytorch-output/korean', modelname),
                None,
                '/gpfs/data/tserre/npant1/hangul_data',
                224,
                layer)
    
    base_filepath = os.path.join(base_filepath, modelname)
    # get all folder names in the base filepath
    folder_names = [f for f in os.listdir(base_filepath) if os.path.isdir(os.path.join(base_filepath, f))]
    print(folder_names)
    
    results_data = []
    for layer_name in folder_names:
        filepath = os.path.join(base_filepath, layer_name)
        # go through csv files in the folder
        csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
        csv_files = [os.path.join(filepath, csv_file) for csv_file in csv_files]
        print(f'Processing {filepath}')
        accs = k.get_accuracy(csv_files)
        print(f'Accuracy for {filepath}: {accs}')
        
        formatted_accs = {f"{k[0]}_vs_{k[1]}": v for k, v in accs.items()}
        formatted_accs['layer'] = layer_name
        results_data.append(formatted_accs)

    df = pd.DataFrame(results_data)
    
    # Set the 'layer' column as the index
    if 'layer' in df.columns:
        df.set_index('layer', inplace=True)
        
    df.to_csv(os.path.join(base_filepath, 'all_acc.csv'), index=True)
        
    return df
    
BASE = '/oscar/data/tserre/xyu110/pytorch-output/korean'
MODEL_NAME_LIST = ['hmax_old_original', 'hmax_old', 'hmax_new_tricks', 'chresmax_v3_bypass_only',
                   'chresmax_abs_bypass_only']
MODEL_NAME_DEBUG = ['chresmax_v3_bypass_only_c2b']

for model_name in MODEL_NAME_DEBUG:
    get_all_acc(model=None, modelname=model_name, layer='', base_filepath=BASE)
    df = pd.read_csv(os.path.join(BASE, model_name, 'all_acc.csv'), index_col=0)
    
    # getting stats
    max_acc = df.max(axis=0)
    avg_acc = df.mean(axis=0)
    std_acc = df.std(axis=0)
    
    summary_df = pd.DataFrame({
        'Max': max_acc,
        'Average': avg_acc,
        'Std Dev': std_acc
    })
    
    # print(f"\n--- Summary Statistics for {model_name} ---")
    # print(summary_df)
    
    # print("\n--- Formatted as 'Average ± Std Dev' ---")
    # for acc_type in summary_df.index:
    #     avg = summary_df.loc[acc_type, 'Average']
    #     std = summary_df.loc[acc_type, 'Std Dev']

    #     print(f"{acc_type:<10}: {avg:.4f} ± {std:.4f}")
        
    max_tuple = tuple(max_acc)
    std_tuple = tuple(std_acc)
    print(f"{model_name} = {max_tuple}")
    print(f"{model_name}_err = {std_tuple}")
