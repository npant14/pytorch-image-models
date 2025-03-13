import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os

def msg_wrapper(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines

def csv_wrapper(file_path):
    df = pd.read_csv(file_path)
    accs = df['eval_top1'].tolist()
    losses = df['train_loss'].tolist()
    return accs, losses
    
def draw_result_curve(BASE, sizes, model='alexnet', acc=True):
    dic = {}
    for size in sizes:
        # path = os.path.join(BASE, f"alexnet_size_{size}_scale_0.08/summary.csv")
        path = os.path.join(BASE, f"hmax_ip_{size}/summary.csv")
        accs, losses = csv_wrapper(path)
        if acc:
            dic[size] = accs
        else:
            dic[size] = losses
        
    max_len = max([len(x) for x in dic.values()])

    plt.figure(figsize=(10, 6))
    for size in sizes:
        temp = dic[size]
        temp += [None] * (max_len - len(temp))
        plt.plot(range(1, max_len+1), temp, label=str(size))
        
    if acc:
        plt.xlabel('Epoch')
        plt.ylabel('Top1 Acc')
        plt.title(f'Acc of {model} w/ Different IP Bands')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(f'{model}_acc.png')
    else:
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f'Loss of {model} w/ Different IP Bands')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(f'{model}_loss.png')
    plt.close()

BASE = "/oscar/data/tserre/xyu110/pytorch-output/train/"

sizes = [160, 192, 227, 321, 382, 454]
ip_bands = [1, 2]
draw_result_curve(BASE, ip_bands, model='chmax_bypass', acc=True)
draw_result_curve(BASE, ip_bands, model='chmax_bypass', acc=False)

