import pandas as pd
import numpy as np
import plotly.express as px
import os 
# Load your processed dataframe (model_scores_v4.csv with merged ImageNet accuracy)
df = pd.read_csv("../data/model_scores_with_accuracy.csv")
df['test_scale_tuple'] = df['test_scale'].apply(eval)
df['test_scale_mean'] = df['test_scale_tuple'].apply(lambda x: sum(x) / len(x))

def calculate_flatness_two_slopes(group):
    group = group.sort_values(by='test_scale_mean')
    x = group['test_scale_mean'].values
    y = group['score'].values

    if len(x) < 3:
        return pd.Series({'flatness_score_new': np.nan, 'left_slope': np.nan, 'right_slope': np.nan})

    middle_idx = len(x) // 2
    left_slope = abs((y[middle_idx] - y[0]) / (x[middle_idx] - x[0])) if middle_idx > 0 else np.nan
    right_slope = abs((y[-1] - y[middle_idx]) / (x[-1] - x[middle_idx])) if middle_idx < len(x) - 1 else np.nan
    slope_avg = np.nanmean([left_slope, right_slope])
    flatness_score_new = 1 / (1 + slope_avg)

    return pd.Series({
        'flatness_score_new': flatness_score_new,
        'left_slope': left_slope,
        'right_slope': right_slope
    })

def classify_model(model_name):
    name = model_name.lower()
    if 'inception' in name or 'resnet' in name or 'resnext' in name or 'conv' in name or 'vgg' in name:
        return 'CNN'
    elif 'vit' in name or 'deit' in name or 'swin' in name or 'beit' in name or 'cait' in name or 'eva' in name:
        return 'Transformer'
    elif 'adv' in name:
        return 'Adversarial'
    elif 'mim' in name or 'dino' in name or 'mae' in name or 'self' in name or 'moco' in name:
        return 'Self-Supervised'
    elif 'in21k' in name or 'imagenet21k' in name:
        return 'Pretrained on ImageNet21k'
    else:
        return 'Other'

# make folder for results if it doesn't exist
if not os.path.exists("../results"):
    os.makedirs("../results")

# Group by model and calculate flatness
flatness_df = df.groupby('model').apply(calculate_flatness_two_slopes).reset_index()

# Add ImageNet accuracy and network type
flatness_df = flatness_df.merge(df.groupby('model')['imagenet_top1_accuracy'].first().reset_index(), on='model')
flatness_df['network_type'] = flatness_df['model'].apply(classify_model)

# Exponential decay flatness
flatness_df['flatness_score_exp'] = np.exp(- (flatness_df['left_slope'] + flatness_df['right_slope']) / 2)

# Log-based flatness
epsilon = 1e-6
flatness_df['flatness_score_log'] = -np.log((flatness_df['left_slope'] + flatness_df['right_slope']) / 2 + epsilon)

# Min-max normalized flatness
avg_slope = (flatness_df['left_slope'] + flatness_df['right_slope']) / 2
min_slope, max_slope = avg_slope.min(), avg_slope.max()
flatness_df['flatness_score_norm'] = 1 - ((avg_slope - min_slope) / (max_slope - min_slope))

# Save plots
fig_log = px.scatter(flatness_df, x='imagenet_top1_accuracy', y='flatness_score_log', text='model',
                     size='flatness_score_log', color='network_type',
                     title='Log-Based Flatness Score vs Accuracy',
                     hover_data=['left_slope', 'right_slope'])
fig_log.write_html("flatness_log_vs_accuracy.html")

fig_norm = px.scatter(flatness_df, x='imagenet_top1_accuracy', y='flatness_score_norm', text='model',
                      size='flatness_score_norm', color='network_type',
                      title='Normalized Flatness Score vs Accuracy',
                      hover_data=['left_slope', 'right_slope'])


fig_norm.write_html("../results/flatness_norm_vs_accuracy.html")

# Optional: export results
flatness_df.to_csv("../results/flatness_scores_by_model.csv", index=False)
