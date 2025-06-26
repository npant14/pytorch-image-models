import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


matrix_size = 54

def vis_korean(csv_filename):
    matrix_df = pd.read_csv(csv_filename, header=None)

    # Verify the shape of the loaded matrix
    if matrix_df.shape != (matrix_size, matrix_size):
        print(f"Warning: The loaded matrix has shape {matrix_df.shape}, expected ({matrix_size}, {matrix_size}).")

    plt.figure(figsize=(20, 18)) # Increased size for better visibility of 52x52

    sns.heatmap(
        matrix_df,
        cmap="viridis",  # Choose a colormap. 'viridis' is good, 'YlGnBu' or 'coolwarm' are other options.
        annot=True,      # Display the value in each cell
        fmt=".2f",       # Format the annotations to two decimal places
        linewidths=.5,   # Add lines between cells
        linecolor='black', # Color of the lines
        cbar=True,       # Show the color bar
        square=True,     # Ensure cells are square
        xticklabels=False, # Hide x-axis labels for a cleaner look
        yticklabels=False  # Hide y-axis labels
    )

    # Add a title to the heatmap
    plt.title(f'Visualize for {csv_filename}', fontsize=20)

    # Ensure the plot layout is tight to prevent labels/titles from being cut off
    plt.tight_layout()

    # Save the heatmap to a file
    output_image_filename = f"{csv_filename[:-4]}.png"
    plt.savefig(output_image_filename, dpi=300) # Save with high resolution
    
    
# korean_results = "/users/xyu110/pytorch-image-models/korean/chresmax_v3_bypass_only/model_backbone.c2b_seq.4"
korean_results = "/users/xyu110/pytorch-image-models/korean/hmax_old_ckpt95/model_pre.c2b"
    
for filename in os.listdir(korean_results):
    if filename.endswith('.csv'):
        vis_korean(os.path.join(korean_results, filename))
