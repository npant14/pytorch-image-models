import os
import pandas as pd
import plotly.graph_objects as go
import argparse

def plot_learning_curves(base_folder, pattern, output_file="learning_curves.html"):
    # Prepare the figure
    fig = go.Figure()

    # Scan through each folder within the base_folder that matches the pattern
    # scan recursively through the base_folder
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path) and pattern in folder:
            csv_path = os.path.join(folder_path, 'summary.csv')
            if os.path.exists(csv_path):
                # Read the CSV file without column names
                df = pd.read_csv(csv_path)
                print(csv_path)
                print(df.head())
                #df.columns = ['epoch', 'train_loss', 'val_loss', 'top1', 'top5', 'lr']
                try:
                    df['eval_top1'] = df['eval_top1'].astype(float)   # Convert top1 to float
                except:
                    print(f"Error converting eval_top1 to float in {folder}")
                    continue
                # Add traces for train_loss, val_loss, and top1 accuracy
                #fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], mode='lines', name=f'{folder} - Train Loss'))
                #fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], mode='lines', name=f'{folder} - Validation Loss'))
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['eval_top1'], mode='lines', name=f'{folder} - Top1 Accuracy'))

    # Update layout for interactivity
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Epoch',
        yaxis_title=' Accuracy',
        legend_title='Folders',
        hovermode='x unified'
    )

    # Save the plot as an HTML file
    fig.write_html(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from folders with summary.csv files.")
    parser.add_argument("base_folder", type=str, help="Base folder containing subfolders")
    parser.add_argument("pattern", type=str, help="Pattern to match subfolders (e.g., 'folder*')")
    parser.add_argument("--output", type=str, default="learning_curves.html", help="Output HTML file name")
    args = parser.parse_args()
    
    plot_learning_curves(args.base_folder, args.pattern, args.output)
