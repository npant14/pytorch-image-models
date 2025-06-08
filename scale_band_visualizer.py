import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import random
from pathlib import Path
from flask import Flask, render_template_string, request
import argparse
from class_selector import select_class

# Load WordNet ID to Class Label Mapping
wordnet_to_label_txt = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/EVF-SAM/wordnetids_to_labels.txt"
wordnet_to_label = {}
with open(wordnet_to_label_txt, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            wordnet_to_label[parts[0]] = int(parts[1])-1

def load_data():
    csv_file = "/cifs/data/tserre_lrs/projects/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
    mask_lookup_json = "/cifs/data/tserre_lrs/projects/projects/prj_hmax_masks/HMAX/SAM_Imagenet/sam2/image_to_mask_lookup.json"
    
    df = pd.read_csv(csv_file)
    with open(mask_lookup_json, 'r') as f:
        mask_lookup = json.load(f)
    
    return df, mask_lookup

def get_random_samples(df, class_id=None, samples_per_band=20):
    # Filter by class if specified
    if class_id is not None:
        df = df[df.iloc[:, 0].str.split('_').str[0].map(lambda x: wordnet_to_label.get(x, -1)) == class_id]
        if len(df) == 0:
            print(f"Warning: No samples found for class ID {class_id}")
            return {band: pd.DataFrame() for band in range(1, 6)}

    # Group by scale band and get random samples
    samples = {}
    for band in range(1, 6):  # Scale bands 1-5
        band_df = df[df.iloc[:, 9] == band]
        if len(band_df) == 0:
            print(f"Warning: No samples found for scale band {band}")
            samples[band] = pd.DataFrame()  # Empty DataFrame for this band
            continue
            
        # If we have fewer samples than requested, use all available samples
        n_samples = min(samples_per_band, len(band_df))
        band_samples = band_df.sample(n=n_samples, replace=True)
        samples[band] = band_samples
    
    return samples

def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((200, 200))  # Resize for consistent display
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_html_content(samples, mask_lookup, class_id=None):
    class_name = "All Classes"
    if class_id is not None:
        # Find the class name from the first non-empty sample
        for band in range(1, 6):
            if not samples[band].empty:
                wordnet_id = samples[band].iloc[0, 0].split('_')[0]
                class_name = f"Class {class_id} ({wordnet_id})"
                break

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scale Band Visualizer - {class_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
            .column {{ display: flex; flex-direction: column; gap: 10px; }}
            .image-container {{ 
                border: 1px solid #ccc; 
                padding: 5px; 
                text-align: center;
                position: relative;
            }}
            .image-container img {{ max-width: 200px; height: auto; }}
            .image-info {{ 
                font-size: 12px; 
                margin-top: 5px;
                word-wrap: break-word;
            }}
            .controls {{ margin-bottom: 20px; }}
            button {{ 
                padding: 10px 20px; 
                font-size: 16px; 
                cursor: pointer;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }}
            button:hover {{ background-color: #45a049; }}
            .scale-band-header {{
                background-color: #f0f0f0;
                padding: 10px;
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .stats {{
                margin: 20px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .no-samples {{
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                margin: 10px 0;
            }}
            .class-selector {{
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .class-selector input {{
                padding: 5px;
                margin-right: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Scale Band Visualizer - {class_name}</h1>
        <div class="class-selector">
            <form action="/" method="get">
                <input type="number" name="class_id" min="0" max="999" value="{class_id if class_id is not None else ''}" placeholder="Enter class ID (0-999)">
                <button type="submit">Filter by Class</button>
            </form>
        </div>
        <div class="controls">
            <button onclick="window.location.reload()">Refresh Samples</button>
        </div>
        <div class="stats">
            <h3>Statistics</h3>
            <p>Total samples per band: 20</p>
            <p>Total images displayed: 100</p>
        </div>
        <div class="grid">
    """
    
    # Add column headers
    for band in range(1, 6):
        html_content += f'<div class="scale-band-header">Scale Band {band}</div>'
    
    # Add images for each scale band
    for i in range(20):  # 20 samples per band
        for band in range(1, 6):
            if samples[band].empty:
                html_content += """
                <div class="no-samples">
                    No samples available for this scale band
                </div>
                """
                continue
                
            sample = samples[band].iloc[i % len(samples[band])]  # Use modulo to handle fewer samples
            img_file = sample.iloc[0]
            
            if img_file in mask_lookup:
                img_path = mask_lookup[img_file]["image_path"]
                img_base64 = image_to_base64(img_path)
                
                if img_base64:
                    wordnet_id = img_file.split('_')[0]
                    class_label = wordnet_to_label.get(wordnet_id, "Unknown")
                    center_x = sample.iloc[7]
                    center_y = sample.iloc[6]
                    
                    html_content += f"""
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,{img_base64}" alt="Sample {i+1}">
                        <div class="image-info">
                            Class: {class_label}<br>
                            Center: ({center_x:.1f}, {center_y:.1f})
                        </div>
                    </div>
                    """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content

app = Flask(__name__)
df, mask_lookup = load_data()

@app.route('/')
def index():
    class_id = request.args.get('class_id')
    if class_id is not None:
        class_id = int(class_id)
    samples = get_random_samples(df, class_id)
    html_content = generate_html_content(samples, mask_lookup, class_id)
    return render_template_string(html_content)

def main():
    parser = argparse.ArgumentParser(description='Scale Band Visualizer Server')
    parser.add_argument('--port', type=int, required=True, help='Port to run the server on (required)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == "__main__":
    main() 