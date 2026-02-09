
import os
import re
import glob
import json
import pandas as pd
import csv

def create_folders(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def save_dataframe(df, filename, folder_path):
    # Check if the folder exists, if not, create it
    create_folders(folder_path)
    
    # Define the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Save the DataFrame to the specified file path
    df.to_csv(file_path, index=False)  # Assuming you want to save it as a CSV and not include the index

def save_dict_to_csv(data, headers, filename='data.csv'):
    # Check if the file exists and whether we need headers
    file_exists = os.path.isfile(filename)

    with open(filename, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writerow(headers)

        writer.writerow(data)
