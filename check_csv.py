import pandas as pd

# Read the CSV file
csv_path = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
df = pd.read_csv(csv_path)

# Print column names
print("\nColumn names:")
print(df.columns.tolist())

# Print first few rows
print("\nFirst few rows:")
print(df.head())

# Print basic info
print("\nDataFrame info:")
print(df.info()) 