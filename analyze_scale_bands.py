import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
CSV_FILE = "/cifs/data/tserre_lrs/projects/projects/prj_concept_surgery/finetuning_models/foreground_proportions_five1.csv"

# Read the CSV file
df = pd.read_csv(CSV_FILE)

# Print basic statistics about scale_band
print("\nScale Band Distribution:")
print(df['scale_band'].value_counts().sort_index())
print("\nPercentage Distribution:")
print((df['scale_band'].value_counts(normalize=True) * 100).sort_index())

# Create a histogram
plt.figure(figsize=(10, 6))
df['scale_band'].hist(bins=len(df['scale_band'].unique()))
plt.title('Distribution of Scale Bands')
plt.xlabel('Scale Band')
plt.ylabel('Count')
plt.savefig('scale_band_distribution.png')
plt.close()

# Print additional statistics
print("\nAdditional Statistics:")
print(f"Total number of samples: {len(df)}")
print(f"Number of unique scale bands: {df['scale_band'].nunique()}")
print(f"Mean scale band: {df['scale_band'].mean():.2f}")
print(f"Median scale band: {df['scale_band'].median():.2f}")
print(f"Standard deviation: {df['scale_band'].std():.2f}") 