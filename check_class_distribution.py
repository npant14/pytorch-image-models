import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_path = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
print("Reading CSV file...")
df = pd.read_csv(csv_path)

# Print basic information
print("\nTotal number of images:", len(df))
print("\nNumber of unique classes:", df['Class'].nunique())

# Get class distribution
class_dist = df['Class'].value_counts()
print("\nClass distribution (top 10):")
print(class_dist.head(10))

# Get scale band distribution
scale_dist = df['scale_band'].value_counts()
print("\nScale band distribution:")
print(scale_dist)

# Save distribution to a text file
with open('class_distribution.txt', 'w') as f:
    f.write("Total number of images: {}\n".format(len(df)))
    f.write("Number of unique classes: {}\n\n".format(df['Class'].nunique()))
    f.write("Class distribution:\n")
    f.write(class_dist.to_string())
    f.write("\n\nScale band distribution:\n")
    f.write(scale_dist.to_string()) 