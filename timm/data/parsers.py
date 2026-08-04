import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.classes = sorted(self.df['Class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row['Path'])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['Class']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_parser(name, root, split, class_map=None, **kwargs):
    if name == 'csv':
        dataset = CSVDataset(
            root=root,
            csv_file="/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv",
            transform=kwargs.get('transform')
        )
        return dataset
    else:
        raise RuntimeError(f'Unknown parser {name}') 