import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Source and destination paths
SOURCE_DIR = "/gpfs/data/tserre/npant1/ILSVRC/train"
DEST_DIR = "/gpfs/data/tserre/irodri15/ILSVRC_scale/train"
SCALE_CSV = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_concept_surgery/finetuning_models/fp_checked2.csv"
NUM_WORKERS = 8

def process_scale_band(args):
    try:
        scale_band, group, target_count, df = args
        logger.info(f"Starting processing scale band {scale_band}")
        
        # Create a list to store all operations
        operations = []
        
        current_count = len(group)
        logger.info(f"Scale band {scale_band}: Current count = {current_count}, Target count = {target_count}")
        
        if current_count < target_count:
            # Need to duplicate some images
            num_to_add = target_count - current_count
            logger.info(f"Scale band {scale_band}: Need to add {num_to_add} samples")
            
            # Randomly sample with replacement to get additional images
            additional_samples = group.sample(n=num_to_add, replace=True)
            logger.info(f"Scale band {scale_band}: Generated {len(additional_samples)} additional samples")
            
            # Combine original and additional samples
            all_samples = pd.concat([group, additional_samples])
        else:
            # Randomly sample without replacement to get target number
            all_samples = group.sample(n=target_count, replace=False)
            logger.info(f"Scale band {scale_band}: Selected {len(all_samples)} samples")
        
        # Prepare copy operations
        logger.info(f"Scale band {scale_band}: Preparing copy operations")
        for idx, (_, row) in enumerate(all_samples.iterrows()):
            if idx % 1000 == 0:
                logger.info(f"Scale band {scale_band}: Processed {idx} files")
                
            img_file = row['Image File']
            class_id = row['class']
            
            # Source and destination paths
            src_path = os.path.join(SOURCE_DIR, class_id, img_file)
            dest_path = os.path.join(DEST_DIR, class_id, img_file)
            
            # Create class directory in destination
            dest_class_dir = os.path.join(DEST_DIR, class_id)
            os.makedirs(dest_class_dir, exist_ok=True)
            
            # Add to operations list
            operations.append((src_path, dest_path))
        
        # Execute copy operations
        logger.info(f"Scale band {scale_band}: Starting copy operations for {len(operations)} files")
        copied_count = 0
        skipped_count = 0
        for idx, (src_path, dest_path) in enumerate(operations):
            if idx % 1000 == 0:
                logger.info(f"Scale band {scale_band}: Copied {copied_count} files, Skipped {skipped_count} files")
                
            if os.path.exists(src_path):
                if not os.path.exists(dest_path):  # Only copy if destination doesn't exist
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
                else:
                    skipped_count += 1
            else:
                logger.warning(f"Scale band {scale_band}: Source file not found: {src_path}")
                skipped_count += 1
        
        logger.info(f"Scale band {scale_band}: Completed. Copied {copied_count} files, Skipped {skipped_count} files")
        return copied_count
        
    except Exception as e:
        logger.error(f"Error processing scale band {scale_band}: {str(e)}")
        raise

def create_balanced_dataset():
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(DEST_DIR, exist_ok=True)
        logger.info(f"Created destination directory: {DEST_DIR}")
        
        # Read the scale information
        logger.info("Reading scale information...")
        df = pd.read_csv(SCALE_CSV)
        logger.info(f"Read {len(df)} rows from CSV")
        
        # Get the distribution of scale bands
        scale_counts = df['scale_band'].value_counts()
        logger.info("\nCurrent scale band distribution:")
        logger.info(scale_counts)
        
        # Determine target number of samples (use the maximum count)
        target_count = scale_counts.max()
        logger.info(f"\nTarget samples per scale band: {target_count}")
        
        # Group images by scale band
        scale_groups = df.groupby('scale_band')
        logger.info(f"Created {len(scale_groups)} scale band groups")
        
        # Prepare arguments for parallel processing
        process_args = [(scale_band, group, target_count, df) 
                       for scale_band, group in scale_groups]
        logger.info(f"Prepared {len(process_args)} scale bands for processing")
        
        # Process scale bands in parallel
        logger.info(f"\nProcessing scale bands using {NUM_WORKERS} workers...")
        with Pool(NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap(process_scale_band, process_args), 
                              total=len(process_args),
                              desc="Processing scale bands"))
        
        logger.info(f"Completed processing. Results: {results}")
        
        # Verify the final distribution
        logger.info("\nVerifying final distribution...")
        final_counts = {}
        total_files = 0
        for class_dir in os.listdir(DEST_DIR):
            class_path = os.path.join(DEST_DIR, class_dir)
            if os.path.isdir(class_path):
                files_in_class = os.listdir(class_path)
                total_files += len(files_in_class)
                for img_file in files_in_class:
                    # Find the scale band for this image
                    img_info = df[df['Image File'] == img_file]
                    if not img_info.empty:
                        scale_band = img_info['scale_band'].iloc[0]
                        final_counts[scale_band] = final_counts.get(scale_band, 0) + 1
        
        logger.info(f"\nTotal files in destination: {total_files}")
        logger.info("\nFinal scale band distribution:")
        for scale_band, count in sorted(final_counts.items()):
            logger.info(f"Scale band {scale_band}: {count} images")
            
    except Exception as e:
        logger.error(f"Error in create_balanced_dataset: {str(e)}")
        raise

if __name__ == "__main__":
    create_balanced_dataset() 