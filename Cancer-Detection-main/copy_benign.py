import pandas as pd
import shutil
import os

metadata_path = r"c:\Users\Usama\Downloads\HAM10000\HAM10000_metadata_cleaned.csv"
images_src_dir = r"c:\Users\Usama\Downloads\HAM10000"
dest_dir = r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\benign"

os.makedirs(dest_dir, exist_ok=True)

df = pd.read_csv(metadata_path)
# 'nv' is benign
benign_df = df[df['dx'] == 'nv'].head(600)

count = 0
for _, row in benign_df.iterrows():
    img_id = row['image_id']
    src_path = os.path.join(images_src_dir, f"{img_id}.jpg")
    dest_path = os.path.join(dest_dir, f"{img_id}.jpg")
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        count += 1

print(f"Copied {count} benign images to {dest_dir}")
