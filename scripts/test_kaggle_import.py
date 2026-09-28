import os
import kagglehub
import pandas as pd

import time

print("1. Downloading dataset from kagglehub...")
path = None
for attempt in range(5):
    try:
        print(f"Download attempt {attempt + 1}/5...")
        path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
        time.sleep(5)

if not path:
    print("Failed to download dataset after 5 attempts.")
    exit(1)
print(f"Dataset downloaded to: {path}")

files = os.listdir(path)
print(f"Files found in dataset: {files}")

# Find csv file
csv_files = [f for f in files if f.endswith(".csv")]
if not csv_files:
    # Check subdirectories
    for root, dirs, fnames in os.walk(path):
        for f in fnames:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

print(f"CSV files detected: {csv_files}")

if csv_files:
    target_csv = csv_files[0] if os.path.isabs(csv_files[0]) else os.path.join(path, csv_files[0])
    print(f"\n2. Loading first 5 rows from Kaggle downloaded file: {target_csv}")
    df_kaggle = pd.read_csv(target_csv, nrows=5)
    print("Kaggle dataset shape (first 5 rows preview):", df_kaggle.shape)
    print("Kaggle columns:", list(df_kaggle.columns))
    print("\nKaggle head(2):")
    print(df_kaggle.head(2))

    local_csv = "dataset/borg_traces_data.csv"
    if os.path.exists(local_csv):
        print(f"\n3. Loading first 5 rows from local file: {local_csv}")
        df_local = pd.read_csv(local_csv, nrows=5)
        print("Local columns:", list(df_local.columns))
        print("\nLocal head(2):")
        print(df_local.head(2))

        # Check column match
        kaggle_cols = set(df_kaggle.columns)
        local_cols = set(df_local.columns)
        common_cols = kaggle_cols.intersection(local_cols)
        print(f"\nTotal Kaggle cols: {len(kaggle_cols)}, Total Local cols: {len(local_cols)}")
        print(f"Common columns count: {len(common_cols)}")
        print(f"Columns in Kaggle but not Local: {kaggle_cols - local_cols}")
        print(f"Columns in Local but not Kaggle: {local_cols - kaggle_cols}")
else:
    print("No CSV found in the downloaded dataset.")
