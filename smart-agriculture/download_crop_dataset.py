import kagglehub
import os
import shutil

# Download dataset
cache_path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")

# Move to project dataset folder
target_path = "./dataset/crop_yield"
if os.path.exists(target_path):
    shutil.rmtree(target_path)
shutil.copytree(os.path.join(cache_path, "dataset/DATASET"), target_path)

print("Path to dataset files:", target_path)

# Verify dataset structure
for root, dirs, files in os.walk(target_path):
    print(f"Directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Files: {files[:5]}")