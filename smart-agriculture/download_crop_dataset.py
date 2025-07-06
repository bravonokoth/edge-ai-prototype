import kagglehub
path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset", path="./dataset/crop_yield")
print("Path to dataset:", path)