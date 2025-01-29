import kagglehub

# Download latest version
path = kagglehub.dataset_download("data")

print("Path to dataset files:", path)