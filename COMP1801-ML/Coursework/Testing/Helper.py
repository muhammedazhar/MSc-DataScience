try:
    # Importing general libraries
    import sys
    import glob
    import pandas as pd

    import torch

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("System is using Apple Silicon MPS!\n")
        # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
        print(f"Is Apple MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is Apple MPS available? {torch.backends.mps.is_available()}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found!\n")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU instead :(")
        print("This might take some time.\n")

except Exception as e:
    print(f"Error : {e}")

# Find the CSV file in the Datasets directory
data_path = '../Datasets/*.csv'
file_list = glob.glob(data_path)

for file in file_list:
    print(f"Found file: {file}")

# Ensure there is exactly one file
if len(file_list) == 1:
    # Load the dataset
    df = pd.read_csv(file_list[0])
    print(f"Loaded dataset: {file_list[0]}\n")
else:
    raise FileNotFoundError("No CSV file found or multiple CSV files found in the Datasets directory.")
    sys.exit(1)