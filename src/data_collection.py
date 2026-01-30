import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Absolute path of THIS file
CURRENT_FILE = os.path.abspath(__file__)

# src/ directory
SRC_DIR = os.path.dirname(CURRENT_FILE)

# project root: mlops_3/
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# data paths
data_path = os.path.join(PROJECT_ROOT, "data", "water_potability.csv")
output_dir = os.path.join(PROJECT_ROOT, "data", "raw")

print("Reading data from:", data_path)  # debug (you can remove later)

# Read data
data = pd.read_csv(data_path)

# Split
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

# Save outputs
os.makedirs(output_dir, exist_ok=True)
train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
