import pandas as pd
import numpy as np
from pathlib import Path

# project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# data paths
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"

# load data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

def fill_missing_with_median(df):
    """
    Fill missing values in numeric columns with their median.
    """
    df = df.copy()  # avoid modifying original dataframe

    for column in df.select_dtypes(include="number").columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

data_path = Path("data") / "processed"
data_path.mkdir(parents=True, exist_ok=True)

train_processed_data.to_csv(data_path / "train_processed.csv", index=False)
test_processed_data.to_csv(data_path / "test_processed.csv", index=False)