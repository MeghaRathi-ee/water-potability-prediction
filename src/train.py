import pandas as pd
import numpy as np
import os

import pickle

from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Resolve project root safely
# -----------------------------
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# -----------------------------
# Paths
# -----------------------------
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# -----------------------------
# Load data
# -----------------------------
train_data = pd.read_csv(TRAIN_DATA_PATH)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved at: {MODEL_PATH}")