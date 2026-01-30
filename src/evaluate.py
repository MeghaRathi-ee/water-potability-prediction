import pandas as pd
import os
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------
# Resolve project root safely
# -------------------------------------------------
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# -------------------------------------------------
# Paths
# -------------------------------------------------
TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "test_processed.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "model.pkl"
)

METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")

# -------------------------------------------------
# Load processed test data
# -------------------------------------------------
test_data = pd.read_csv(TEST_DATA_PATH)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# Predict
# -------------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------------
# Compute metrics
# -------------------------------------------------
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# -------------------------------------------------
# Save metrics
# -------------------------------------------------
os.makedirs(METRICS_DIR, exist_ok=True)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation completed")
print(f"📊 Metrics saved at: {METRICS_PATH}")
