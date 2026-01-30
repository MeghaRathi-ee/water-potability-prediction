import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# Resolve project root safely (DO NOT CHANGE THIS)
# -------------------------------------------------
CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# -------------------------------------------------
# Paths (processed data → model)
# -------------------------------------------------
TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "train_processed.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# -------------------------------------------------
# Load processed training data
# -------------------------------------------------
train_data = pd.read_csv(TRAIN_DATA_PATH)

X_train = train_data.drop(columns=['Potability'], axis=1)
y_train = train_data['Potability']

# -------------------------------------------------
# Train model
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Save model artifact
# -------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model training complete")
print(f"📦 Model saved at: {MODEL_PATH}")
