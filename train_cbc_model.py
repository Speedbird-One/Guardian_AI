import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "data/diagnosed_cbc_data_v4.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found.")

df = pd.read_csv(file_path)

# Features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Stage 1: Healthy vs Anemia (binary)
y_stage1 = y.apply(lambda label: "Healthy" if label.lower() == "healthy" else "Anemia")

# Train/Test Split
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y_stage1, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Stage 1: Healthy vs Anemia Classifier
# -----------------------------
print("\n--- Stage 1: Healthy vs Anemia ---")
sm1 = SMOTE(random_state=42)
X1_res, y1_res = sm1.fit_resample(X_train, y1_train)

clf_stage1 = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
clf_stage1.fit(X1_res, y1_res)

y1_pred = clf_stage1.predict(X_test)
print("Classification Report (Stage 1):")
print(classification_report(y1_test, y1_pred))
cm1 = confusion_matrix(y1_test, y1_pred, labels=["Healthy", "Anemia"])
print("Confusion Matrix:\n", cm1)

# -----------------------------
# Stage 2: Multiclass Anemia Classifier
# -----------------------------
print("\n--- Stage 2: Anemia Subtype ---")
# Keep only anemia cases
X_train_anemia = X_train[y2_train != "Healthy"]
y2_train_anemia = y2_train[y2_train != "Healthy"]

X_test_anemia = X_test[y2_test != "Healthy"]
y2_test_anemia = y2_test[y2_test != "Healthy"]

sm2 = SMOTE(random_state=42)
X2_res, y2_res = sm2.fit_resample(X_train_anemia, y2_train_anemia)

clf_stage2 = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)
clf_stage2.fit(X2_res, y2_res)

if len(X_test_anemia) > 0:
    y2_pred = clf_stage2.predict(X_test_anemia)
    print("Classification Report (Stage 2):")
    print(classification_report(y2_test_anemia, y2_pred))
    print("Confusion Matrix:\n", confusion_matrix(y2_test_anemia, y2_pred))

# -----------------------------
# Save Models + Medians
# -----------------------------
median_values = X_train.median()

joblib.dump(clf_stage1, "models/cbc_model_stage1.pkl")
joblib.dump(clf_stage2, "models/cbc_model_stage2.pkl")
joblib.dump(median_values, "models/cbc_medians.pkl")

print("\n✅ Models and medians saved successfully!")
