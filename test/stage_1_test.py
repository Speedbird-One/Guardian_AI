import pandas as pd
import joblib

# -----------------------------
# Load Stage 1 model + medians
# -----------------------------
clf_stage1 = joblib.load("models/cbc_model_stage1.pkl")
medians = joblib.load("models/cbc_medians.pkl")

# Stage 1 threshold
stage1_threshold = 0.8

# -----------------------------
# Synthetic Test Cases for Stage 1
# -----------------------------
test_cases = [
    {
        "label": "Healthy",
        "HGB": 14.2, "HCT": 42.1, "RBC": 4.8, "WBC": 7500, "PLT": 230000,
        "NEUTp": 60, "LYMp": 30, "NEUTn": 4500, "LYMn": 2250,
        "MCV": 88, "MCH": 29, "MCHC": 33, "PDW": 11, "PCT": 0.22
    },
    {
        "label": "Borderline Anemia",
        "HGB": 12.5, "HCT": 37, "RBC": 4.1, "WBC": 6000, "PLT": 220000,
        "NEUTp": 50, "LYMp": 35, "NEUTn": 3000, "LYMn": 1800,
        "MCV": 85, "MCH": 29, "MCHC": 34, "PDW": 12, "PCT": 0.21
    },
    {
        "label": "Normocytic Anemia",
        "HGB": 9.8, "HCT": 29.5, "RBC": 3.6, "WBC": 5200, "PLT": 210000,
        "NEUTp": 55, "LYMp": 35, "NEUTn": 2860, "LYMn": 1820,
        "MCV": 82, "MCH": 28, "MCHC": 34, "PDW": 12, "PCT": 0.20
    }
]

# -----------------------------
# Run Stage 1 Predictions
# -----------------------------
for case in test_cases:
    df = pd.DataFrame([case])
    label = df.pop("label")[0]

    # Align features & fill missing values
    df_test = df.reindex(columns=clf_stage1.feature_names_in_).fillna(medians)

    # Predict Stage 1
    stage1_pred = clf_stage1.predict(df_test)[0]
    stage1_proba = clf_stage1.predict_proba(df_test)[0]
    stage1_confidence = stage1_proba.max()

    # Apply threshold
    if stage1_pred == "Healthy" or stage1_confidence < stage1_threshold:
        final_pred = "Healthy"
    else:
        final_pred = "Anemia"

    # -----------------------------
    # Print Results
    # -----------------------------
    print(f"\n=== Test Case: {label} ===")
    print(f"Stage 1 Prediction: {stage1_pred} (confidence: {stage1_confidence:.2f})")
    print(f"Final Stage 1 Diagnosis: {final_pred}")
