import pandas as pd
import joblib

# -----------------------------
# Load models and medians
# -----------------------------
clf_stage1 = joblib.load("models/cbc_model_stage1.pkl")
clf_stage2 = joblib.load("models/cbc_model_stage2.pkl")
medians = joblib.load("models/cbc_medians.pkl")

# Stage 1 threshold
stage1_threshold = 0.8

# -----------------------------
# Synthetic Test Cases
# -----------------------------
test_cases = [
    # 1. Healthy
    {
        "label": "Healthy",
        "HGB": 14.2, "HCT": 42.1, "RBC": 4.8, "WBC": 7500, "PLT": 230000,
        "NEUTp": 60, "LYMp": 30, "NEUTn": 4500, "LYMn": 2250,
        "MCV": 88, "MCH": 29, "MCHC": 33, "PDW": 11, "PCT": 0.22
    },
    # 2. Borderline anemia
    {
        "label": "Borderline Anemia",
        "HGB": 12.5, "HCT": 37, "RBC": 4.1, "WBC": 6000, "PLT": 220000,
        "NEUTp": 50, "LYMp": 35, "NEUTn": 3000, "LYMn": 1800,
        "MCV": 85, "MCH": 29, "MCHC": 34, "PDW": 12, "PCT": 0.21
    },
    # 3. Normocytic anemia
    {
        "label": "Normocytic Anemia",
        "HGB": 9.8, "HCT": 29.5, "RBC": 3.6, "WBC": 5200, "PLT": 210000,
        "NEUTp": 55, "LYMp": 35, "NEUTn": 2860, "LYMn": 1820,
        "MCV": 82, "MCH": 28, "MCHC": 34, "PDW": 12, "PCT": 0.20
    },
    # 4. Microcytic anemia
    {
        "label": "Microcytic Anemia",
        "HGB": 8.5, "HCT": 27, "RBC": 3.4, "WBC": 6000, "PLT": 200000,
        "NEUTp": 50, "LYMp": 40, "NEUTn": 3000, "LYMn": 2400,
        "MCV": 70, "MCH": 22, "MCHC": 32, "PDW": 11, "PCT": 0.19
    },
    # 5. Macrocytic anemia
    {
        "label": "Macrocytic Anemia",
        "HGB": 10.2, "HCT": 31, "RBC": 3.3, "WBC": 6200, "PLT": 210000,
        "NEUTp": 55, "LYMp": 35, "NEUTn": 2900, "LYMn": 1800,
        "MCV": 110, "MCH": 36, "MCHC": 33, "PDW": 13, "PCT": 0.21
    }
]

# -----------------------------
# Test Suite Execution
# -----------------------------
for case in test_cases:
    df = pd.DataFrame([case])
    df_label = df.pop("label")[0]

    # Stage 1: Healthy vs Anemia
    df_stage1 = df.reindex(columns=clf_stage1.feature_names_in_).fillna(medians)
    stage1_pred = clf_stage1.predict(df_stage1)[0]
    stage1_proba = clf_stage1.predict_proba(df_stage1)[0]
    stage1_confidence = stage1_proba.max()

    if stage1_pred == "Healthy" or stage1_confidence < stage1_threshold:
        final_pred = "Healthy"
        stage2_pred = None
        stage2_proba = None
    else:
        # Stage 2: Anemia subtype
        df_stage2 = df.reindex(columns=clf_stage2.feature_names_in_).fillna(medians)
        stage2_pred = clf_stage2.predict(df_stage2)[0]
        stage2_proba = clf_stage2.predict_proba(df_stage2)[0]
        final_pred = stage2_pred

    # -----------------------------
    # Print Results
    # -----------------------------
    print(f"\n=== Test Case: {df_label} ===")
    print(f"Stage 1 Prediction: {stage1_pred} (confidence: {stage1_confidence:.2f})")
    if final_pred == "Healthy":
        print("Final Diagnosis: Healthy")
    else:
        print("Stage 2 Prediction: Anemia subtype")
        for cls, p in zip(clf_stage2.classes_, stage2_proba):
            print(f"  {cls}: {p:.2f}")
        print(f"Final Diagnosis: {final_pred}")
