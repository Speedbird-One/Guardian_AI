import os
import re
import sys
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
import joblib

# -----------------------------
# CBC Feature Synonyms
# -----------------------------
CBC_SYNONYMS = {
    "HGB": ["haemoglobin", "hemoglobin", "hgb"],
    "HCT": ["haematocrit", "hematocrit", "hct"],
    "RBC": ["rbc", "total rbc", "total r.b.c. count"],
    "WBC": ["wbc", "total wbc", "total w.b.c. count"],
    "PLT": ["platelet", "total platelet count", "plt"],
    "NEUTp": ["neutrophils", "neut%", "neut"],
    "LYMp": ["lymphocytes", "lymp%", "lym"],
    "NEUTn": ["absolute neutrophils", "neut#"],
    "LYMn": ["absolute lymphocytes", "lym#"],
    "MCV": ["mcv"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    "PDW": ["pdw"],
    "PCT": ["pct"]
}

# -----------------------------
# Utilities: Read PDF/Image Text
# -----------------------------
def read_file_text(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():  # scanned PDF → OCR fallback
            try:
                text = pytesseract.image_to_string(Image.open(file_path))
            except:
                pass
    else:
        try:
            text = pytesseract.image_to_string(Image.open(file_path))
        except:
            pass
    return text

# -----------------------------
# Extract CBC Values
# -----------------------------
def extract_cbc_values(text):
    values = {}
    for feature, keywords in CBC_SYNONYMS.items():
        for key in keywords:
            pattern = rf"{key}\s*[:\-]?\s*([\d.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values[feature] = float(match.group(1))
                break
    return values

# -----------------------------
# Load Models & Medians
# -----------------------------
clf_stage1 = joblib.load("models/cbc_model_stage1.pkl")
clf_stage2 = joblib.load("models/cbc_model_stage2.pkl")
medians = joblib.load("models/cbc_medians.pkl")

# Stage 1 probability threshold
stage1_threshold = 0.8

# -----------------------------
# Prediction Function
# -----------------------------
def predict_from_report(file_path):
    text = read_file_text(file_path)
    extracted = extract_cbc_values(text)
    df_report = pd.DataFrame([extracted])

    # Stage 1: Healthy vs Anemia
    df_stage1 = df_report.reindex(columns=clf_stage1.feature_names_in_).fillna(medians)
    stage1_pred = clf_stage1.predict(df_stage1)[0]
    stage1_proba = clf_stage1.predict_proba(df_stage1)[0]
    stage1_confidence = stage1_proba.max()

    if stage1_pred == "Healthy" or stage1_confidence < stage1_threshold:
        final_pred = "Healthy"
        stage2_pred = None
        stage2_proba = None
    else:
        # Stage 2: Anemia subtype
        df_stage2 = df_report.reindex(columns=clf_stage2.feature_names_in_).fillna(medians)
        stage2_pred = clf_stage2.predict(df_stage2)[0]
        stage2_proba = clf_stage2.predict_proba(df_stage2)[0]
        final_pred = stage2_pred

    # -----------------------------
    # Print Results
    # -----------------------------
    print("\n📊 Extracted CBC Values:")
    print(df_report)

    print("\n🧾 Stage 1: Healthy vs Anemia")
    for cls, p in zip(clf_stage1.classes_, stage1_proba):
        print(f"{cls}: {p:.2f}")
    print(f"Stage 1 Final Prediction: {final_pred} (confidence: {stage1_confidence:.2f})")

    if stage2_pred:
        print("\n🧾 Stage 2: Anemia Subtype Probabilities")
        for cls, p in zip(clf_stage2.classes_, stage2_proba):
            print(f"{cls}: {p:.2f}")
        print(f"Stage 2 Final Prediction: {stage2_pred}")

    return final_pred

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    report_file = sys.argv[1]   #input("Enter CBC PDF file path: ").strip()
    if not os.path.exists(report_file):
        print(f"File '{report_file}' not found.")
    else:
        predict_from_report(report_file)
