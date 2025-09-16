# chatbot.py
import json
import os
import re
from collections import defaultdict
from rapidfuzz import fuzz
from openai import OpenAI
import pandas as pd
import pdfplumber
from PIL import Image
import pytesseract
import joblib
from dotenv import load_dotenv

# -----------------------------
# Load Dataset
# -----------------------------
with open("data/conditions_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# -----------------------------
# OpenAI API Setup
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Settings
# -----------------------------
MIN_SYMPTOMS_FOR_DIAGNOSIS = 2
REL_CONFIDENCE_THRESHOLD = 0.55
CBC_WEIGHT_BOOST = 0.25
MAX_MEMORY = 10  # last 10 messages (user + assistant)

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
# Load CBC Models & Medians
# -----------------------------
clf_stage1 = joblib.load("models/cbc_model_stage1.pkl")
clf_stage2 = joblib.load("models/cbc_model_stage2.pkl")
medians = joblib.load("models/cbc_medians.pkl")
stage1_threshold = 0.8

# -----------------------------
# Utility: Strip Markdown Formatting
# -----------------------------
def strip_markdown(text):
    if not text:
        return text
    # Remove bold/italic markers
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    # Remove inline code/backticks
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Remove headings (###, ##, # at start of lines)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text.strip()

# -----------------------------
# CBC Utilities
# -----------------------------
def read_file_text(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():  # OCR fallback
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

def predict_cbc(file_path):
    text = read_file_text(file_path)
    extracted = extract_cbc_values(text)
    df_report = pd.DataFrame([extracted])

    df_stage1 = df_report.reindex(columns=clf_stage1.feature_names_in_).fillna(medians)
    stage1_pred = clf_stage1.predict(df_stage1)[0]
    stage1_proba = clf_stage1.predict_proba(df_stage1)[0]
    stage1_confidence = stage1_proba.max()

    if stage1_pred == "Healthy" or stage1_confidence < stage1_threshold:
        final_pred = "Healthy"
        stage2_pred = None
    else:
        df_stage2 = df_report.reindex(columns=clf_stage2.feature_names_in_).fillna(medians)
        stage2_pred = clf_stage2.predict(df_stage2)[0]
        final_pred = stage2_pred

    return df_report.iloc[0].to_dict(), final_pred, stage1_proba, stage2_pred

# -----------------------------
# Symptom Utilities
# -----------------------------
def normalize_symptom(symptom):
    symptom = symptom.lower().strip()
    for key, syns in dataset.get("synonyms", {}).items():
        if symptom == key or symptom in syns:
            return key
    return symptom

def extract_symptoms_hybrid(user_text):
    found = []
    text = user_text.lower()

    # Exact phrase / synonym match
    for key, syns in dataset.get("synonyms", {}).items():
        for term in [key] + syns:
            if re.search(rf"\b{re.escape(term.lower())}\b", text):
                found.append(key)

    # Fuzzy fallback
    for cond in dataset["conditions"]:
        for s in cond["symptoms"]:
            if s["name"] not in found:
                if fuzz.partial_ratio(s["name"].lower(), text) >= 90:
                    found.append(s["name"])

    return list(set(found))

symptom_freq = defaultdict(int)
for cond in dataset["conditions"]:
    for s in cond["symptoms"]:
        symptom_freq[s["name"]] += 1

def normalized_weight(weight, symptom_name):
    freq = symptom_freq.get(symptom_name, 1)
    return weight / freq

def score_conditions(user_symptoms, dataset, cbc_prediction=None):
    scores = defaultdict(float)
    normalized = [normalize_symptom(s) for s in user_symptoms]

    for cond in dataset["conditions"]:
        cond_symptoms = [s["name"] for s in cond["symptoms"]]
        for s in cond["symptoms"]:
            if s["name"] in normalized:
                w = s.get("weight", 1)

                # CBC boost if relevant
                if cbc_prediction and "anemia" in cbc_prediction.lower() and "anemia" in cond["name"].lower():
                    w *= (1 + CBC_WEIGHT_BOOST)

                # Normalize by number of condition symptoms
                scores[cond["name"]] += normalized_weight(w, s["name"]) / len(cond_symptoms)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def merge_actions(cond1, cond2, dataset):
    c1 = next(c for c in dataset["conditions"] if c["name"] == cond1)
    c2 = next(c for c in dataset["conditions"] if c["name"] == cond2)
    precautions = list(set(c1["precautions"] + c2["precautions"]))
    medications = list(set(c1["medications"] + c2["medications"]))
    precautions = [f"- {p}" for p in precautions]
    medications = [f"- {m}" for m in medications]
    severity = c1["severity"] if c1["severity"] == c2["severity"] else "Varies"
    return precautions, medications, severity

def severity_message(severity):
    if severity.lower() == "mild":
        return "😊 Overall assessment: This condition appears mild and manageable at home."
    elif severity.lower() == "moderate":
        return "⚖️ Overall assessment: This condition is moderate. Monitor closely and seek care if it worsens."
    elif severity.lower() == "severe":
        return "🚨 Overall assessment: This condition is serious. Please consult a doctor immediately."
    elif severity.lower() == "emergency":
        return "🆘 Overall assessment: This is a medical emergency. Seek urgent medical attention!"
    else:
        return "ℹ️  Overall assessment: Severity may vary depending on your exact condition. Please consult a doctor at the earliest."

# -----------------------------
# LLM Intent Detection
# -----------------------------
def detect_intent(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content":"You are Guardian AI, a medical assistant that classifies input as 'FAQ' or 'Diagnosis'."},
                {"role": "user","content": user_input}
            ]
        )
        classification = response.choices[0].message.content.strip().lower()
        return "diagnosis" if "diagnosis" in classification else "faq"
    except:
        FAQ_KEYWORDS = ["what","how","causes","treatment","medication","difference","side effect","schedule","prevent","symptoms","duration"]
        return "faq" if any(k in user_input.lower() for k in FAQ_KEYWORDS) else "diagnosis"

# -----------------------------
# Mild conditions whitelist (robust)
# -----------------------------
MILD_CONDITIONS_WHITELIST = {
    "common cold",
    "influenza",
    "influenza (flu)",
    "tonsillitis / pharyngitis"
}

# -----------------------------
# Chatbot Loop with Memory
# -----------------------------
def chatbot():
    print("\n👋 Hello! I'm Guardian AI.")
    print("Here's what I can do for you:")
    print("- 💡 Answer your questions about diseases and healthcare.")
    print("- 🧾 Attempt to diagnose conditions from your listed symptoms and suggest suitable precautions/medications.")
    print("- 🧬 Analyse CBC blood reports.")
    print("\nType your question or symptoms (e.g., 'I have a cough and fever').")
    print("Type 'cbc:<file_path>' for a CBC report.")
    print("Type 'quit' to exit.\n")

    latest_cbc_prediction = None
    messages = [{"role":"system","content":"You are Guardian AI, a professional medical assistant."}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit","exit"]:
            print("\n👋 Take care! Stay healthy.\n")
            break

        # CBC Report Input
        if user_input.lower().startswith("cbc:"):
            file_path = user_input[4:].strip()
            if not os.path.exists(file_path):
                print(f"⚠️ File '{file_path}' not found.\n")
                continue
            extracted, cbc_prediction, stage1_proba, stage2_pred = predict_cbc(file_path)
            latest_cbc_prediction = cbc_prediction

            print("\n📊 CBC Values:")
            for k,v in extracted.items(): print(f"- {k}: {v}")
            if stage2_pred: print(f"ℹ️  Anemia subtype detected: {stage2_pred}")

            # Map CBC to dataset conditions
            cbc_conditions = [c for c in dataset["conditions"] if cbc_prediction.lower() in c["name"].lower()]
            if cbc_conditions:
                print("\n🤔 Likely condition(s) from dataset:")
                for c in cbc_conditions: print(f"- {c['name']}")
                precautions = list({p for c in cbc_conditions for p in c["precautions"]})
                medications = list({m for c in cbc_conditions for m in c["medications"]})
                print("\n🛡️  Precautions:")
                for p in precautions: print(f"- {p}")
                print("\n💊 Medications (consult a doctor before use):")
                for m in medications: print(f"- {m}")
                severity_set = set(c["severity"].lower() for c in cbc_conditions)
                severity = severity_set.pop() if len(severity_set)==1 else "Varies"
                print("\n" + strip_markdown(severity_message(severity)) + "\n")
            else:
                print(f"\nℹ️  CBC indicates: {cbc_prediction}\n")

            # Save to memory
            messages.append({"role":"user","content":user_input})
            messages.append({"role":"assistant","content":f"CBC analyzed: {cbc_prediction}"})
            messages = messages[-MAX_MEMORY:]
            continue

        # Symptom Input
        user_symptoms = extract_symptoms_hybrid(user_input)
        intent = detect_intent(user_input)

        # FAQ Path
        if intent=="faq" or len(user_symptoms)<MIN_SYMPTOMS_FOR_DIAGNOSIS:
            messages.append({"role":"user","content":user_input})
            messages = messages[-MAX_MEMORY:]
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                answer = response.choices[0].message.content
                answer = strip_markdown(answer)
                print("\n💡 Guardian AI:", answer + "\n")
                messages.append({"role":"assistant","content":answer})
                messages = messages[-MAX_MEMORY:]
            except:
                print("\n💡 Guardian AI: Unable to process your question.\n")
            continue

        # Diagnosis Path
        print(f"\n🔎 Detected symptoms: {', '.join(user_symptoms)}")
        scores = score_conditions(user_symptoms,dataset,latest_cbc_prediction)
        if not scores:
            print("\n❓ No matching conditions found.\n")
            continue

        top_conditions = scores[:2]

        # ✅ Always show mild conditions if they appear in top matches
        mild_hit = None
        for cond, score in top_conditions:
            if cond.lower() in MILD_CONDITIONS_WHITELIST:
                mild_hit = (cond, score)
                break

        if mild_hit:
            cond, score = mild_hit
            cond_data = next(c for c in dataset["conditions"] if c["name"].lower() == cond.lower())
            print("\n🤔 Most likely condition:")
            print(f"- {cond} (score: {score:.2f})")
            if latest_cbc_prediction: print(f"💉 CBC influence: {latest_cbc_prediction}")
            print("\n🛡️  Precautions:")
            for p in cond_data["precautions"]: print(f"- {p}")
            print("\n💊 Medications (consult a doctor):")
            for m in cond_data["medications"]: print(f"- {m}")
            print("\n" + strip_markdown(severity_message(cond_data["severity"])) + "\n")

            messages.append({"role":"user","content":user_input})
            messages = messages[-MAX_MEMORY:]
            continue

        # General fallback logic
        if len(user_symptoms) <= 2:
            if top_conditions[0][1] < 0.5:
                print("\n🤔 Your symptoms are quite general and could indicate several mild conditions (like a common cold or flu).")
                print("Please monitor for additional symptoms and consult a doctor if things worsen.\n")
                messages.append({"role":"user","content":user_input})
                messages = messages[-MAX_MEMORY:]
                continue
        elif len(top_conditions) > 1 and abs(top_conditions[0][1] - top_conditions[1][1]) < 0.1 and top_conditions[0][1] < 0.6:
            print("\n🤔 Your symptoms are non-specific and could indicate several mild conditions (like a common cold or flu).")
            print("Please monitor and consult a doctor if things worsen.\n")
            messages.append({"role":"user","content":user_input})
            messages = messages[-MAX_MEMORY:]
            continue

        # Normal diagnosis logic
        if len(top_conditions)==1 or top_conditions[1][1]<REL_CONFIDENCE_THRESHOLD*top_conditions[0][1]:
            chosen = top_conditions[0][0]
            cond_data = next(c for c in dataset["conditions"] if c["name"]==chosen)
            print("\n🤔 Most likely condition:")
            print(f"- {chosen} (score: {top_conditions[0][1]:.2f})")
            if latest_cbc_prediction: print(f"💉 CBC influence: {latest_cbc_prediction}")
            print("\n🛡️  Precautions:")
            for p in cond_data["precautions"]: print(f"- {p}")
            print("\n💊 Medications (consult a doctor):")
            for m in cond_data["medications"]: print(f"- {m}")
            print("\n" + strip_markdown(severity_message(cond_data["severity"])) + "\n")
        else:
            cond1, cond2 = top_conditions[0][0], top_conditions[1][0]
            precautions, medications, severity = merge_actions(cond1,cond2,dataset)
            print("\n🤔 Most likely conditions:")
            for cond,score in top_conditions: print(f"- {cond} (score: {score:.2f})")
            if latest_cbc_prediction: print(f"💉 CBC influence: {latest_cbc_prediction}")
            print("\n🛡️  Precautions:")
            for p in precautions: print(p)
            print("\n💊 Medications (consult a doctor):")
            for m in medications: print(m)
            print("\n" + strip_markdown(severity_message(severity)) + "\n")

        # Save to memory
        messages.append({"role":"user","content":user_input})
        messages = messages[-MAX_MEMORY:]


if __name__=="__main__":
    chatbot()
