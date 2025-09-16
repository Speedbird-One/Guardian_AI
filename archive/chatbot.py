import json
import os
from dotenv import load_dotenv
from collections import defaultdict
from rapidfuzz import fuzz
from openai import OpenAI

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
REL_CONFIDENCE_THRESHOLD = 0.55  # second condition must be at least 55% of top score

# -----------------------------
# Symptom Normalization
# -----------------------------
def normalize_symptom(symptom):
    symptom = symptom.lower().strip()
    for key, syns in dataset.get("synonyms", {}).items():
        if symptom == key or symptom in syns:
            return key
    return symptom

# -----------------------------
# Hybrid Symptom Extraction
# -----------------------------
def extract_symptoms_hybrid(user_text):
    found = []
    text = user_text.lower()

    # Match synonyms
    for key, syns in dataset.get("synonyms", {}).items():
        for term in [key] + syns:
            if fuzz.partial_ratio(term.lower(), text) >= 80:
                found.append(key)

    # Match condition symptom names directly
    for cond in dataset["conditions"]:
        for s in cond["symptoms"]:
            if fuzz.partial_ratio(s["name"].lower(), text) >= 80:
                found.append(s["name"])

    return list(set(found))

# -----------------------------
# Precompute Symptom Frequency
# -----------------------------
symptom_freq = defaultdict(int)
for cond in dataset["conditions"]:
    for s in cond["symptoms"]:
        symptom_freq[s["name"]] += 1

def normalized_weight(weight, symptom_name):
    freq = symptom_freq.get(symptom_name, 1)
    return weight / freq

# -----------------------------
# Score Conditions
# -----------------------------
def score_conditions(user_symptoms, dataset):
    scores = defaultdict(float)
    normalized = [normalize_symptom(s) for s in user_symptoms]

    for cond in dataset["conditions"]:
        for s in cond["symptoms"]:
            if s["name"] in normalized:
                scores[cond["name"]] += normalized_weight(s.get("weight", 1), s["name"])

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -----------------------------
# Merge Actions with Emojis
# -----------------------------
def merge_actions(cond1, cond2, dataset):
    c1 = next(c for c in dataset["conditions"] if c["name"] == cond1)
    c2 = next(c for c in dataset["conditions"] if c["name"] == cond2)

    precautions = list(set(c1["precautions"] + c2["precautions"]))
    medications = list(set(c1["medications"] + c2["medications"]))

    precautions = [f"- {p}" for p in precautions]
    medications = [f"- {m}" for m in medications]

    severity = c1["severity"] if c1["severity"] == c2["severity"] else "Varies"

    return precautions, medications, severity

# -----------------------------
# Severity Message
# -----------------------------
def severity_message(severity):
    if severity.lower() == "mild":
        return "😊 Overall assessment: This condition appears **mild** and manageable at home."
    elif severity.lower() == "moderate":
        return "⚖️ Overall assessment: This condition is **moderate**. Monitor closely and seek care if it worsens."
    elif severity.lower() == "severe":
        return "🚨 Overall assessment: This condition is **serious**. Please consult a doctor immediately."
    elif severity.lower() == "emergency":
        return "🆘 Overall assessment: This is a **medical emergency**. Seek urgent medical attention!"
    else:
        return "ℹ️  Overall assessment: Severity may vary depending on your exact condition. Please consult a doctor at the earliest."

# -----------------------------
# LLM-Assisted Intent Detection
# -----------------------------
def detect_intent(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant that determines whether a user input is a symptom report for diagnosis or a general health FAQ."},
                {"role": "user", "content": f"Classify the following input as either 'FAQ' or 'Diagnosis':\n\"{user_input}\""}
            ]
        )
        classification = response.choices[0].message.content.strip().lower()
        if "diagnosis" in classification:
            return "diagnosis"
        else:
            return "faq"
    except Exception as e:
        # fallback: use simple heuristic if API fails
        FAQ_KEYWORDS = ["what", "how", "causes", "treatment", "medication", "difference", "side effect", "schedule", "prevent", "symptoms", "duration"]
        text = user_input.lower()
        if any(word in text for word in FAQ_KEYWORDS):
            return "faq"
        return "diagnosis"

# -----------------------------
# Chatbot Loop
# -----------------------------
def chatbot():
    print("\n👋 Hello! I'm **Guardian AI**, your personal health assistant.")
    print("Here's what I can do for you:")
    print("- 💡 Answer your questions about diseases and healthcare.")
    print("- 🧾 Attempt to diagnose conditions from your listed symptoms and suggest suitable precautions/medications.")
    print("- 🧬 Analyse CBC blood reports (coming soon).")
    print("\n👉 Type your question or symptoms (e.g., 'I have a cough and fever').")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Take care! Stay healthy.\n")
            break

        user_symptoms = extract_symptoms_hybrid(user_input)
        intent = detect_intent(user_input)

        if intent == "faq" or len(user_symptoms) < MIN_SYMPTOMS_FOR_DIAGNOSIS:
            # FAQ path
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are Guardian AI, a professional medical assistant that answers healthcare questions clearly and concisely."},
                        {"role": "user", "content": user_input}
                    ]
                )
                answer = response.choices[0].message.content
                print("\n" + "💡 Guardian AI:", answer + "\n")
            except Exception as e:
                print(f"⚠️ Error calling OpenAI API: {e}")
                print("\n" + "💡 Guardian AI: I couldn't detect specific symptoms and my API fallback is currently unavailable. Please try rephrasing your question.\n")
            continue

        # -----------------------------
        # Diagnosis Path
        # -----------------------------
        print(f"\n🔎  Detected symptoms: {', '.join(user_symptoms)}")
        scores = score_conditions(user_symptoms, dataset)
        if not scores:
            print("\n❓ Sorry, I couldn't match your symptoms to any known condition.\n")
            continue

        top_conditions = scores[:2]

        # Relative confidence check
        if len(top_conditions) == 1 or top_conditions[1][1] < REL_CONFIDENCE_THRESHOLD * top_conditions[0][1]:
            chosen = top_conditions[0][0]
            cond_data = next(c for c in dataset["conditions"] if c["name"] == chosen)

            print("\n🤔 Based on your symptoms, the most likely condition is:")
            print(f"- {chosen} (score: {top_conditions[0][1]:.2f})")

            precautions = [f"- {p}" for p in cond_data["precautions"]]
            medications = [f"- {m}" for m in cond_data["medications"]]

            print("\n🛡️  Suggested precautions:")
            for p in precautions:
                print(p)

            print("\n💊 Possible medications (consult a doctor before use):")
            for m in medications:
                print(m)

            print("\n" + severity_message(cond_data["severity"]) + "\n")
        else:
            cond1, cond2 = top_conditions[0][0], top_conditions[1][0]
            precautions, medications, severity = merge_actions(cond1, cond2, dataset)

            print("\n🤔 Based on your symptoms, the most likely conditions are:")
            for cond, score in top_conditions:
                print(f"- {cond} (score: {score:.2f})")

            print("\n🛡️  Suggested precautions:")
            for p in precautions:
                print(p)

            print("\n💊 Possible medications (consult a doctor before use):")
            for m in medications:
                print(m)

            print("\n" + severity_message(severity) + "\n")

# -----------------------------
# Run Chatbot
# -----------------------------
if __name__ == "__main__":
    chatbot()
