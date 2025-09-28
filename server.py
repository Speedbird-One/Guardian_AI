# server.py
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import (
    extract_symptoms_hybrid,
    detect_intent,
    score_conditions,
    merge_actions,
    severity_message,
    dataset,
    MIN_SYMPTOMS_FOR_DIAGNOSIS,
    REL_CONFIDENCE_THRESHOLD,
    MAX_MEMORY
)

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

app = Flask(__name__)
CORS(app)

# Minimal in-memory session per user (can be expanded)
chat_memory = []  # stores last MAX_MEMORY messages

@app.route("/chat", methods=["POST"])
def chat():
    global chat_memory
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_input = data["message"]
    chat_memory.append({"role": "user", "content": user_input})
    chat_memory = chat_memory[-MAX_MEMORY:]

    # Extract symptoms
    user_symptoms = extract_symptoms_hybrid(user_input)
    intent = detect_intent(user_input)

    # FAQ path (fallback to LLM)
    if intent == "faq" or len(user_symptoms) < MIN_SYMPTOMS_FOR_DIAGNOSIS:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                        {"role":"system", "content": "You are Guardian AI, a professional medical assistant. "
                                                    "Provide clear, accurate, but short and crisp answers (no more than 5-6 line) to user questions."},
                        {"role":"user", "content": user_input}
                    ]
            )
            answer = response.choices[0].message.content.strip()
            answer = strip_markdown(answer)
            chat_memory.append({"role": "assistant", "content": answer})
            chat_memory = chat_memory[-MAX_MEMORY:]
            return jsonify({"response": answer})
        except Exception as e:
            return jsonify({"response": f"Unable to process your question: {str(e)}"})

    # Diagnosis path
    scores = score_conditions(user_symptoms, dataset)
    if not scores:
        return jsonify({"response": "❓ No matching conditions found."})

    top_conditions = scores[:2]
    response_lines = []

    if len(top_conditions) == 1 or top_conditions[1][1] < REL_CONFIDENCE_THRESHOLD * top_conditions[0][1]:
        chosen = top_conditions[0][0]
        cond_data = next(c for c in dataset["conditions"] if c["name"] == chosen)
        response_lines.append(f"🤔 Most likely condition: {chosen} (score: {top_conditions[0][1]:.2f})")
        response_lines.append("\n🛡️  Precautions:")
        for p in cond_data["precautions"]:
            response_lines.append(f"- {p}")
        response_lines.append("\n💊 Medications (consult a doctor):")
        for m in cond_data["medications"]:
            response_lines.append(f"- {m}")
        response_lines.append("\n" + severity_message(cond_data["severity"]))
    else:
        cond1, cond2 = top_conditions[0][0], top_conditions[1][0]
        precautions, medications, severity = merge_actions(cond1, cond2, dataset)
        response_lines.append("🤔 Most likely conditions:")
        for cond, score in top_conditions:
            response_lines.append(f"- {cond} (score: {score:.2f})")
        response_lines.append("\n🛡️  Precautions:")
        for p in precautions:
            response_lines.append(p)
        response_lines.append("\n💊 Medications (consult a doctor):")
        for m in medications:
            response_lines.append(m)
        response_lines.append("\n" + severity_message(severity))

    answer = "\n".join(response_lines)
    chat_memory.append({"role": "assistant", "content": answer})
    chat_memory = chat_memory[-MAX_MEMORY:]
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
