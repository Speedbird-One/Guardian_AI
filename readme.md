# Guardian AI

Guardian AI is a healthcare chatbot that combines machine learning, rule-based symptom mapping, and LLMs to provide:

- Symptom-based diagnosis using a curated medical dataset.
- CBC blood report analysis with trained ML models (two-stage Random Forest)
- Healthcare FAQs powered by GPT-4o-mini.
- Flask API + web frontend for easy user interaction.

Key features:

- Hybrid exact + fuzzy symptom extraction for robust matching.
- CBC anomaly detection + anemia subtype classification.
- Conversation memory for context-aware dialogue.
- Clean console/web output (Markdown-free).

Future Work: CBC PDF upload via Flask, persistent database for health records, predictive CBC trend analysis, and personalised preventive healthcare plans.
