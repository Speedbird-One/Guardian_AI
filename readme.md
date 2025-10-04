# Guardian AI - AI-Driven Public Health Chatbot

**Team:** "Spiderman: Far from Diseases "
**Problem Statement:** SIH25049 - AI-Driven Public Health Chatbot for Disease Awareness 

Guardian AI is a multilingual health chatbot designed to provide reliable preventive care information, bridging the healthcare gap for rural and semi-urban communities.

## Key Features
Our platform is packed with features to make public health information accessible and actionable:

- **Multilingual Q&A:** Get instant answers to health questions about symptoms, hygiene, and prevention in multiple languages via text and voice.
- **Blood Report Analysis:** Upload a blood report for an instant analysis of potential health risks, complete with personalized health tips.
- **Live Outbreak Alerts:** Receive real-time alerts about local disease outbreaks using data from government databases.
- **Symptom Checker:** A preliminary assessment tool to help guide users on their next steps for care.
- **Vaccine Tracker:** Easily manage family immunization schedules and find nearby vaccination camps.
- **Connected Care:** Directly connect with doctors for teleconsultation through integration with platforms like eSanjeevani.
- **Interactive Health Quizzes:** Engage with fun quizzes to improve health literacy.

---

## Technical Architecture
Guardian AI is built on a modern, scalable tech stack designed for robust performance and reliability. The system processes user queries through a sophisticated AI/ML pipeline to provide accurate, actionable insights.

Our architecture is composed of the following layers:
- **Frontend:** A responsive and intuitive user interface built with React.js / Next.js.
- **Backend:** Scalable and robust APIs powered by Django or FastAPI (Python).
- **AI/ML Core:**
  - **Natural Language Understanding:** Pre-trained Large Language Models from Hugging Face provide conversational AI capabilities.
  - **Intelligent Orchestration:** LangChain connects the language models with medical knowledge bases from trusted sources like WHO and ICMR.
  - **Predictive Analytics:** Scikit-learn is used to deploy machine learning models for blood report analysis.

---

## Project Impact
Guardian AI aims to empower communities and strengthen the public health infrastructure in three key ways:

- **Empowering Community Health:** Enables early identification of health risks and provides clear information in local languages, helping users make informed decisions.
- **Bridging the Healthcare Gap:** Offers 24/7 on-demand access to preliminary health advice, saving families valuable time and money on travel for non-urgent queries.
- **Strengthening Public Health:** Frees up healthcare workers by handling common questions and acts as an early warning system through outbreak alerts, reducing the strain on the public health system.

---

## Getting Started
To get a local copy up and running, follow these simple steps.

**Prerequisites**

- Node.js & npm
- Python 3.8+ & pip

**Installation**

1. Clone the repo

```bash
git clone https://github.com/Speedbird-One/Guardian_AI.git
```

2. Install dependencies

```bash
pip install -r requirements.txt`
```

3. Set up environment variables

Create a `.env` file and add your API keys.

4. Run the application

Start the backend server: `python server.py`

Open the frontend: `html/index.html`

## License
This project is distributed under the MIT License. See `LICENSE.txt` for more information.
