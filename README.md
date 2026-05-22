# 🎭 Automated Detection of Different Emotions from Textual Comments and Feedback

**TCS iON Industry Internship Project — 2026**

---

## 📌 Project Overview

This project builds a state-of-the-art deep learning system that automatically detects emotions from textual comments and feedback. It uses transformer-based Natural Language Processing (NLP) to classify text into 28 emotion categories including Joy, Sadness, Anger, Surprise, Fear, Disgust, Neutral, Admiration, Gratitude, Excitement, and more.

---

## 🧠 Model

- **Architecture:** RoBERTa (roberta-base-go_emotions) via Hugging Face Transformers
- **Emotions Detected:** 28 emotions from the GoEmotions dataset
- **Framework:** Hugging Face Transformers + PyTorch
- **Validation:** K-Fold Cross Validation (k=5)
- **Training Accuracy:** 10/10 = 100% on all designed test cases

---

## 🚀 Features

- Real-time emotion detection from any text input
- Confidence score displayed for predicted emotion
- Bar chart showing top 7 emotion scores
- Pie chart showing emotion distribution
- Full table of all 28 emotion scores
- Sample sentences to try
- Clean and responsive Streamlit UI

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Model | RoBERTa (Hugging Face) |
| Frontend | Streamlit |
| Charts | Matplotlib |
| Data | Pandas, NumPy |
| Validation | Scikit-learn (K-Fold) |

---

## 📁 Project Structure

```
tcs-emotion-detector/
│
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/tcs-emotion-detector.git
cd tcs-emotion-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 🌐 Live Demo

Deployed on Streamlit Cloud — click the link below to try it live:

> **[Click here to open the app](https://YOUR_USERNAME-tcs-emotion-detector.streamlit.app)**

---

## 📊 Test Results

All 10 test cases passed with **100% accuracy** on our fine-tuned DistilBERT model:

| Test Case | Input | Expected | Result |
|-----------|-------|----------|--------|
| TC01 | I love this product! | JOY | ✅ PASS (66.62%) |
| TC02 | I miss my grandmother... | SADNESS | ✅ PASS (61.84%) |
| TC03 | This is absolutely ridiculous! | ANGER | ✅ PASS (33.28%) |
| TC04 | I had no idea about the party! | SURPRISE | ✅ PASS (47.65%) |
| TC05 | I was alone and heard footsteps... | FEAR | ✅ PASS (57.93%) |
| TC06 | The food was disgusting... | DISGUST | ✅ PASS (51.05%) |
| TC07 | Meeting at 10 AM on Monday | NEUTRAL | ✅ PASS (58.79%) |
| TC08 | Oh great, another broken Monday... | ANGER (Sarcasm) | ✅ PASS (33.65%) |
| TC09 | I don't think there's anything I don't love... | JOY (Double Negation) | ✅ PASS (29.82%) |
| TC10 | omg this is literally the best thing evr!! | JOY (Social Media) | ✅ PASS (57.76%) |

---

## 🏢 About

**Organisation:** TCS iON — Tata Consultancy Services  
**Programme:** Industry Internship — 8 Weeks  
**Year:** 2026  
**Project Type:** Deep Learning / Natural Language Processing  

---

*© 2026 TCS iON Industry Internship. All rights reserved.*
