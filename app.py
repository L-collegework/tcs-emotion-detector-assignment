import streamlit as st
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Emotion Detector | TCS iON", page_icon="🎭", layout="centered")

st.markdown("""
<style>
.title-text { font-size:2.2rem; font-weight:700; color:#1F3864; text-align:center; margin-bottom:0.2rem; }
.subtitle-text { font-size:1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
.result-box { background:#fff; border-left:6px solid #2E5DA6; border-radius:10px;
              padding:1.2rem 1.5rem; margin-top:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
.emotion-label { font-size:2rem; font-weight:700; }
.score-text { font-size:1rem; color:#555; margin-top:0.3rem; }
.footer-text { text-align:center; color:#bbb; font-size:0.78rem; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

EMOJI = {"joy":"😊","sadness":"😢","anger":"😠","surprise":"😮","fear":"😨","disgust":"🤢","neutral":"😐"}
COLOR = {"joy":"#f6c90e","sadness":"#4a90d9","anger":"#e74c3c","surprise":"#f39c12","fear":"#8e44ad","disgust":"#27ae60","neutral":"#95a5a6"}
def get_emoji(l): return EMOJI.get(l.lower(),"🎭")
def get_color(l): return COLOR.get(l.lower(),"#2E5DA6")

@st.cache_resource
def load_model():
    with st.spinner("⏳ Loading trained model — please wait..."):
        tokenizer = AutoTokenizer.from_pretrained("./emotion_model_final")
        model = AutoModelForSequenceClassification.from_pretrained("./emotion_model_final")
        model.eval()
        with open("./label_encoder.pkl","rb") as f:
            le = pickle.load(f)
    return tokenizer, model, le

tokenizer, model, le = load_model()
label_names = list(le.classes_)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    pred_id = int(np.argmax(probs))
    return label_names[pred_id], float(probs[pred_id]), label_names, probs.tolist()

# ── Header ────────────────────────────────────────────────────
st.markdown("<div class='title-text'>🎭 Emotion Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>TCS iON Industry Project &nbsp;·&nbsp; Automated Detection of Emotions from Textual Comments and Feedback</div>", unsafe_allow_html=True)
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────
user_input = st.text_area("Enter a sentence or paragraph:", 
                           placeholder="e.g. I love this product! It made my day so much better.", 
                           height=150)
c1, c2 = st.columns([3,1])
with c1: analyse = st.button("🔍 Analyse Emotion", use_container_width=True)
with c2: clear   = st.button("🗑️ Clear",           use_container_width=True)
if clear: st.rerun()

# ── Results ───────────────────────────────────────────────────
if analyse:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before clicking Analyse.")
    else:
        with st.spinner("Analysing..."):
            emotion, confidence, all_labels, all_scores = predict(user_input.strip())

        color = get_color(emotion)
        emoji = get_emoji(emotion)

        st.markdown(f"""
        <div class='result-box'>
            <div class='emotion-label' style='color:{color};'>{emoji} &nbsp; {emotion.upper()}</div>
            <div class='score-text'>Confidence Score: <strong>{confidence:.2%}</strong></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart
        st.subheader("📊 Emotion Score Breakdown")
        fig, ax = plt.subplots(figsize=(8,4))
        bar_colors = [get_color(l) for l in all_labels]
        bars = ax.barh(all_labels, all_scores, color=bar_colors, edgecolor="white", height=0.55)
        top_idx = all_labels.index(emotion)
        bars[top_idx].set_edgecolor("#1F3864")
        bars[top_idx].set_linewidth(2.5)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.set_title("Confidence Across All Emotions", fontsize=12, pad=10)
        ax.invert_yaxis()
        for bar, score in zip(bars, all_scores):
            ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                    f"{score:.2%}", va="center", fontsize=9.5, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        # Pie chart
        st.subheader("🥧 Emotion Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,6))
        wedges, texts, autotexts = ax2.pie(
            all_scores,
            labels=[f"{get_emoji(l)} {l}" for l in all_labels],
            colors=bar_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.80,
            wedgeprops=dict(edgecolor="white", linewidth=1.5)
        )
        for at in autotexts: at.set_fontsize(9)
        ax2.set_title("Emotion Confidence Distribution", fontsize=12, pad=15)
        fig2.tight_layout()
        st.pyplot(fig2)

        # Raw scores table
        with st.expander("📋 View Raw Scores Table"):
            df_res = pd.DataFrame({
                "Emotion":          [l.upper() for l in all_labels],
                "Confidence Score": [f"{s:.4f}" for s in all_scores],
                "Percentage":       [f"{s:.2%}"  for s in all_scores]
            })
            st.dataframe(df_res, use_container_width=True, hide_index=True)

# ── Sample sentences ──────────────────────────────────────────
with st.expander("💡 Try these sample sentences"):
    for label, sentence in [
        ("😊 Joy",      "I love this product! It made my day so much better."),
        ("😢 Sadness",  "I miss my grandmother so much. Life feels empty without her."),
        ("😠 Anger",    "This is absolutely ridiculous! The service was terrible."),
        ("😮 Surprise", "I had no idea they were planning a surprise party for me!"),
        ("😨 Fear",     "I was alone in the dark hallway and heard footsteps behind me."),
        ("🤢 Disgust",  "The food was absolutely disgusting. It smelled awful."),
        ("😐 Neutral",  "The meeting is scheduled for Monday at 10 AM."),
        ("😠 Sarcasm",  "Oh great, another Monday with zero working systems. Just what I needed."),
    ]:
        st.markdown(f"**{label}** — *{sentence}*")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 TCS iON Project")
    st.markdown("""
**Project:** Automated Detection of Different Emotions from Textual Comments and Feedback

**Model:** DistilBERT (fine-tuned on our custom dataset)

**Emotions Detected:**
- 😊 Joy &nbsp;&nbsp; 😢 Sadness &nbsp;&nbsp; 😠 Anger
- 😮 Surprise &nbsp;&nbsp; 😨 Fear
- 🤢 Disgust &nbsp;&nbsp; 😐 Neutral

**Framework:** Hugging Face Transformers

**Validation:** K-Fold Cross Validation (k=5)

**Frontend:** Streamlit
    """)
    st.markdown("---")
    st.caption("TCS iON Industry Internship · 2026")

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    "<div class='footer-text'>TCS iON Industry Internship · 8-Week Project · 2026 · "
    "Automated Emotion Detection using NLP & Transformers</div>",
    unsafe_allow_html=True
)
