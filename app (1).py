"""
Toxic Comment Detection — Streamlit Deployment App
DAT401 Natural Language Processing — Classical NLP Assignment

Run locally with:  streamlit run app.py
Requires: final_model.pkl in the same directory (a fitted scikit-learn
Pipeline containing the TF-IDF vectorizer + the selected classifier).
"""

import re
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ---- One-time NLTK resource download (cached) ----
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

setup_nltk()

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Identical preprocessing pipeline used at training time."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")


# ---- Page setup ----
st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Toxic Comment Detector")
st.caption(
    "Classical NLP pipeline (TF-IDF + Logistic Regression) trained on the "
    "Jigsaw Toxic Comment dataset — DAT401 Classical NLP Assignment"
)

model = load_model()

with st.expander("ℹ️ About this model — validation performance"):
    st.markdown(
        """
        | Metric | Non-Toxic | Toxic |
        |---|---|---|
        | Precision | 0.98 | 0.68 |
        | Recall | 0.96 | 0.84 |
        | F1-score | 0.97 | 0.75 |

        Overall accuracy: **95%** &nbsp;|&nbsp; Toxic F1: **0.75** &nbsp;|&nbsp; Macro F1: **0.86**

        The dataset is imbalanced (~9.6% toxic), so **F1-score** — not accuracy —
        was used to select the final model. See the accompanying technical
        report for full evaluation and error analysis.
        """
    )

st.divider()

user_input = st.text_area(
    "Enter a comment to check:",
    height=140,
    placeholder="Type or paste a comment here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Analyze", type="primary", use_container_width=True)

if predict_clicked:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing — input cannot be empty.")
    elif len(user_input.strip()) < 3:
        st.warning("⚠️ Input is too short to classify meaningfully. Please enter a longer comment.")
    else:
        cleaned = clean_text(user_input)

        if not cleaned:
            st.warning(
                "⚠️ After preprocessing, no meaningful words remained "
                "(e.g. input was only numbers, symbols, or stop words). Please try different text."
            )
        else:
            pred = model.predict([cleaned])[0]

            # Not all classifiers (e.g. LinearSVC) expose predict_proba
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned])[0]
            elif hasattr(model, "decision_function"):
                import numpy as np
                score = model.decision_function([cleaned])[0]
                proba_toxic = 1 / (1 + np.exp(-score))  # sigmoid approx of the margin
                proba = [1 - proba_toxic, proba_toxic]

            st.divider()
            if pred == 1:
                st.error("🚫 **Prediction: TOXIC**")
            else:
                st.success("✅ **Prediction: NON-TOXIC**")

            if proba is not None:
                st.write(f"Model confidence — Toxic: **{proba[1]*100:.1f}%**  |  Non-Toxic: **{proba[0]*100:.1f}%**")
                st.progress(float(proba[1]))

            with st.expander("🔍 See processed text (what the model actually saw)"):
                st.code(cleaned if cleaned else "(empty after preprocessing)")

st.divider()
st.caption(
    "Deployment pipeline: User Input → Preprocessing (clean_text) → "
    "TF-IDF Vectorizer → Logistic Regression → Prediction. "
    "Model and vectorizer are bundled as a single scikit-learn Pipeline (final_model.pkl)."
)
