import streamlit as st
import pickle
import re


# Page configuration
st.set_page_config(
    page_title="Personality Detector",
    page_icon="🧠",
    layout="centered"
)


# Load model and vectorizer
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


# UI
st.title("👩🏻‍🏫 Personality Detector")
st.caption(
    "Introvert vs Extrovert "
)

st.divider()

user_input = st.text_area(
    "Enter a few sentences about yourself or your thoughts:",
    height=150,
    placeholder="I prefer staying home and reading over going to parties..."
)

if st.button("Predict Personality"):

    if not user_input.strip():
        st.warning("Please enter some text first.")

    else:
        cleaned = clean_text(user_input)

        vec_text = vectorizer.transform([cleaned])

        pred = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0]

        label = "Introvert" if pred == 1 else "Extrovert"
        confidence = proba[pred] * 100

        if pred == 1:
            st.success(
                f"Prediction: **{label}** ({confidence:.1f}% confidence)"
            )
        else:
            st.info(
                f"Prediction: **{label}** ({confidence:.1f}% confidence)"
            )

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Extrovert Probability",
            f"{proba[0] * 100:.1f}%"
        )

        col2.metric(
            "Introvert Probability",
            f"{proba[1] * 100:.1f}%"
        )

st.divider()

st.caption(
    " Dataset: MBTI (Kaggle)"
)
