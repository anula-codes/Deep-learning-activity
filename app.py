import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200  # must match training
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "best_model.keras")
TOKENIZER_PATH = os.path.join(BASE, "tokenizer.pkl")

# Show what the server can actually see if a file is missing
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    files = [f"{f} ({os.path.getsize(os.path.join(BASE, f))} bytes)"
             for f in sorted(os.listdir(BASE)) if os.path.isfile(os.path.join(BASE, f))]
    st.error("Model or tokenizer file not found. Files in the app folder: " + ", ".join(files))
    st.stop()


@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_assets()

st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review and the model will predict if it is positive or negative.")

review = st.text_area("Your review", height=150)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        p = float(model.predict(padded, verbose=0)[0][0])

        if p > 0.5:
            st.success(f"Positive ({p:.1%} confidence)")
        else:
            st.error(f"Negative ({1 - p:.1%} confidence)")
        st.progress(p)
