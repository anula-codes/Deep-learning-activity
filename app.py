import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200  # must match training


@st.cache_resource
def load_assets():
    model = load_model("best_model.keras")
    with open("tokenizer.pkl", "rb") as f:
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
