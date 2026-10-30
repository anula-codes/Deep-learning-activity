# Movie Review Sentiment Analysis

Classifies movie reviews as **positive** or **negative** using deep learning. Three models (LSTM, stacked BiLSTM and 1D CNN) were trained and compared, and the best one is served through a Streamlit web app.

**Live app:** https://deep-learning-activity-cpkfarvuegwzcja7ml5dco.streamlit.app

<img width="568" height="305" alt="Screenshot 2026-10-30 145247" src="https://github.com/user-attachments/assets/bf6ea6f1-52a8-40d3-a313-8ae5a63aa15a" />
<img width="569" height="321" alt="Screenshot 2026-10-30 145355" src="https://github.com/user-attachments/assets/a2312eaa-6ce8-4c6b-b0b4-625fe1735248" />



## Dataset

- 50,000 movie reviews: 25,000 positive and 25,000 negative
- Split: 40,000 for training, 10,000 for testing (80/20); 20% of the training data is used for validation
- Text preprocessing: Keras `Tokenizer` with a 5,000-word vocabulary, reviews padded to 200 tokens

## Models

| Model | Architecture | Epochs | Batch size | Test accuracy |
| --- | --- | --- | --- | --- |
| 1: LSTM | Embedding(128), LSTM(128), Dense(1) | 5 | 64 | about 0.86 |
| 2: Stacked BiLSTM | Embedding(64), BiLSTM(64), BiLSTM(32), Dense(32), Dense(1), dropout 0.3 | 8 | 64 | about 0.86 |
| 3: 1D CNN | Embedding(128), Conv1D(128), Conv1D(64), Dense(64), Dense(32), Dense(1), dropout 0.5 | 10 | 128 | about 0.87 |

All models use the Adam optimizer and binary cross-entropy loss. The 1D CNN scored highest on the test set and is the model used in the app. It overfits mildly (training accuracy about 0.99 against validation about 0.87), which early stopping would reduce.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app |
| `best_model.keras` | Trained 1D CNN model |
| `tokenizer.pkl` | Fitted tokenizer used to prepare input text |
| `requirements.txt` | Python dependencies |

## Tech stack

Python, TensorFlow / Keras, NumPy, Streamlit. Hosted on Streamlit Community Cloud.
