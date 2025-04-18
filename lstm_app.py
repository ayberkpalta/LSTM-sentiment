import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache_resource
def load_sentiment_model():
    model = load_model("LSTM_Model_new.keras")
    tokenizer = joblib.load("lstm_tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_sentiment_model()

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]

    if prediction >= 0.55:
        return "🟢 Positive", float(prediction)
    elif prediction <= 0.45:
        return "🔴 Negative", float(prediction)
    else:
        return "🟡 Neutral", float(prediction)


st.title("💬 LSTM-Based Sentiment Analysis")
st.write("Enter a comment, and we will analyze the sentiment!")

user_input = st.text_area("Enter your comment here", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.subheader("Result")
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Score: `{score:.3f}`")
