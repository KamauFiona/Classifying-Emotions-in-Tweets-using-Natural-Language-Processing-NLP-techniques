import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    return text.lower()

def predict_emotion(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    return prediction[0]  # Directly return the string label (e.g., "Happy", "Sad")

# Streamlit UI
st.title("Tweet Emotion Classifier")
tweet = st.text_area("Enter a tweet:")

if st.button("Predict Emotion"):
    result = predict_emotion(tweet)
    st.success(f"The predicted emotion is: **{result}**")
