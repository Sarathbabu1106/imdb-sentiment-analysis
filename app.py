import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")
st.write("A Logistic Regression model trained on 50,000 IMDB reviews with 90.6% accuracy.")

# Input box
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if user_input.strip() != "":
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        sentiment = "Positive ✅" if prediction == 1 else "Negative ❌"
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before predicting.")
