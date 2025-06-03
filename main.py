import streamlit as st

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {
    value: key for (key, value) in word_index.items()
}

model = load_model('imdb_rnn_model.h5')


def decode_review(text):
    """Decode the review text from integers to words."""
    return ' '.join(
        reverse_word_index.get(i - 3, '?') for i in text
    )

def preprocess_review(review):
    """Preprocess the review text for prediction."""
    # Convert the review to integers
    encoded_rev = [word_index.get(word, 2) + 3 for word in review.lower().split()]
    # Pad the sequence to ensure it has the same length as the training data
    padded = sequence.pad_sequences([encoded_rev], maxlen=500)
    return padded

def predict_review(review):
    predictInput = preprocess_review(review)
    prediction = model.predict(predictInput)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]


st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to predict its sentiment (positive or negative):")
review_input = st.text_area("Review Text", height=200)
if st.button('Predict'):
    if review_input:
        sentiment, score = predict_review(review_input)
        st.write(f"Sentiment: {sentiment} (Score: {score:.2f})")
    else:
        st.write("Please enter a review to analyze.")
else:
    st.write("Click the button to predict the sentiment of the review.")
# Display instructions
