import streamlit as st
import joblib
import zipfile
import os

# Unzip the model files if they are not already unzipped
if not os.path.exists('random_forest_tfidf_tuned_model.pkl') or not os.path.exists('tfidf_tokenizer.pkl'):
    with zipfile.ZipFile('random_forest_tfidf_tuned_model.zip', 'r') as zip_ref:
        zip_ref.extractall()  # Extract all files in the current directory

# Load the Random Forest model and TF-IDF tokenizer
rf_model = joblib.load('random_forest_tfidf_tuned_model.pkl')
tfidf_tokenizer = joblib.load('tfidf_tokenizer.pkl')

# Load the saved test accuracy
test_accuracy_rf_tfidf_refined = joblib.load('test_accuracy_rf_tfidf_refined.pkl')

# Streamlit UI
st.title('Hate Speech Detection')

# Section: Analyze Sentiment of Text and Detect Hate Speech
st.subheader('Analyze Text for Hate Speech and Sentiment')
input_text = st.text_area('Enter text to analyze:', '')

# Define a function to detect hate speech, sentiment, and confidence scores
def detect_hate_speech_and_sentiment(text):
    # Preprocess the text using the TF-IDF tokenizer
    text_tfidf = tfidf_tokenizer.transform([text])

    # Make prediction using the Random Forest model
    prediction = rf_model.predict(text_tfidf)[0]
    
    # Get probability estimates (confidence scores) for each class
    probabilities = rf_model.predict_proba(text_tfidf)[0]  # Get probabilities for the input
    
    prob_sum = sum(probabilities)

    # If probabilities do not sum to 1, normalize them
    if prob_sum != 1:
        normalized_probabilities = probabilities / prob_sum
    else:
        normalized_probabilities = probabilities

    # Map the label to sentiment (you can adjust this mapping as per your specific use case)
    sentiments = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    sentiment = sentiments[prediction]

    # Use model to detect if it's hate speech or not (assuming binary classification for hate speech)
    hate_speech_prediction = 'Hate Speech' if prediction == 2 else 'Not Hate Speech'

    return hate_speech_prediction, sentiment, normalized_probabilities

if st.button('Analyze'):
    if input_text:
        # Get predictions and confidence scores
        hate_speech, sentiment, probabilities = detect_hate_speech_and_sentiment(input_text)

        # Display sentiment result first
        st.write(f'Predicted Sentiment: **{sentiment}**')

        # Display hate speech detection result
        st.write(f'Hate Speech Detection: **{hate_speech}**')

        # Display confidence scores with their corresponding labels
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        confidence_scores = {label: f'{prob * 100:.2f}%' for label, prob in zip(sentiment_labels, probabilities)}

        st.write('Confidence Scores (as percentages):')
        for label, score in confidence_scores.items():
            st.write(f'{label}: {score}')
        
        # Display the test accuracy of the model
        st.write(f"Test Accuracy of the model: **{test_accuracy_rf_tfidf_refined:.4f}**")
        
        # Feedback section
        correct = st.radio("Was this prediction correct?", ("Yes", "No"))
        if correct == "Yes":
            st.write("Great! Thanks for your feedback.")
            # You can log the correct prediction in a feedback system or database here
        else:
            st.write("Thanks for the feedback! We'll use it to improve.")
            # You can log the incorrect prediction and actual label here

    else:
        st.warning('Please enter text for analysis.')
