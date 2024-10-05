import streamlit as st
import joblib

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

# Define a function to detect hate speech and sentiment
def detect_hate_speech_and_sentiment(text):
    # Preprocess the text using the TF-IDF tokenizer
    text_tfidf = tfidf_tokenizer.transform([text])

    # Make prediction using the Random Forest model
    prediction = rf_model.predict(text_tfidf)[0]
    
    # Map the label to sentiment (you can adjust this mapping as per your specific use case)
    sentiments = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    sentiment = sentiments[prediction]
    
    # Use model to detect if it's hate speech or not (assuming binary classification for hate speech)
    hate_speech_prediction = 'Hate Speech' if prediction == 2 else 'Not Hate Speech'
   

    return sentiment, hate_speech_prediction

if st.button('Analyze'):
    if input_text:
        # Get predictions
        hate_speech, sentiment = detect_hate_speech_and_sentiment(input_text)

        # Display results
        st.write(f'Hate Speech Detection: **{hate_speech}**')
        st.write(f'Predicted Sentiment: **{sentiment}**')
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

# Section: Display Model Performance Metrics
st.subheader('Model Performance Metrics')

# Display the test accuracy that was saved from training
st.write(f"Test Accuracy of the predicted output: {test_accuracy_rf_tfidf_refined:.4f}")
