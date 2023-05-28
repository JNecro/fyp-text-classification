import streamlit as st
import joblib
import re
import nltk
import os
from lime import lime_text
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir(r'C:\Users\User\Desktop\yo\DEGREE\FYP\Datasets')

# Load the saved model and preprocessing objects
model = joblib.load('textclassification.joblib')

# Function to preprocess the text
def preprocess_text(text):
    # Clean the text (convert to lowercase and remove punctuations and special characters)
    text = re.sub(r'[^\w\s]', '', text.lower().strip())

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    tokens = [word for word in tokens if word not in stopwords]

    # Join the tokens back to form preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Function to explain predictions using LIME
def explain_prediction(text):
    explainer = lime_text.LimeTextExplainer(class_names=model.classes_)
    explained = explainer.explain_instance(text, model.predict_proba, num_features=3)
    return explained

# Streamlit application
def main():
    # Set the page title
    st.title('Publication Classification')

    # Input field for the user to enter the test sample
    test_sample = st.text_area('Enter the test sample:', '')

    if st.button('Classify'):
        # Preprocess the text
        preprocessed_text = preprocess_text(test_sample)

        # Vectorize the preprocessed text
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Make predictions
        predicted_class = model.predict(vectorized_text)[0]
        predicted_probabilities = model.predict_proba(vectorized_text)[0]

        # Display the classification results
        st.write('Predicted Class:', predicted_class)
        st.write('Predicted Probabilities:', predicted_probabilities)

        # Explain the prediction using LIME
        explanation = explain_prediction(test_sample)
        st.write('Explanation:')
        st.write(explanation.as_list())

# Run the Streamlit application
if __name__ == '__main__':
    main()
