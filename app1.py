import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Check if the model files exist
vectorizer_path = 'vectorizer.pkl'
model_path = 'model.pkl'

if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
    st.error("Model files not found. Please make sure 'vectorizer.pkl' and 'model.pkl' exist.")
else:
    # Load the vectorizer
    tfidf = pickle.load(open(vectorizer_path, 'rb'))

    # Load or train the model
    if os.path.exists(model_path):
        # If the model file exists, load it
        model = pickle.load(open(model_path, 'rb'))
    else:
        # If the model file doesn't exist, train the model
        st.warning("Model not found. Training the model...")

        # Assuming you have your training data in X_train, y_train
        # Preprocess your text data
        X_train_preprocessed = [transform_text(text) for text in X_train]

        # Create a TfidfVectorizer and fit-transform your preprocessed data
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_preprocessed)

        # Train a Multinomial Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Save the vectorizer and the model
        with open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(tfidf_vectorizer, vectorizer_file)

        with open('model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)

    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"An error occurred: {e}")
