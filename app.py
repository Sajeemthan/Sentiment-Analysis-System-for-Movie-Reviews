import pandas as pd 
from bs4 import BeautifulSoup
import pickle as pk
import re, string, unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load the saved components
model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))
lb = pk.load(open('label_binarizer.pkl', 'rb'))

review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    # Preprocess the input review
    def preprocess_review(text):
        # Apply the same preprocessing steps used during training
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'\[[^]]*\]', '', text)
        text = re.sub(r'[^a-zA-z0-9\s]', '', text)
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        text = ' '.join(filtered_tokens)
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text
    
    cleaned_review = preprocess_review(review)
    review_vector = vectorizer.transform([cleaned_review])
    
    # Make prediction
    result = model.predict(review_vector)
    
    # Convert numerical prediction back to label
    sentiment = lb.inverse_transform(result)[0]
    
    if sentiment == 'negative':
        st.write('Negative Review')
    else:
        st.write('Positive Review')