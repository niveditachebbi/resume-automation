import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from textblob import TextBlob
import string

# ---------- Download NLTK resources ----------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- Helper functions ----------
def clean_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def get_frequency(tokens):
    """Return word frequency"""
    return Counter(tokens)

def get_sentiment(text):
    """Return polarity and sentiment"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# ---------- Streamlit UI ----------
st.title("📄 Multi-PDF Analyzer with NLP & Sentiment")
st.markdown("Upload multiple PDF files, tokenize, lemmatize, analyze frequency, and perform sentiment analysis.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### File: {uploaded_file.name}")
        
        # Read PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Process text
        tokens = clean_text(text)
        freq = get_frequency(tokens)
        sentiment = get_sentiment(text)
        
        # Display results
        st.markdown("**Top 10 Frequent Words:**")
        st.write(freq.most_common(10))
        
        st.markdown("**Sentiment Analysis:**")
        st.write(sentiment)
        
        st.markdown("---")








    

