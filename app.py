import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
st.title("📄 Multi-PDF Analyzer with Job Description & TF-IDF Match")
st.markdown("Upload multiple PDF files and input a job description to analyze sentiment, tokens, and TF-IDF similarity.")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if uploaded_files and job_description:
    # Read all resume texts first
    resume_texts = []
    file_names = []
    for uploaded_file in uploaded_files:
        file_names.append(uploaded_file.name)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        resume_texts.append(text)
    
    # Combine job description and resumes for TF-IDF
    all_docs = resume_texts + [job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # TF-IDF similarity (cosine similarity) between each resume and job description
    jd_vector = tfidf_matrix[-1]  # Last vector is job description
    resume_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(resume_vectors, jd_vector)
    
    # Prepare results
    results = []
    jd_tokens = clean_text(job_description)
    
    for i, text in enumerate(resume_texts):
        tokens = clean_text(text)
        sentiment = get_sentiment(text)
        matched_words = set(tokens).intersection(set(jd_tokens))
        match_percent = round(float(similarities[i][0]) * 100, 2)
        
        results.append({
            "File Name": file_names[i],
            "Num Tokens": len(tokens),
            "Sentiment": sentiment,
            "Matching Words": ", ".join(list(matched_words)[:10]),  # limit to 10 words
            "TF-IDF Match %": match_percent
        })
    
    # Display results as a table
    st.markdown("### Analysis Summary for All PDFs")
    df = pd.DataFrame(results).sort_values(by="TF-IDF Match %", ascending=False)
    st.dataframe(df)










    

