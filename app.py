import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import nltk
import string
import sqlite3
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Download NLTK data only once
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize SentenceTransformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Clean text ----------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])

# ---------- Highlight missing ----------
def highlight_missing_words(text, missing_keywords):
    highlighted = []
    for word in text.split():
        clean_word = word.strip(string.punctuation).lower()
        if clean_word in missing_keywords:
            highlighted.append(f"<span style='color:red; font-weight:bold'>{word}</span>")
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

# ---------- Database setup ----------
def create_connection():
    conn = sqlite3.connect('resume_matcher.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume TEXT,
            job_description TEXT,
            similarity_score REAL,
            keyword_match REAL,
            common_keywords TEXT,
            missing_keywords TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

conn, c = create_connection()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Resume Matcher (LLM)", layout="centered")
st.title("🤖 Resume vs Job Description Matcher (LLM-Powered)")
st.markdown("Compare your resume with a job description using semantic similarity and keyword overlap.")

# ---------- Inputs ----------
st.markdown("### Upload or Paste Your Resume")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text = ""

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        resume_text += page.get_text()
else:
    resume_text = st.text_area("Or paste your resume here", height=200)

st.markdown("### Paste Job Description")
job_description = st.text_area("Paste the job description here", height=200)

# ---------- Main Logic ----------
if st.button("Compare Now"):
    if not resume_text.strip() or not job_description.strip():
        st.warning("Please provide both resume and job description.")
    else:
        # Cleaned versions for keyword match
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_description)

        # Embedding-based similarity
        resume_embed = embed_model.encode(resume_text, convert_to_tensor=True)
        jd_embed = embed_model.encode(job_description, convert_to_tensor=True)
        similarity_score = float(util.cos_sim(resume_embed, jd_embed)[0][0]) * 100

        # Keyword overlap
        resume_tokens = cleaned_resume.split()
        jd_tokens = cleaned_jd.split()
        resume_counter = Counter(resume_tokens)
        jd_set = set(jd_tokens)

        common_keywords = {word for word in jd_set if word in resume_counter}
        missing_keywords = jd_set - common_keywords
        keyword_match = (len(common_keywords) / len(jd_set)) * 100 if jd_set else 0

        # ---------- Output ----------
        st.markdown("### 📊 Match Scores")
        col1, col2 = st.columns(2)
        col1.metric("LLM Similarity", f"{similarity_score:.2f} %")
        col2.metric("Keyword Match", f"{keyword_match:.2f} %")
        st.progress(int(similarity_score))

        if common_keywords:
            st.markdown("**✅ Common Keywords:** " + ", ".join(sorted(common_keywords)))
        if missing_keywords:
            st.markdown("**⚠️ Missing Keywords:** " + ", ".join(sorted(missing_keywords)))

        st.markdown("### 🧠 Resume with Highlighted Missing Keywords")
        highlighted = highlight_missing_words(resume_text, missing_keywords)
        st.markdown(highlighted, unsafe_allow_html=True)

        # ---------- Download CSV ----------
        common_list = list(common_keywords)
        missing_list = list(missing_keywords)
        max_len = max(len(common_list), len(missing_list))
        common_list += [""] * (max_len - len(common_list))
        missing_list += [""] * (max_len - len(missing_list))

        df = pd.DataFrame({
            "Common Keywords": common_list,
            "Missing Keywords": missing_list
        })

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name='comparison_result.csv', mime='text/csv')

        # ---------- Store in DB ----------
        c.execute('''
            INSERT INTO match_results (resume, job_description, similarity_score, keyword_match, common_keywords, missing_keywords)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            resume_text,
            job_description,
            similarity_score,
            keyword_match,
            ', '.join(sorted(common_keywords)),
            ', '.join(sorted(missing_keywords))
        ))
        conn.commit()

# ---------- History ----------
if st.checkbox("📜 Show Past Matches"):
    results = c.execute("SELECT id, similarity_score, keyword_match, common_keywords, missing_keywords FROM match_results ORDER BY id DESC LIMIT 5").fetchall()
    if results:
        df_history = pd.DataFrame(results, columns=["ID", "LLM %", "Keyword %", "Common", "Missing"])
        st.markdown("### 📂 Last 5 Comparison Records")
        st.dataframe(df_history)
    else:
        st.info("No past comparisons found.")








    

