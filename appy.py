import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# ---------------- LOAD FILES ---------------- #

nltk.download('stopwords')

model = pickle.load(open('resume_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))

# ---------------- SKILLS DATABASE ---------------- #

skills_database = [
    "python",
    "java",
    "sql",
    "machine learning",
    "html",
    "css",
    "javascript",
    "react",
    "mongodb",
    "flask",
    "data science",
    "deep learning",
    "power bi",
    "tableau",
    "django",
    "c++",
    "aws"
]

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 45px;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 18px;
    margin-bottom: 30px;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #4CAF50;
    background-color: #262730;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.skill-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #1E1E1E;
    margin: 5px;
    display: inline-block;
    color: #4CAF50;
    border: 1px solid #4CAF50;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E1E1E;
    border-left: 8px solid #4CAF50;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

def clean_resume(text):

    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^A-Za-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.lower()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in skills_database:

        if skill in text:
            found_skills.append(skill)

    return found_skills

# ---------------- HEADER ---------------- #

st.markdown(
    '<p class="title">📄 AI Resume Screening System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning + NLP Based ATS Resume Analyzer</p>',
    unsafe_allow_html=True
)

# ---------------- MAIN LAYOUT ---------------- #

col1, col2 = st.columns([2, 1])

with col1:

    resume_text = st.text_area(
        "Paste Your Resume Here",
        height=350,
        placeholder="Paste complete resume text..."
    )

    analyze = st.button("Analyze Resume")

with col2:

    st.info("""
    ### Features
    
    ✅ Resume Classification  
    ✅ Skill Extraction  
    ✅ ATS Screening  
    ✅ NLP Processing  
    ✅ ML Prediction  
    """)

# ---------------- ANALYSIS ---------------- #

if analyze:

    if resume_text.strip() == "":
        st.warning("Please paste resume text.")
    else:

        cleaned = clean_resume(resume_text)

        vector = tfidf.transform([cleaned]).toarray()

        prediction = model.predict(vector)

        skills = extract_skills(resume_text)

        ats_score = min(len(skills) * 12, 100)

        # ---------- Prediction ---------- #

        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Job Role</h2>
            <h1 style="color:#4CAF50;">
                {prediction[0]}
            </h1>
        </div>
        """, unsafe_allow_html=True)

        # ---------- ATS Score ---------- #

        st.subheader("📊 ATS Resume Score")

        st.progress(ats_score)

        st.success(f"ATS Score: {ats_score}%")

        # ---------- Skills ---------- #

        st.subheader("🛠 Skills Detected")

        if skills:

            for skill in skills:
                st.markdown(
                    f'<div class="skill-box">{skill}</div>',
                    unsafe_allow_html=True
                )

        else:
            st.error("No skills detected.")

        # ---------- Feedback ---------- #

        st.subheader("💡 HR Screening Feedback")

        if ats_score >= 80:
            st.success("Excellent resume for ATS screening.")

        elif ats_score >= 60:
            st.warning("Good resume but can be improved with more relevant skills.")

        else:
            st.error("Resume needs improvement and more technical skills.")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption("Developed using Streamlit, Machine Learning, and NLP")
