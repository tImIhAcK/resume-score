import pandas as pd
from pypdf import PdfReader
import streamlit as st
import docx
import spacy
from dataclasses import dataclass
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Dict

# Load the preprocessed resume data
data = pd.read_pickle("cleaned_resume_data.pkl")

nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"
ruler = nlp.add_pipe('entity_ruler')
ruler.from_disk(skill_pattern_path)

patterns = data.Category.unique()
for a in patterns:
    ruler.add_patterns([{"label": "Job-Category", "pattern": a}])

@dataclass
class SkillExtractor:
    nlp: spacy.Language
    stopwords: List[str]
    lemmatizer: WordNetLemmatizer

    def extract_skills(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        doc = self.nlp(text)
        skills = []
        skill_counts = {}

        for ent in doc.ents:
            if ent.label_ == "SKILL":
                skill = ent.text.lower()
                skills.append(skill)
                if skill in skill_counts:
                    skill_counts[skill] += 1
                else:
                    skill_counts[skill] = 1

        return skills, skill_counts

    def preprocess_text(self, text: str) -> str:
        text = re.sub(
            r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\\w+:\/\/\\S+)|^rt|http.+?"',
            " ",
            text,
        )
        text = text.lower()
        text = text.split()
        text = [
            self.lemmatizer.lemmatize(word)
            for word in text
            if not word in self.stopwords
        ]
        text = " ".join(text)
        return text
    
# Initialize the SkillExtractor
stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
extractor = SkillExtractor(nlp, stopwords, lemmatizer)


# ====READING FILES====
def read_pdf(file):
    reader = PdfReader(file)
    number_of_pages = len(reader.pages)
    text = ""
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# ==== APP BUILD ====
st.title('Resume Score Checker')
st.subheader("Check the score of your resume against job description")

job_desc = st.text_area(
    "Job description",
    placeholder="Enter the job description",
    max_chars=5000
)

resume = st.file_uploader("Upload resume", type=['pdf', 'doc', 'docx'])

if resume is not None:
    file_extension = resume.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = read_pdf(resume)
    elif file_extension in ['doc', 'docx']:
        text = read_docx(resume)
    else:
        st.error("Unsupported file type.")
        text = None

    if text:
        resume_skills, _ = extractor.extract_skills(text.lower())  # Correctly handle the tuple returned
        resume_skills = set(resume_skills)
        
if job_desc:
    job_skills, _ = extractor.extract_skills(job_desc.lower())  # Correctly handle the tuple returned
    job_skills = set(job_skills)
    score = 0
    for x in job_skills:
        if x in resume_skills:
            score += 1

    if job_skills:  # Avoid division by zero
        match = round(score / len(job_skills) * 100, 1)
        st.write(f"Your current resume is {match}% matched to the job requirements")
    else:
        st.write("No skills were found in the job description.")
        
