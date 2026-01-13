import streamlit as st
from utils import extract_text_from_pdf, extract_skills
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Skill Matcher", layout="wide")

st.title("📄 AI-Based Resume Skill Matcher")
st.markdown(
    "Upload resumes and paste a job description to check skill matches and similarity scores."
)

# Upload Multiple Resumes
resume_files = st.file_uploader(
    "Upload Resume(s) (PDF)", type=["pdf"], accept_multiple_files=True
)

# Job Description
job_description = st.text_area("Paste Job Description", height=200)

# Button to trigger matching
if st.button("✅ Match Resume"):
    if resume_files and job_description:
        job_description_lower = job_description.lower()
        job_skills = extract_skills(job_description_lower)

        st.subheader("💼 Job Skills Required")
        st.write(job_skills)

        results = []

        for resume_file in resume_files:
            resume_text = extract_text_from_pdf(resume_file)
            resume_skills = extract_skills(resume_text)
            matched_skills = set(resume_skills).intersection(set(job_skills))
            missing_skills = set(job_skills).difference(set(resume_skills))

            # Skill Match %
            match_percentage = (
                len(matched_skills) / len(job_skills) * 100 if job_skills else 0
            )

            # Semantic similarity using TF-IDF
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_text, job_description])
            similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

            results.append(
                {
                    "resume_name": resume_file.name,
                    "resume_skills": resume_skills,
                    "matched_skills": list(matched_skills),
                    "missing_skills": list(missing_skills),
                    "match_percentage": match_percentage,
                    "similarity_score": similarity_score,
                }
            )

        # Display results
        for res in results:
            st.markdown(f"### 📄 {res['resume_name']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Resume Skills:**", res["resume_skills"])
                st.write("**Matched Skills:**", res["matched_skills"])
            with col2:
                st.write("**Missing Skills:**", res["missing_skills"])
                st.write(f"**Skill Match %:** {res['match_percentage']:.2f}%")
                st.progress(int(res["match_percentage"]))
                st.write(f"**Semantic Similarity Score:** {res['similarity_score']:.2f}")
    else:
        st.warning("⚠️ Please upload at least one resume and provide a job description.")