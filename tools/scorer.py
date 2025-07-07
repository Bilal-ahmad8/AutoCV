from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from langchain.tools import tool
import json

# Load once globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_semantic_similarity(text1, text2):
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def score_skills(user_skills, job_skills):
    user_skills_lower = set([s.lower() for s in user_skills])
    job_skills_lower = set([s.lower() for s in job_skills])

    exact_matches = user_skills_lower & job_skills_lower
    missing_skills = job_skills_lower - user_skills_lower

    score = ( len(exact_matches) * 2 + len(missing_skills) * -1 )
    max_score = len(job_skills_lower) * 2  # All match = full score

    normalized_score = max(score / max_score, 0)
    return normalized_score, list(exact_matches), list(missing_skills)


def education_fields_match(profile_education, job_requirements, threshold=0.6):
    """
    Returns True if any degree field in the user's education matches 
    the required field in the job requirement based on fuzzy similarity.
    """
    # Step 1: Extract user's fields of study
    user_fields = []
    for entry in profile_education:
        degree = entry.get("Degree", "")
        if degree:
            user_fields.append(degree.lower())

    # Step 2: Infer required field from job requirements
    required_field = ""
    for req in job_requirements:
        if "degree" in req.lower():
            required_field = req.lower()
            break

    if not required_field or not user_fields:
        return False  # Can't determine match if data is missing

    # Step 3: Fuzzy compare
    for field in user_fields:
        ratio = SequenceMatcher(None, field, required_field).ratio()
        if ratio >= threshold:
            return True

    return False


def score_bonus(profile, job):
    bonus = 0
    explanation = []

    # Experience match
    profile_exp = float(profile.get("Experience Years", 0))
    required_exp = 0
    for req in job.get("requirement", []):
        if "year" in req.lower():
            match = [int(s) for s in req.split() if s.isdigit()]
            if match:
                required_exp = match[0]
                break
    if profile_exp >= required_exp:
        bonus += 0.5
        explanation.append("✅ Experience requirement met")
    else:
        explanation.append("❌ Experience below required")

    # Education match 
    profile_edu = profile.get("Education", [])
    job_reqs = job.get("requirement", [])
    if education_fields_match(profile_edu, job_reqs):
        bonus += 0.5
        explanation.append("✅ Education field matches job requirement")
    elif job_reqs:
        explanation.append("⚠️ Education field may not match the requirement")
    else:
        bonus += 0.5
        explanation.append("✅ No strict education requirement found")

    return bonus, explanation

def build_embedding_text(profile):
    summary = profile.get("Summary", "")
    responsibilities = "\n".join(profile.get("Responsibilities", []))
    return summary + "\n" + responsibilities

@tool
def score_resume_against_job(profile, job):
    """
    Evaluate how well a user's profile matches a given job description.

    This function performs a multi-factor scoring of the resume against the job, combining:
    1. Semantic similarity between profile summary/responsibilities and job description/responsibilities.
    2. Skill match score based on direct skill overlap and missing skills.
    3. Bonus points for experience and education alignment.

    Parameters:
    ----------
    profile : dict
        A structured user profile containing:
        - "Summary": str
        - "Skills": List[str]
        - "Responsibilities": List[str]
        - "Experience Years": float
        - "Education": List[Dict[str, str]]

    job : dict
        A structured job description containing:
        - "job_description": str
        - "skill_required": List[str]
        - "requirement": List[str]
        - "responsibilities": List[str]

    Returns:
    -------
    dict
        {
            "final_score": float,                 # Weighted score between 0.0–1.0
            "semantic_similarity": float,        # Cosine similarity between profile and job content
            "skill_score": float,                # Normalized skill match score
            "bonus_score": float,                # Bonus based on experience and education
            "matched_skills": List[str],         # List of overlapping skills
            "missing_skills": List[str],         # List of required skills not in user profile
            "bonus_explanation": List[str]       # Human-readable explanation of bonus criteria
        }
    """
    # Semantic similarity
    #job = json.load(job)
    job_text = job.get("job_description", "") + "\n" + "\n".join(job.get("responsibilities", []))
    user_text = build_embedding_text(profile)
    semantic_score = compute_semantic_similarity(user_text, job_text)

    # Skills
    skill_score, matched_skills, missing_skills = score_skills(
        profile.get("Skills", []),
        job.get("skill_required", [])
    )

    # Bonus
    bonus_score, bonus_explanation = score_bonus(profile, job)

    # Final weighted score
    final_score = (
        0.4 * semantic_score +
        0.4 * skill_score +
        0.2 * bonus_score
        )

    result = {
        "final_score": round(final_score, 3),
        "semantic_similarity": round(semantic_score, 3),
        "skill_score": round(skill_score, 3),
        "bonus_score": round(bonus_score, 3),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "bonus_explanation": bonus_explanation }
    
    return result

profile = """{
  "Job Title": [
    "Data Scientist",
    "Machine Learning Engineer",
    "Data Analyst"
  ],
  "Experience Years": "1",
  "Education": [
    {
      "Degree": "Bachelor of Commerce",
      "University": "University of Lucknow",
      "Graduation Year": "2025"
    },
    {
      "Degree": "High School",
      "University": "Christ Church College",
      "Graduation Year": "2022"
    },
    {
      "Degree": "High School",
      "University": "Christ Church College",
      "Graduation Year": "2020"
    }
  ],
  "Responsibilities": [
    "Data Analysis and Interpretation",
    "Machine Learning Model Development and Deployment",
    "Data Preprocessing and Cleaning",
    "Model Optimization and Evaluation",
    "Feature Engineering",
    "Pipeline Automation",
    "Data Collection and Validation",
    "Workflow Streamlining",
    "Client Onboarding Support",
    "Building and improving AI-powered systems",
    "Developing and deploying user-friendly applications",
    "Implementing Machine Learning algorithms",
    "Training and optimizing models",
    "Data augmentation and handling missing data",
    "Implementing RAG systems"
  ],
  "Skills": [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "RAG",
    "AI Agents",
    "PyTorch",
    "TensorFlow",
    "Keras",
    "Scikit-Learn",
    "XGBoost",
    "LGBoost",
    "Optuna",
    "MLOps",
    "CI/CD Pipelines",
    "MLflow",
    "DVC",
    "Git/GitHub",
    "DagsHub",
    "AWS",
    "Apache Airflow",
    "SQL",
    "Docker",
    "Data Analysis",
    "Data Preprocessing",
    "Model Training",
    "Model Evaluation",
    "Feature Engineering",
    "Data Visualization",
    "Time Series Analysis",
    "Natural Language Processing (NLP)",
    "Computer Vision",
    "Text Preprocessing",
    "Bidirectional LSTM",
    "Word2Vec",
    "CNN",
    "Transfer Learning",
    "DenseNet121",
    "ROC Curve Analysis",
    "Imputation",
    "Classification",
    "Regression",
    "Clustering",
    "Supervised Learning",
    "Unsupervised Learning",
    "Generative AI",
    "Data Collection",
    "Data Validation",
    "Workflow Automation",
    "User-friendly form design",
    "Healthcare Data Analysis",
    "SVM",
    "Random Forest",
    "Patient Outcome Prediction"
  ]
}"""

job = """{
  "job_description": "Weekday AI is seeking an experienced Fullstack Engineer to develop scalable web applications and microservices.  The ideal candidate will have expertise in React (frontend) and FastAPI/Python (backend), along with experience working with cloud platforms and infrastructure tools.",
  "skill_required": [
    "React",
    "FastAPI",
    "Python",
    "PostgreSQL",
    "Google Cloud Platform (GCP)",
    "Docker",
    "Kubernetes",
    "Microservices architecture",
    "API design",
    "Git"
  ],
  "requirement": [
    "3 to 10+ years of experience as a Full Stack Developer",
    "Bachelor’s or Master’s degree in Computer Science or a related field"
  ],
  "responsibilities": [
    "Design, develop, and deploy microservices using FastAPI/Python, primarily on Google Cloud Platform (GCP)",
    "Build responsive, performant frontend applications using React and integrate them seamlessly with backend services",
    "Design and optimize relational databases using PostgreSQL",
    "Implement and manage CI/CD pipelines using GCP-native tools and best practices",
    "Collaborate with cross-functional teams to deliver features and enhancements",
    "Ensure high performance, scalability, and maintainability of applications across the stack",
    "Design, document, and implement RESTful APIs",
    "Conduct code reviews, enforce best practices, and mentor junior developers where necessary",
    "Stay current with emerging technologies and recommend improvements to enhance development workflows"
  ]
}"""

import time
if __name__ == '__main__':
    start = time.time()
    profile = json.loads(profile)
    job = json.loads(job)
    details = score_resume_against_job(profile=profile, job=job)
    print(details)
    end = time.time()
    print(end-start)
    for key in details.keys():
        print(details[key])
