from langchain_groq import ChatGroq
from parser.resume_schema import parser
from langchain_core.prompts import PromptTemplate
from tools.fill_resume import (render_tex, compile_tex)
import json
from dotenv import load_dotenv


def ResumeWriter(state):
    load_dotenv()
    resume = state['full_resume_text']
    desc = state['best_job_description']
    model = ChatGroq(model='llama-3.3-70b-versatile')
    prompt = PromptTemplate(
template="""You are an expert Career Coach and professional Resume Writer specializing in the tech industry. Your task is to transform a user's standard resume into a compelling, tailored document that is highly optimized for a specific job description and an Applicant Tracking System (ATS).

You must analyze the inputs and generate a new JSON object that follows these core directives:

### CORE DIRECTIVES

1.  **Analyze the Job Description First:**
* Begin by deeply analyzing the **Target Job Description**. Identify the top 5-7 key skills, technologies, and required qualifications (e.g., 'data analysis', 'machine learning', 'Python', 'Scikit-Learn', 'automated pipelines'). These keywords are your primary guide for tailoring the entire resume.

2.  **Rewrite the Objective Statement:**
* Transform the `objective` into a dynamic, 2-sentence professional summary.
* The first sentence should highlight the candidate's primary expertise (e.g., "Data Analyst with experience in...").
* The second sentence should state their objective, explicitly mentioning the target company and role, and incorporating 1-2 key skills from the job description.

3.  **Optimize the Skills Section:**
* Review the user's `skills` list.
* Prioritize and reorder the skills to feature the most relevant ones from the job description at the beginning of the list.
* If the job description mentions a "soft skill" (e.g., "analytical skills," "communication") that the user demonstrates in their experience, ensure it is included in the skills list.

4.  **Enhance Experience & Projects with the STAR Method:**
* This is the most critical step. For every bullet point in the `experience` and `projects` sections, rewrite it using the **STAR method (Situation, Task, Action, Result)**.
* **Action-Oriented Verbs:** Start every bullet point with a powerful action verb (e.g., Engineered, Architected, Streamlined, Optimized, Accelerated, Quantified). Avoid passive language.
* **Quantify Achievements:** Add metrics to show impact. Instead of "improved accuracy," write "achieved 92% accuracy." Instead of "reducing time," write "reducing preprocessing time by 50%." If the original resume lacks numbers, infer reasonable and professional estimates based on the project's context.
* **Seamless Keyword Integration:** Naturally weave the keywords from the job description into the bullet points. For example, if the job requires experience with "data augmentation," and the user has a relevant point, rewrite it to be: "Reduced model overfitting by implementing data augmentation techniques, increasing the training dataset size by 5x."

5.  **Maintain Professional Tone:**
* The final output must be professional, confident, and focused on results and impact.

6.  **Strictly Adhere to Schema:**
* Your final output must be **only the JSON object**, perfectly adhering to the provided schema. Do not add, remove, or change the keys in the JSON structure. Do not include any explanations or text outside of the JSON.

### --- INPUTS ---

**1. Original User Resume (JSON):**
{resume}

**2. Target Job Description:**
{description}

**3. Required JSON Output Schema:**
{schema}

### --- TAILORED RESUME (JSON OUTPUT) ---
""",
input_variables=['resume', 'description', 'schema']
)

    final_prompt = prompt.invoke({'resume': resume, 'description': desc, 'schema': parser.get_format_instructions()}) 

    tailor_resume = model.invoke(final_prompt).content
    if tailor_resume:
        json_start_tag = "```json"
        json_end_tag = "```"
        if isinstance(tailor_resume, str):
            if tailor_resume.startswith(json_start_tag) and tailor_resume.endswith(json_end_tag):
                tailor_resume = tailor_resume[len(json_start_tag): - len(json_end_tag)]
                json_content = json.loads(tailor_resume)
            else:
                json_content = json.load(tailor_resume)

            tex_content = render_tex(json_content)
            resume_path = compile_tex(tex_content)
    return {'tailored_resume_path': resume_path,
            'graph_executed' : True}
