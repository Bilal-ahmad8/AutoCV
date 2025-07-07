import asyncio, json
from tools.crawl_tool import extract_description
from tools.job_finder import job_search_tool
from tools.scorer import score_resume_against_job
from typing import TypedDict, Optional, List, Annotated
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.graph import END
from dotenv import load_dotenv

class AgentState(TypedDict):
    user_profile: dict
    job_queries: Annotated[str, add_messages]
    job_listings: List[dict]
    scored_jobs: List[dict]
    best_job: Optional[dict]
    best_job_description: Optional[str]
    current_query: Annotated[str, add_messages]




# Nodes

def generate_query(state: AgentState) -> AgentState:
    load_dotenv()
    model = ChatGroq(model='gemma2-9b-it')
    prompt = PromptTemplate(template="""You are helpful AI Agent thats take Input User profile base on that 
                            suggest job roles e.g., Junior Machine Learning Engineer, React Developer, DevOps Engineer, Data Scientist, Senior Content Writer ,etc. 
                            Output should only one string which ever perfectly matches user profile 
                            Profile: {context}
                            Previous query: {query} """, input_variables=['context', 'query'])
    prompt_final = prompt.invoke({'context': state["user_profile"], 'query': state["current_query"]})
    result = model.invoke(prompt_final)
    query = result.content
    print(query)
    return {
        **state,
        "current_query": query,
        "job_queries": state.get("job_queries", []) + [query],
    }

def query_state(state:AgentState) -> AgentState:
        return {**state,
                "current_query": "No jobs found from last query. Try different query."}

def retry_query(state:AgentState) -> str:
    return "score_jobs" if state['job_listings'] else 'retry_query'
    

def find_jobs(state: AgentState) -> AgentState:
    jobs = job_search_tool.run(state["current_query"][-1].content)
    if jobs is not None:
        return {**state, "job_listings": jobs}
    else:
        return {**state, "job_listings": [], "current_query": "No jobs found from last query. Try different query."}

async def score_jobs(state: AgentState) -> AgentState:
    scored = []
    for job in state["job_listings"]:
        desc = await extract_description.arun(job["url"])
        #Create the dictionary that matches the expected tool_input for score_resume_against_job
        if desc:
            json_start_tag = "```json"
            json_end_tag = "```"

            if desc.startswith(json_start_tag) and desc.endswith(json_end_tag):
                desc = desc[len(json_start_tag) :-len(json_end_tag)]

            if isinstance(desc, str):
                desc = json.loads(desc)

            tool_input_for_scorer = {
                "profile": state["user_profile"],
                "job": desc
            }
            score = await score_resume_against_job.arun(tool_input_for_scorer)
        scored.append({"job": job, "url": job["url"] ,"score": score, "description": desc})
    return {**state, "scored_jobs": scored}

def pick_best_job(state: AgentState) -> AgentState:
    load_dotenv()
    model = ChatGroq(model='llama-3.3-70b-versatile')
    prompt = PromptTemplate(template="""
        You are an expert job matching assistant.

        You are given a list of jobs that have already been scored based on how well they match a user's profile.
        Each job contains:
        - The job details (title, URL)
        - The job description
        - A detailed score breakdown including:
            - final_score (0.0 to 1.0)
            - semantic_similarity
            - skill_score
            - bonus_score
            - matched_skills
            - missing_skills
            - bonus_explanation

        Your goal is to select the **single best matching job**. Favor jobs with:
        - Higher `final_score`
        - Fewer `missing_skills`
        - High `semantic_similarity` and `skill_score`
        - Strong `bonus_score` and helpful `bonus_explanation`

        Think critically. If two jobs have similar scores, prefer the one with better alignment in required skills and education/experience match.

        Respond only with a JSON object in the following format:
        ```json
        {{
        "job": {{...}},
        "url": {{...}}
        }}
        Jobs to choose from:
        {context}""", input_variables=['context'])

    prompt_final = prompt.invoke({'context': state["scored_jobs"]})
    result = model.invoke(prompt_final)

    if result.content:
        json_start_tag = "```json"
        json_end_tag = "```"
        content = result.content
        if content.startswith(json_start_tag) and content.endswith(json_end_tag):
            json_content = content[len(json_start_tag) :-len(json_end_tag)]

            selected = json.loads(json_content)  
    uid = selected['job']['url']

    for entry in state["scored_jobs"]:
        if entry['job']['url'] == uid:
            description = entry['description']
            break
        else:
            description = None
    return {
        **state,
        "best_job": selected["job"],
        "best_job_description": description
    }

def should_retry(state: AgentState) -> str:
    if state["best_job"] and state["best_job"].get("title"):
        return END
    return "generate_query"

# --------------------
# Graph Construction
# --------------------
graph = StateGraph(AgentState)

graph.add_node("generate_query", generate_query)
graph.add_node("query_state", query_state)
graph.add_node("find_jobs", find_jobs)
graph.add_node("score_jobs", score_jobs)
graph.add_node("pick_best_job", pick_best_job)

graph.set_entry_point("generate_query")

graph.add_edge("generate_query", "find_jobs")
graph.add_conditional_edges("find_jobs", retry_query)
graph.add_edge("score_jobs", "pick_best_job")
graph.add_conditional_edges("pick_best_job", should_retry)

app = graph.compile()

# --------------------
# Run Example
# --------------------
async def JobLooker(state):
    result = await app.ainvoke(state)
    if result['best_job']:
        return result       


if __name__ == "__main__":
    initial_state = { "user_profile": {
    "Job Title": [
      "Frontend Developer",
      "React Developer",
      "Web Developer"
    ],
    "Experience Years": "1",
    "Education": [
      {
        "Degree": "Bachelor of Computer Applications",
        "University": "University of Mumbai",
        "Graduation Year": "2025"
      },
      {
        "Degree": "High School",
        "University": "St. Xavier's College Junior College",
        "Graduation Year": "2022"
      },
      {
        "Degree": "High School",
        "University": "St. Xavier's College Junior College",
        "Graduation Year": "2020"
      }
    ],
    "Responsibilities": [
      "Developing and maintaining web applications using React.js",
      "Creating responsive and user-friendly UI components",
      "Collaborating with backend developers for API integration",
      "Debugging and optimizing frontend performance",
      "Implementing reusable components and front-end libraries",
      "Maintaining code quality and documentation",
      "Building single-page applications (SPAs)",
      "Using version control tools like Git",
      "Participating in code reviews and agile ceremonies",
      "Writing unit and integration tests for frontend features",
      "Ensuring cross-browser compatibility",
      "Implementing form validation and handling edge cases",
      "Deploying frontend apps to staging and production environments",
      "Working with RESTful APIs and JSON data structures",
      "Refactoring legacy code for better performance and readability"
    ],
    "Skills": [
      "React.js",
      "JavaScript (ES6+)",
      "HTML5",
      "CSS3",
      "Tailwind CSS",
      "Bootstrap",
      "Redux",
      "React Router",
      "Next.js",
      "TypeScript",
      "Git/GitHub",
      "RESTful APIs",
      "JSON",
      "Node.js (Basic)",
      "NPM/Yarn",
      "Webpack",
      "Babel",
      "Responsive Design",
      "Cross-browser Compatibility",
      "Axios",
      "Jest",
      "React Testing Library",
      "CI/CD (Basic)",
      "Figma to Code Conversion",
      "UI/UX Design Principles",
      "State Management",
      "Component-based Architecture",
      "Single Page Applications (SPA)",
      "Hooks (useState, useEffect, etc.)",
      "Linting and Formatting Tools (ESLint, Prettier)",
      "Firebase (Basic Auth and Hosting)",
      "Performance Optimization",
      "Form Handling (Formik, Yup)",
      "Agile Development",
      "Scrum",
      "Debugging Tools (Chrome DevTools)",
      "Local Storage & Session Storage",
      "DOM Manipulation",
      "Basic SEO Principles for SPAs",
      "Error Boundary Handling in React",
      "Code Splitting and Lazy Loading",
      "Accessibility (ARIA roles, semantic HTML)",
      "Mobile-first Design",
      "Version Control",
      "Basic Linux Commands",
      "Basic AWS (S3, Amplify)"
    ]},
        "job_queries": [],
        "job_listings": [],
        "scored_jobs": [],
        "best_job": None,
        "best_job_description": None,
        "current_query": " ", }
    result = asyncio.run(JobLooker(initial_state))
    print("Best Job Match:", result["best_job"])
    print("\n\n")
    print("Job Description:", result["best_job_description"])
    print("\n\n")
    print('score:', result['score'])
    print('\n\n')
    print("url:", result['url'])
    print('\n\n')
    print(result)

