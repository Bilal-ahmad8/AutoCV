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
    best_job_score : Optional[dict]
    current_query: str
    previous_query_result: Annotated[str, add_messages]
    retry_count : int




# Nodes

def generate_query(state: AgentState) -> AgentState:
    load_dotenv()
    model = ChatGroq(model='gemma2-9b-it')
    prompt = PromptTemplate(template="""You are a helpful AI Agent suggesting job roles based on a user profile. Your goal is to find a job title that will yield good search results.

User Profile: {context}

You have already tried the following job queries: **{tried_queries}**

The last query result was: {result}

Based on this, generate a **new and different** job role that closely matches the user profile but is likely to yield different search results. Output only the single job title string.
""", 
        input_variables=['context', 'tried_queries', 'result'])
    prompt_final = prompt.invoke({'context': state["user_profile"], 'tried_queries':state["job_queries"], 'result': state['previous_query_result']})
    result = model.invoke(prompt_final)
    query = result.content
    #print(query)
    return {
        "current_query": query,
        "job_queries": state.get("job_queries", []) + [query],
    }

def query_state(state:AgentState) -> AgentState:
        return {**state,
                "previous_query_result": "No jobs found from last query. Try different query.",
                "retry_count": state.get("retry_count", 0) + 1}

def more_query(state: AgentState) -> str:
    job_count = len(state.get('job_listings', []))
    retries = state.get("retry_count", 0)

    if job_count >= 5:
        return "score_jobs"
    if retries >= 4:
        return END

    return "query_state"

    

async def find_jobs(state: AgentState) -> AgentState:
    jobs = await job_search_tool.arun(state["current_query"])
    existing_jobs = state.get("job_listings", [])
    if jobs is not None:
        return {**state, "job_listings": existing_jobs + jobs}
    else:
        return {**state, "job_listings": existing_jobs }

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
        else:
            selected = json.loads(json_content) 
    uid = selected['job']['url']

    for entry in state["scored_jobs"]:
        if entry['job']['url'] == uid:
            description = entry['description']
            score = entry['score']
            break
        else:
            description = None
            score = None
    return {
        **state,
        "best_job": selected["job"],
        "best_job_description": description,
        "best_job_score" : score
    }

def should_retry(state: AgentState) -> str:
    if state["best_job"] and state["best_job"].get("title"):
        return END
    return "generate_query"


# Graph 

graph = StateGraph(AgentState)

graph.add_node("generate_query", generate_query)
graph.add_node("query_state", query_state)
graph.add_node("find_jobs", find_jobs)
graph.add_node("score_jobs", score_jobs)
graph.add_node("pick_best_job", pick_best_job)

graph.set_entry_point("generate_query")

graph.add_edge("generate_query", "find_jobs")
graph.add_conditional_edges("find_jobs", more_query)
graph.add_edge('query_state', "generate_query")
graph.add_edge("score_jobs", "pick_best_job")
graph.add_conditional_edges("pick_best_job", should_retry)

app = graph.compile()


async def JobLooker( external_state):
    internal_state =  {"user_profile": external_state['user_profile'],
        "job_queries": [],
        "job_listings": [],
        "scored_jobs": [],
        "best_job": None,
        "best_job_description": None,
        "best_job_score": None,
        "current_query": " ",
        "retry_count": 0 }
    result = await app.ainvoke(internal_state)
    if result['best_job']:
        return {'best_job': result['best_job'],
                'best_job_description': result['best_job_description'],
                'best_job_score': result['best_job_score']}      
