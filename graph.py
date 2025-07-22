from nodes.resume_parser import ResumeParser
from nodes.profile_writer import ProfileWriter
from nodes.job_looker import JobLooker
from nodes.resume_writer import ResumeWriter
from langgraph.graph import StateGraph
from utils.document_loader import extract_all
from typing import TypedDict


class MainState(TypedDict):
    pdf_path : str
    raw_resume_text : str
    full_resume_text: str
    user_profile : dict
    best_job : dict
    best_job_description: dict
    best_job_score : dict
    tailored_resume_path : str
    graph_executed : bool

graph = StateGraph(MainState)

graph.add_node("extract_resume", extract_all)
graph.add_node("resume_parser", ResumeParser)
graph.add_node("profile_writer", ProfileWriter)
graph.add_node('job_looker', JobLooker )
graph.add_node('resume_writer', ResumeWriter)

graph.set_entry_point("extract_resume")

graph.add_edge("extract_resume", "resume_parser")
graph.add_edge("resume_parser", "profile_writer")
graph.add_edge("profile_writer", "job_looker")
graph.add_edge("job_looker", "resume_writer")


agent = graph.compile()


