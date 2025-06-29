from langchain_google_genai import ChatGoogleGenerativeAI
from tools.document_loader import extract_all
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ResumeParser:
    def __init__(self, context):
        load_dotenv()
        self.model = ChatGoogleGenerativeAI('gemma-3-1b-it')

        self.prompt = PromptTemplate(template = """
        You are a resume parser AI assistant.

        You will be given the full text of a person's resume. Your job is to extract the key information from it and return it as a JSON object matching the following structure, If resume missing any information do not fill that entry.:

        ```json
        {{
        "name": "Full name of the person",
        "phone_number": "Phone number of the person",
        "skills": ["List", "of", "skills"],
        "education": ["List of education entries"],
        "summary": "Brief professional summary",
        "projects": {{
            "Project Name 1": "What was done in that project",
            "Project Name 2": "Details about the second project"
        }},
        "hyperlinks": {{"anchor_text1" :"https://link1.com", 
                        "anchor_text2" :"https://link2.com",}},
        "work_experience": {{"Work experience title 1": Description of work title 1,
                                "Work experience title 2": "Work experience title 2" }}],
        "email": "email@example.com",
        "location": "City or address",
        "certifications": ["Certification 1", "Certification 2"]
        }}

        Text of person's resume :
        {context}

        """, input_variables= ["context"])

        self.final_prompt = self.prompt.invoke({'context': context})

    
    def invoke(self):
        result = self.model.invoke(self.final_prompt)
        return result.content

