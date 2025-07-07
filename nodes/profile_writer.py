from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


class ProfileWriter:
    def __init__(self, context:str):
        load_dotenv()
        self.model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash-lite')

        self.prompt = PromptTemplate(template = """
                    You are a highly capable AI assistant specialized in extracting structured information from resumes or CVs.

                    Your task is to analyze the provided resume text and generate a structured JSON user profile with the following fields:

                    {{
                                     
                    "Job Title": [String] Most suitable job title(s) based on experience, responsibilities, and skills e.g., Junior Data Scientist, React Developer, Senior Content writer, Machine Learning Engineer, Data Analyst, Data Engineer, DevOps Engineer, etc.",
                    "Experience Years": String (e.g., "0.5", "1", "2") (Give only number e.g., 0.5, 1 ,2)Approximate number of total professional experience years (based on explicit dates or inferred from roles)",
                    "Education": [
                        {{
                        "Degree": String, "Highest relevant degrees, universities, and graduation years (if available)"
                        "University": String,
                        "Graduation Year": String
                        }}
                    ],
                    "Responsibilities": [String] "Key job responsibilities and tasks performed, extracted or inferred from the resume",
                    "Skills": [String] "List of relevant skills explicitly mentioned and inferred from roles, tools, technologies, or context"

                    }}

                    Guidelines:
                    - **IMPORTANT** Always return the response in valid Structured JSON with Array Datatypes.
                    - Use semantic understanding to infer missing but logically implied details (e.g., experience duration or responsibilities).
                    - Include inferred skills and responsibilities even if not explicitly listed, based on job titles, tools, and work history.
                    - The goal is to make the profile useful for job matching or recommendation systems.

                    Resume Content:
                    {context}
                    """, input_variables=['context']
                    )

        self.final_prompt = self.prompt.invoke({'context': context})

    def invoke(self):
        result = self.model.invoke(self.final_prompt)
        final = result.content
        json_start_tag = "```json"
        json_end_tag = "```"

        if final.startswith(json_start_tag) and final.endswith(json_end_tag):
            json_content = final[len(json_start_tag): -len(json_end_tag)]
            # strip any leading/trailing whitespace that might remain
            json_content = json_content.strip()
        else:
            json_content = final

        return final

