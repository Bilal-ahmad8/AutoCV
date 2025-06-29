from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


class SummarizeResume:
    def __init__(self, context:str):
        load_dotenv()
        self.model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash-lite')

        self.prompt = PromptTemplate(template="""You are a Helpful AI assistant that Generate a User Profile Based on their Resume / CV content.
                                the profile should conatin:
                                Job Title : Most Suitable Job titles
                                Experience : junior, senior, fresher, etc
                                Education : Education of the user
                                Skills : Different Skills that support the Job Title (Note :- sill should be  given to you explicitly and you are also required to fetch out different skill based on user profile)
                                Summary : Short Summary , that should contain the essence of whole resume based on this job can be matched.
                                
                                Provide the Answer in JSON
                                Resume Content : {context}
                                """, input_variables=['context'])

        self.final_prompt = self.prompt.invoke({'context': context})

    def invoke(self):
        result = self.model.invoke(self.final_prompt)

        return result.content

