import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv


def prettify_description(text):
   load_dotenv()
   model = ChatGroq(model='gemma2-9b-it')
   prompt = PromptTemplate(template = """
You are a helpful AI bot that receives the full scraped text of a job listing web page.
Your task is to carefully analyze the content and extract key details about the job.
Return a structured JSON object with the following fields:

- job_description: A concise and accurate summary of the job. Maintain the tone and key terms from the original text.
- skill_required: A list of specific skills explicitly or implicitly required for the job.
- requirement: Any mandatory qualifications or expectations (e.g. years of experience, degrees).
- responsibilities (optional): Key duties or tasks the job entails.
- preferred_qualifications (optional): Nice-to-have skills, experience, or traits.

Only return the final result in valid JSON format.

Here is the full scraped site text:
{context}
"""
)
   
   final_prompt = prompt.invoke({'context': text})
   result = model.invoke(final_prompt)
   return result.content
   
@tool
async def extract_description(url):
      """This Tool takes input a job posting URL and returns json schema for:
      'job_description'
      'skill_required'
      'requirement'
      'responsibilities' """
      browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1280,
        viewport_height=800,
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/116.0.5845.111 Safari/537.36" ))
      
      run_cfg = CrawlerRunConfig(
        simulate_user=True,          # move mouse, etc.
        override_navigator=True ,     # override navigator for stealth,
        remove_overlay_elements=True,   # auto-close pop-ups
        wait_until="networkidle" )     # wait for page load 
      
      async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

      if not result.success:
        raise RuntimeError(f"Crawl failed: {result.error_message}")

      html = result.cleaned_html or result.html or ""
      text = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)

      description = prettify_description(text)
      if description:
        json_start_tag = "```json"
        json_end_tag = "```"
        if description.startswith(json_start_tag) and description.endswith(json_end_tag):
          json_content = description[len(json_start_tag):-len(json_end_tag)]
    # Optional: strip any leading/trailing whitespace that might remain
          json_content = json_content.strip()
        else:
           json_content = description

      return json_content

      
if __name__ == '__main__':
   result = asyncio.run(extract_description('https://www.adzuna.in/details/5248392382?utm_medium=api&utm_source=e1612962'))
   print(result)



""""result = ```json
{
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
  ]
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
}
```"""