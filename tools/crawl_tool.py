import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv


def prettify_description(text):
   load_dotenv()
   model = ChatGroq(model='llama-3.1-8b-instant')
   prompt = PromptTemplate(template = """
You are a helpful AI agent that receives the full scraped text of a job listing web page.

Your task is to analyze the content carefully and extract structured information about the job.

Return a valid JSON object with the following exact fields:

- job_description: A concise and accurate summary of the job. Preserve tone and key terms from the original text.
- skill_required: A list of specific skills explicitly or implicitly required for the role (technologies, tools, soft skills).
- requirement: A list of any mandatory qualifications or expectations (e.g., degrees, certifications, years of experience).
- responsibilities: A list of the core duties or tasks associated with the job. If not available, return an empty list.
- preferred_qualifications: A list of nice-to-have skills, experience, or traits. If none are found, return an empty list.
- required_experience_years: A number indicating the minimum required years of experience. 
    - If a range is mentioned (e.g., "3–5 years"), use the **lower bound**.
    - If a phrase like "3 to 10+ years" appears, take the **first number**.
    - If no value is clearly mentioned, set it to `0`.

⚠️ Output only valid JSON (no explanations or Markdown).
⚠️ The JSON **must match the exact field names and structure**, including the numeric `required_experience_years`.

Here is the full scraped job listing text:
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
        page_timeout= 30000,
        simulate_user=True,          # move mouse, etc.
        override_navigator=True ,     # override navigator for stealth,
        remove_overlay_elements=True,   # auto-close pop-ups
        wait_until="domcontentloaded",
        delay_before_return_html=3.0 )     # wait for page load 
      
      async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

      if not result.success:
        print(f"Crawl failed: {result.error_message}")
        return f"Crawl failed: {result.error_message}"

      html = result.cleaned_html or result.html or ""
      text = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)

      description = prettify_description(text)
      if description:
        json_start_tag = "```json"
        json_end_tag = "```"
        if description.startswith(json_start_tag) and description.endswith(json_end_tag):
          json_content = description[len(json_start_tag):-len(json_end_tag)]
          json_content = json_content.strip()
        else:
           json_content = description

      return json_content

      