import os
from typing import List, Any, Dict
from langchain.tools import tool
from dotenv import load_dotenv
import httpx
load_dotenv()
app_id = os.getenv("APP_ID")
app_key = os.getenv('APP_KEY')



@tool
async def job_search_tool(what:str) -> List[Dict[str, Any]]:
    """
Search for jobs based on a given job title or keyword. 

Returns:
- Job title
- Company name
- Location
- Job description
- Redirect URL to the job posting

"""

    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": what,
        "max_days_old": 10,
        "results_per_page": 3 
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(url, params =params)
    #res = requests.get(url, params=params)
    if res.status_code != 200:
        raise Exception(f"API Error: {res.status_code} - {res.text}")

    data = res.json()

    # Extract relevant info
    results = []
    for job in data.get("results", []):
        results.append({
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "description": job.get("description"),
            "url": job.get("redirect_url")
        })

    return results
