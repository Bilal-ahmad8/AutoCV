import asyncio
import json
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
from graph import agent
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Trackers ---
progress_trackers = {}


# ### MODIFIED AGENT LOGIC ###
async def run_langgraph_agent_real(file_path: str, task_id: str):
    """
    This function runs the actual LangGraph agent using astream() 
    to get node outputs and stream progress.
    """  
    node_to_status = {
        "extract_resume": {"status": "Parsing Resume...", "progress": 15},
        "resume_parser": {"status": "Extracting key skills...", "progress": 30},
        "profile_writer": {"status": "Creating professional profile...", "progress": 50},
        "job_looker": {"status": "Searching for best job matches...", "progress": 75},
        "resume_writer": {"status": "Tailoring resume for the job...", "progress": 90},
    }

    progress_trackers[task_id] = {"queue": asyncio.Queue()}
    final_state = {}

    try:
        # Initial state for graph
        initial_state = {
            "pdf_path" : file_path,
            "raw_resume_text": "",
            "full_resume_text": "",
            "user_profile": None,
            "best_job" : {},
            "best_job_description": {},
            "best_job_score" : {},
            "tailored_resume_path": ""
        }

        # Use astream() to get the output of each node as it runs
        async for output in agent.astream(initial_state):
            # astream() yields a dictionary with one key: the node name
            node_name = list(output.keys())[0]
            node_output = output[node_name]

            # Send a progress update to the UI
            if node_name in node_to_status:
                update = node_to_status[node_name]
                await progress_trackers[task_id]["queue"].put(update)
            
            # Merge the output of the node into our final state dictionary
            final_state.update(node_output)
        
        # Send the final completion update using the fully constructed state
        best_job = final_state.get('best_job', {})
        job_title = best_job.get('title', 'N/A')
        job_url = best_job.get('url', '#')
        resume_path = final_state.get('tailored_resume_path', '')

        final_update = {
            "status": "Process Complete!", 
            "progress": 100, 
            "result": {
                "jobTitle": job_title,
                "jobLink": job_url,
                "resumePath": resume_path
            }
        }
        await progress_trackers[task_id]["queue"].put(final_update)

    except Exception as e:
        print(f"Error during agent execution: {e}")
        error_update = {"status": f"An error occurred: {e}", "progress": 100, "error": True}
        await progress_trackers[task_id]["queue"].put(error_update)
    finally:
        # Clean up the uploaded resume file
        if os.path.exists(file_path):
            os.remove(file_path)


# --- API Endpoints ---
@app.post("/process-resume/")
async def process_resume_endpoint(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    # Ensure the temp directory exists
    os.makedirs("temp", exist_ok=True)
    temp_pdf_path = os.path.join("temp", f"{task_id}_{file.filename}")

    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    asyncio.create_task(run_langgraph_agent_real(temp_pdf_path, task_id))
    return JSONResponse({"taskId": task_id})

@app.get("/stream-progress/{task_id}")
async def stream_progress(request: Request, task_id: str):
    async def event_generator():
        # ... (This function remains the same)
        if task_id not in progress_trackers:
            await asyncio.sleep(0.5)
        queue = progress_trackers[task_id]["queue"]
        try:
            while True:
                if await request.is_disconnected(): break
                update = await queue.get()
                print("SSE update:", update)
                yield {"data": json.dumps(update)}
                if update.get("progress") == 100: break
        finally:
            if task_id in progress_trackers: del progress_trackers[task_id]
    return EventSourceResponse(event_generator())

@app.get("/download/{file_name}")
def download_resume(file_name: str):
    """
    Serves the generated resume file for download from the 'output' directory.
    """
    # Security: Basic check to prevent directory traversal attacks
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        return JSONResponse(status_code=400, content={"message": "Invalid file name."})
    
    # Construct a safe path to the file inside the 'output' directory
    output_dir = "output"
    file_path = os.path.join(output_dir, file_name)
    
    # Check if the file exists and serve it
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/octet-stream', filename=file_name)
    
    return JSONResponse(status_code=404, content={"message": "File not found."})


# --- Frontend Serving ---
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoCV | AI Job Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0B1120;
            color: #e5e7eb;
            overflow: hidden;
            position: relative;
        }}
        .background-gradient {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.15) 0%, rgba(11, 17, 32, 0) 25%),
                        radial-gradient(circle at 80% 70%, rgba(192, 132, 252, 0.15) 0%, rgba(11, 17, 32, 0) 25%);
            z-index: -1;
        }}
        .glass-card {{
            background: rgba(17, 24, 39, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
        }}
        .progress-bar-inner {{
            transition: width 0.5s cubic-bezier(0.4, 1, 0.8, 1);
        }}
        .hidden {{ display: none; }}
        .file-input-label:hover {{
            border-color: #38bdf8;
            background-color: rgba(56, 189, 248, 0.05);
        }}
        .upload-icon {{ transition: transform 0.3s ease; }}
        .file-input-label:hover .upload-icon {{ transform: scale(1.1) translateY(-4px); }}
        .fade-in {{ animation: fadeIn 0.5s ease-out forwards; }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .result-item {{
            animation: resultFadeIn 0.5s ease-out forwards;
            opacity: 0;
        }}
        @keyframes resultFadeIn {{
            from {{ opacity: 0; transform: translateX(-20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
    </style>
</head>
<body class="flex items-center justify-center min-h-screen">
    <div class="background-gradient"></div>
    <div class="w-full max-w-lg mx-auto p-4">
        
        <div id="upload-container" class="glass-card rounded-2xl p-8 shadow-2xl text-center">
            <h1 class="text-3xl font-bold mb-2 text-white">AutoCV AI Job Agent</h1>
            <p class="text-gray-400 mb-8">Let our AI agent find the perfect job and tailor your resume.</p>
            <label for="resume-upload" class="file-input-label border-2 border-dashed border-gray-600 rounded-lg p-10 flex flex-col items-center justify-center cursor-pointer transition-all">
                <svg class="upload-icon w-16 h-16 text-gray-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                <span class="text-white font-semibold text-lg">Upload Your Resume</span>
                <span class="text-gray-500 text-sm mt-1">PDF format only</span>
            </label>
            <input type="file" id="resume-upload" class="hidden" accept=".pdf">
        </div>

        <div id="progress-container" class="glass-card rounded-2xl p-8 shadow-2xl hidden">
            <h2 class="text-2xl font-bold text-center text-white mb-4">Working its magic...</h2>
            <p id="status-text" class="text-center text-sky-300 font-medium mb-6 h-6">Initializing...</p>
            <div class="w-full bg-gray-700/50 rounded-full h-3 mb-8 overflow-hidden">
                <div id="progress-bar" class="progress-bar-inner bg-sky-500 h-3 rounded-full" style="width: 0%"></div>
            </div>
        </div>

        <div id="results-container" class="glass-card rounded-2xl p-8 shadow-2xl hidden">
            <div class="text-center mb-6">
                <svg class="w-16 h-16 text-green-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <h2 class="text-2xl font-bold text-white">Success! We found a match.</h2>
            </div>
            <div class="bg-gray-900/50 p-6 rounded-lg result-item">
                <h3 id="job-title" class="text-xl font-bold text-sky-400"></h3>
                <a id="job-link" href="#" target="_blank" class="text-gray-400 hover:text-sky-400 transition-colors underline">View Job Posting</a>
            </div>
            <button id="download-resume-btn" class="w-full bg-sky-600 hover:bg-sky-700 text-white font-bold py-3 px-4 rounded-lg mt-6 transition-transform transform hover:scale-105 result-item" style="animation-delay: 0.2s;">
                Download Tailored Resume
            </button>
            <button id="start-over-btn" class="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg mt-4 transition-transform transform hover:scale-105 result-item" style="animation-delay: 0.3s;">
                Start Over
            </button>
        </div>
    </div>

    <script>
        const uploadContainer = document.getElementById('upload-container');
        const progressContainer = document.getElementById('progress-container');
        const resultsContainer = document.getElementById('results-container');
        const resumeUploadInput = document.getElementById('resume-upload');
        const progressBar = document.getElementById('progress-bar');
        const statusText = document.getElementById('status-text');
        const downloadBtn = document.getElementById('download-resume-btn');
        const startOverBtn = document.getElementById('start-over-btn');

        let eventSource;
        let jumpedToSearch = false;  

        resumeUploadInput.addEventListener('change', handleFileUpload);
        startOverBtn.addEventListener('click', resetUI);

        async function handleFileUpload(event) {{
            if (event.target.files.length === 0) return;
            const file = event.target.files[0];
            jumpedToSearch = false;
            uploadContainer.classList.add('hidden');
            progressContainer.classList.remove('hidden');
            progressContainer.classList.add('fade-in');
            const formData = new FormData();
            formData.append("file", file);
            try {{
                const response = await fetch('/process-resume/', {{
                    method: 'POST',
                    body: formData,
                }});
                if (!response.ok) throw new Error('File upload failed.');
                const data = await response.json();
                const taskId = data.taskId;
                eventSource = new EventSource(`/stream-progress/${{taskId}}`);
                eventSource.onmessage = function(event) {{
                    const update = JSON.parse(event.data);

                    
                statusText.textContent = update.status;
                progressBar.style.width = `${{update.progress}}%`;


                if (update.progress === 50 && !jumpedToSearch) {{
                    jumpedToSearch = true;
                    statusText.textContent  = "Searching for best job matches...";
                    progressBar.style.width = `75%`;
                }}

                if (update.progress === 100) {{
                    eventSource.close();
                    if (update.error) {{
                        statusText.textContent = "An error occurred. Please try again.";
                    }} else {{
                        displayResults(update.result);
                    }}
                }}
            }};

            eventSource.onerror = function(err) {{
                console.error("EventSource failed:", err);
                statusText.textContent = "Error connecting to progress stream.";
                eventSource.close();
            }};

        }} catch (error) {{
            console.error("Error:", error);
            statusText.textContent = `An error occurred: ${{error.message}}`;
        }}
    }}


        function displayResults(result) {{
            progressContainer.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            resultsContainer.classList.add('fade-in');

            document.getElementById('job-title').textContent = result.jobTitle;
            document.getElementById('job-link').href = result.jobLink;
            
            downloadBtn.onclick = () => {{
                window.location.href = `/download/${{result.resumePath}}`;
            }};
        }}

        function resetUI() {{
            resultsContainer.classList.add('hidden');
            uploadContainer.classList.remove('hidden');
            uploadContainer.classList.add('fade-in');
            progressBar.style.width = '0%';
            statusText.textContent = 'Initializing...';
            resumeUploadInput.value = '';
            if(eventSource) {{
                eventSource.close();
            }}
        }}
    </script>
</body>
</html>
    """

# To run the server: uvicorn your_file_name:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
