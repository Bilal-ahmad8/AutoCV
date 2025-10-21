# AutoCV

This MVP was built as a solo portfolio project to demonstrate core AI capabilities in resume optimization and job matching.
AutoCV is an AI-powered MVP project designed to help users find the most suitable jobs based on their resumes and automatically tailor their resumes to job requirements. Built mainly using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain), it streamlines the job search and resume optimization process.

## Features

- **Resume Upload**: Users can upload their resume (PDF format).
- **Resume Parsing**: The system intelligently extracts and parses resume content.
- **Profile Generation**: Key user profile details are generated from the resume.
- **Job Matching**: Finds the most suitable jobs for the user based on extracted profile data.
- **Tailored Resume Writing**: Automatically rewrites and tailors the resume to match the job requirements.

## Project Workflow

The core workflow is implemented in [`graph.py`](graph.py), structured as a state graph with the following nodes:

1. **Extract Resume**: Loads and extracts text from the uploaded resume (PDF).
2. **Resume Parser**: Parses the extracted text to generate structured information.
3. **Profile Writer**: Creates a user profile from parsed resume data.
4. **Job Looker**: Searches for and selects the best job match for the user.
5. **Resume Writer**: Rewrites the resume to fit the selected job's requirements.

These steps are connected in a directed flow managed by LangGraph, ensuring smooth data transition and processing.

## Tech Stack

- **Python**: Main language for backend logic.
- **LangGraph**: Manages the workflow as a state graph.
- **LangChain**: Powers the AI components and orchestration.
- **TeX**: Used for document formatting and output.

## Getting Started

1. Clone the repository.
2. Install dependencies (see `requirements.txt`).
3. Run the main application (entry point TBD).
4. Upload your resume and follow the guided flow.

## MVP Scope

- **Automated Job Search**: No manual job hunting required.
- **Resume Tailoring**: Instantly adapts your resume for the best job found.
- **Simple Flow**: Designed for quick demonstrations and proof-of-concept.

## Future Improvements

- Support for more file formats
- Smart Closed Source Models (OpenAI, Anthropic, etc.)
- Enhanced job-matching algorithms
- Integration with job boards and APIs
- User interface improvements

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

*AutoCV: An MVP that lets AI optimize your resume and job search with zero manual effort.*
