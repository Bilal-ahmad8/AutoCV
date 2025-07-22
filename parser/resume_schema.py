from langchain.output_parsers import StructuredOutputParser, ResponseSchema
schemas = [
    ResponseSchema(
        name="name",
        description="Full name of the candidate."
    ),
    ResponseSchema(
        name="phone",
        description="Phone number with country code (e.g., +91 9876543210)"
    ),
    ResponseSchema(
        name="email",
        description="Professional email address (e.g., name@example.com)"
    ),
    ResponseSchema(
        name="location",
        description="Current city and country (e.g., Lucknow, India)"
    ),
    ResponseSchema(
        name="objective",
        description="One-to-three sentence career objective tailored to the job"
    ),
    ResponseSchema(
        name="skills",
        description="List of key technical and soft skills (e.g., ['Python', 'SQL', 'Team Leadership'])"
    ),
    ResponseSchema(
        name="experience",
        description="""List of work experiences, where each item includes:
        - role: title of the position (e.g., 'Data Analyst Intern')
        - company: organization name (e.g., 'Abhyaz')
        - location: work location (e.g., 'Chennai')
        - duration: time period (e.g., 'Jan 2023 – Mar 2023')
        - responsibilities: list of achievements and tasks
        """
    ),
    ResponseSchema(
        name="education",
        description="""List of education entries, each with:
        - degree: degree or certification (e.g., 'Bachelor of Commerce')
        - institution: university or school name
        - year: range or graduation year (e.g., '2022–2025')
        """
    ),
    ResponseSchema(
        name="projects",
        description="""List of key projects, where each includes:
        - title: name of the project (e.g., 'Brain Tumor Classification')
        - description: [List of what the project accomplished or involved]
        - technologies: list of tools used (e.g., ['Python', 'CNN', 'Transfer Learning'])
        """
    ),
    ResponseSchema(
        name="certifications",
        description="""(Optional) List of professional certifications with:
        - name: certification title (e.g., 'AWS Certified Solutions Architect')
        - provider: issuing organization (e.g., 'Amazon Web Services')
        - year: year completed or valid (e.g., '2023')
        """,
        required=False
    )
]


parser = StructuredOutputParser.from_response_schemas(schemas)

