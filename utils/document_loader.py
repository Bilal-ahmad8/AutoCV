from langchain_community.document_loaders import PyMuPDFLoader
import fitz
from pathlib import Path

# Load content with LangChain
def load_text(pdf_path: Path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        text = doc.page_content
    return text


def extract_links_with_text(pdf_path: Path):
    doc = fitz.open(pdf_path)
    links_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        links = page.get_links()

        for link in links:
            if "uri" in link:
                # Get the rectangle area of the link
                rect = fitz.Rect(link["from"])
                # Extract text inside that rectangle
                anchor_text = page.get_textbox(rect).strip()
                if len(anchor_text) == 0:
                    pass
                else:
                    anchor_text = anchor_text

                links_data.append({
                    "page": page_num + 1,
                    "anchor_text": anchor_text,
                    "url": link["uri"]
                })

    return links_data

def extract_all(state):
    text = load_text(state['pdf_path'])
    doc_links = extract_links_with_text(state['pdf_path'])
    context_text = text + "\nAnchor Text With Link :\n" + str(doc_links)
    return {'raw_resume_text' : context_text} 
