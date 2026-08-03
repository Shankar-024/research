# ==============================================================================
# 1. INSTALL DEPENDENCIES
# ==============================================================================
import sys
import subprocess
import warnings

# Suppress all DeprecationWarnings (like datetime.utcnow from jupyter_client)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Optionally suppress UserWarnings (like FP16 CPU warnings from Whisper)
warnings.filterwarnings("ignore", category=UserWarning)
print("Installing required libraries...")
packages = [
    "groq>=0.4.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.20",
    "langchain-groq>=0.1.0",
    "sentence-transformers>=2.2.2",
    "faiss-cpu>=1.7.4",
    "pymupdf>=1.23.0",
    "gradio>=4.15.0",
    "arxiv>=2.1.0",
    "duckduckgo-search>=4.2.0",
    "pandas>=2.0.0"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✅ Libraries installed successfully!\n")
!pip install duckduckgo-search serpapi
# ==============================================================================
# 2. IMPORTS & CONFIGURATION
# ==============================================================================
import os
import re
import warnings
import gradio as gr
import fitz  # PyMuPDF
import arxiv
from typing import Dict, List, Any
from google.colab import userdata
from duckduckgo_search import DDGS

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Try loading API keys directly from Google Colab Secrets (userdata.get)
try:

    # Read GROQ_API_KEY from secrets
    try:
        DEFAULT_GROQ_KEY = userdata.get('GROQ_API_KEY')
    except Exception:
        DEFAULT_GROQ_KEY = ""

    # Read SERPAPI_API_KEY from secrets
    try:
        DEFAULT_SERPAPI_KEY = userdata.get('SERPAPI_API_KEY')
    except Exception:
        DEFAULT_SERPAPI_KEY = ""

except ImportError:
    # Environment variable fallback if executing outside Google Colab
    DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
    DEFAULT_SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", os.getenv("SERPAPI_KEY", ""))

print(f"Keys Configured -> Groq Key Loaded: {bool(DEFAULT_GROQ_KEY)} | SerpApi Key Loaded: {bool(DEFAULT_SERPAPI_KEY)}")
# ==============================================================================
# 3. SECTION PARSER MODULE
# ==============================================================================
SECTION_PATTERNS = {
    "Abstract": re.compile(r"^(abstract)", re.IGNORECASE),
    "Introduction": re.compile(r"^(1\.?\s*introduction|introduction)", re.IGNORECASE),
    "Related Work": re.compile(r"^(\d\.?\s*related work|background)", re.IGNORECASE),
    "Methodology": re.compile(r"^(\d\.?\s*methodology|proposed method|system model|architecture)", re.IGNORECASE),
    "Results": re.compile(r"^(\d\.?\s*experiments|results|evaluation)", re.IGNORECASE),
    "Discussion & Gaps": re.compile(r"^(\d\.?\s*discussion|limitations|threats to validity)", re.IGNORECASE),
    "Conclusion": re.compile(r"^(\d\.?\s*conclusion|future work)", re.IGNORECASE),
}

def extract_structured_sections(pdf_path: str) -> Dict[str, str]:
    """Extracts text grouped by academic sections from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = [page.get_text() for page in doc]
    text = "\n".join(full_text)
    lines = text.split("\n")

    sections = {
        "Abstract": "", "Introduction": "", "Related Work": "",
        "Methodology": "", "Results": "", "Discussion & Gaps": "",
        "Conclusion": "", "Other": ""
    }

    current_section = "Abstract"
    for line in lines:
        clean_line = line.strip()
        matched = False
        for sec_name, pattern in SECTION_PATTERNS.items():
            if pattern.match(clean_line) and len(clean_line) < 60:
                current_section = sec_name
                matched = True
                break
        if not matched:
            sections[current_section] += line + "\n"

    return {k: v.strip() for k, v in sections.items() if len(v.strip()) > 50}
# ==============================================================================\n
# 4. EXTERNAL SEARCH MODULE (arXiv + Web)
# ==============================================================================\n
import time
import arxiv
from duckduckgo_search import DDGS
from typing import List, Dict
import serpapi

def search_arxiv_papers(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Searches arXiv with built-in retry and rate-limit handling."""
    # Configure client with delayed retries and smaller page size
    client = arxiv.Client(
        page_size=10,
        delay_seconds=3,
        num_retries=3
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    try:
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "summary": paper.summary[:300].replace("\n", " ") + "...",
                "url": paper.pdf_url,
                "published": str(paper.published.date())
            })
            time.sleep(1)  # Polite delay between iterations
    except Exception as e:
        print(f"arXiv Search Warning/Error: {e}")
        # Return empty list or partial results instead of crashing the app
        return results

    return results

def search_similar_online_papers(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        with DDGS() as ddgs:
            search_query = f"site:arxiv.org OR site:openreview.net {query} paper"
            for r in ddgs.text(search_query, max_results=3):
                results.append({
                    "title": r.get("title", "Untitled"),
                    "link": r.get("href", "#"),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"DuckDuckGo Search Error: {e}")
    return results

def explore_external_papers(topic: str) -> str:
    arxiv_res = search_arxiv_papers(topic)
    web_res = search_similar_online_papers(topic)

    out = "### Relevant arXiv Papers\n"
    if arxiv_res:
        for p in arxiv_res:
            out += f"- **[{p['title']}]({p['url']})** ({p['published']})\n  *{p['summary']}*\n\n"
    else:
        out += "_arXiv API is currently rate-limited (429) or unavailable. Displaying web search results below._\n\n"

    out += "\n### Related Web & OpenReview Papers\n"
    if web_res:
        for w in web_res:
            out += f"- **[{w['title']}]({w['link']})**\n  {w['snippet']}\n\n"
    else:
        out += "_No web search results found._\n"

    return out
# ==============================================================================
# 5. RAG & GROQ ENGINE
# ==============================================================================
class PaperAnalysisRAG:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.parsed_papers = {}

    def ingest_papers(self, paper_dict: Dict[str, Dict[str, str]]):
        self.parsed_papers = paper_dict
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        for paper_name, sections in paper_dict.items():
            for section_name, content in sections.items():
                chunks = splitter.split_text(content)
                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"paper": paper_name, "section": section_name}
                        )
                    )
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def extract_specific_sections(self, paper_name: str, target_sections: List[str]) -> str:
        if paper_name not in self.parsed_papers:
            return "Paper not found."
        output = []
        sections = self.parsed_papers[paper_name]
        for sec in target_sections:
            if sec in sections:
                output.append(f"### {sec}\n{sections[sec]}")
            else:
                output.append(f"### {sec}\n*Section not explicitly identified in paper.*")
        return "\n\n".join(output)

    def generate_comparative_matrix(self, target_aspect: str) -> str:
        context_blocks = []
        for paper, sections in self.parsed_papers.items():
            context_blocks.append(f"--- PAPER: {paper} ---")
            for sec_name, content in sections.items():
                context_blocks.append(f"[{sec_name}]: {content[:800]}...")

        full_context = "\n".join(context_blocks)
        prompt = f"""You are an expert research reviewer. Compare all provided papers based on: **{target_aspect}**.\n\nProvide a detailed comparison table in Markdown with columns:\n| Paper Name | Key Approach / Methodology | Dataset / Evaluation | Main Findings | Identified Limitations |\n\nFollowed by 3 key comparative bullet points.\n\nPAPER CONTEXT:\n{full_context[:12000]}\n"""
        return self.llm.invoke(prompt).content

    def identify_research_gaps(self) -> str:
        context = []
        for paper, sections in self.parsed_papers.items():
            gaps_content = sections.get("Discussion & Gaps", "") + " " + sections.get("Conclusion", "") + " " + sections.get("Methodology", "")
            context.append(f"Paper ({paper}):\n{gaps_content[:2000]}")

        prompt = f"""Synthesize the provided papers to discover open research gaps and future directions.\n\nStructure your analysis into:\n1. **Common Assumptions & Shortcomings** across the literature.\n2. **Unaddressed Edge Cases / Evaluation Gaps**.\n3. **3 Concrete Proposals for Novel Research** building on these gaps.\n\nPapers Summary:\n{"\n---\n".join(context)}\n"""
        return self.llm.invoke(prompt).content

    def query_rag(self, query: str, selected_section: str = "All") -> str:
        if not self.vector_store:
            return "Please upload papers first."

        filter_dict = None if selected_section == "All" else {"section": selected_section}
        docs = self.vector_store.similarity_search(query, k=4, filter=filter_dict)
        retrieved_text = "\n\n".join([f"[Source: {d.metadata['paper']} | Section: {d.metadata['section']}]\n{d.page_content}" for d in docs])

        prompt = f"""Answer the question accurately using ONLY the provided excerpts from research"""
# ==============================================================================
# 6. GRADIO INTERFACE HANDLERS
# ==============================================================================
def initialize_rag(files, api_key):
    global rag_system
    if not api_key:
        return "⚠️ Please enter a valid Groq API Key.", gr.update(choices=[]), gr.update(choices=[])
    if not files:
        return "⚠️ Please upload at least one PDF paper.", gr.update(choices=[]), gr.update(choices=[])

    rag_system = PaperAnalysisRAG(groq_api_key=api_key)
    parsed_all = {}

    for file in files:
        paper_name = os.path.basename(file.name)
        sections = extract_structured_sections(file.name)
        parsed_all[paper_name] = sections

    rag_system.ingest_papers(parsed_all)
    paper_list = list(parsed_all.keys())

    return f"✅ Processed {len(files)} paper(s). Vector store ready!", gr.update(choices=paper_list), gr.update(choices=paper_list)

def extract_sections_ui(paper_name, sections):
    return rag_system.extract_specific_sections(paper_name, sections) if rag_system else "Please initialize system."

def run_comparison(aspect):
    return rag_system.generate_comparative_matrix(aspect) if rag_system else "Please initialize system."

def run_gap_analysis():
    return rag_system.identify_research_gaps() if rag_system else "Please initialize system."

def answer_query(query, section_filter):
    return rag_system.query_rag(query, section_filter) if rag_system else "Please initialize system."

def explore_external_papers(topic):
    arxiv_res = search_arxiv_papers(topic)
    web_res = search_similar_online_papers(topic)

    out = "### Relevant arXiv Papers\n"
    for p in arxiv_res:
        out += f"- **[{p['title']}]({p['url']})** ({p['published']})\n  *{p['summary']}*\n\n"

    out += "\n### Related Web & OpenReview Papers\n"
    for w in web_res:
        out += f"- **[{w['title']}]({w['link']})**\n  {w['snippet']}\n\n"
    return out
# ==============================================================================
# 7. GRADIO APP LAYOUT & LAUNCH
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="PragyanAI Academic Paper RAG Engine") as demo:
    gr.Markdown("# Multi-Paper Academic RAG & Research Gap Explorer")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Groq API Key",
            type="password",
            value=DEFAULT_GROQ_KEY,
            placeholder="gsk_..."
        )
        file_uploader = gr.File(label="Upload PDF Papers", file_count="multiple", file_types=[".pdf"])
        process_btn = gr.Button(" Ingest Papers", variant="primary")

    status_output = gr.Label(value="Upload PDF files and click Ingest to start.")

    with gr.Tabs():
        with gr.TabItem("1. Extract Sections"):
            with gr.Row():
                paper_dropdown = gr.Dropdown(label="Select Paper", choices=[])
                section_selector = gr.CheckboxGroup(
                    choices=["Abstract", "Introduction", "Related Work", "Methodology", "Results", "Discussion & Gaps", "Conclusion"],
                    label="Select Sections to Inspect",
                    value=["Abstract", "Methodology", "Results"]
                )
            extract_btn = gr.Button("Extract Selected Sections")
            extracted_display = gr.Markdown()
            extract_btn.click(extract_sections_ui, inputs=[paper_dropdown, section_selector], outputs=extracted_display)

        with gr.TabItem("2. Comparative Analysis"):
            aspect_input = gr.Textbox(label="Comparison Focus", value="Methodology, Datasets, Models, and Results")
            compare_btn = gr.Button("Generate Comparison Table", variant="primary")
            comparison_display = gr.Markdown()
            compare_btn.click(run_comparison, inputs=[aspect_input], outputs=comparison_display)

        with gr.TabItem("3. Research Gap Finder"):
            gap_btn = gr.Button("Analyze Gaps & Future Work", variant="primary")
            gap_display = gr.Markdown()
            gap_btn.click(run_gap_analysis, outputs=gap_display)

        with gr.TabItem("4. Deep Q&A (RAG)"):
            with gr.Row():
                query_input = gr.Textbox(label="Enter Question", placeholder="What are the main architectural limitations mentioned?")
                section_filter = gr.Dropdown(
                    choices=["All", "Abstract", "Introduction", "Methodology", "Results", "Discussion & Gaps"],
                    value="All",
                    label="Filter Context by Section"
                )
            qa_btn = gr.Button("Ask RAG Engine")
            qa_display = gr.Markdown()
            qa_btn.click(answer_query, inputs=[query_input, section_filter], outputs=qa_display)

        with gr.TabItem("5. Discover Similar Papers"):
            topic_input = gr.Textbox(label="Search Query / Research Topic", placeholder="Multi-agent systems for PCB design")
            search_btn = gr.Button("Find Similar Papers (arXiv + Web)")
            discovery_display = gr.Markdown()
            search_btn.click(explore_external_papers, inputs=[topic_input], outputs=discovery_display)

    process_btn.click(
        initialize_rag,
        inputs=[file_uploader, api_key_input],
        outputs=[status_output, paper_dropdown, paper_dropdown]
    )

# Launch Gradio with public share link enabled for Colab
demo.queue().launch(share=True, debug=True)
