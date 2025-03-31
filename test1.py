import os
import streamlit as st
import pdfkit
import requests
from jinja2 import Template
from langchain_groq import ChatGroq  # Assuming Groq is available in langchain
from langchain.schema import HumanMessage
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fpdf import FPDF
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()

llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model="llama-3.1-8b-instant", temperature=0.5)
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
SERPER_API_KEY = "80487a458a3aad92da184975fffdddafe9f6a325"

def google_search(query):
    """Perform Google search using Serper API"""
    url = "https://google.serper.dev/search"
    payload = {"q": query, "gl": "us", "hl": "en"}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def web_scraper(url):
    """Simple web content scraper"""
    try:
        response = requests.get(url, timeout=10)
        return response.text[:5000]  # Return first 5000 characters
    except:
        return ""

def process_content(content):
    """Process and chunk content for vector storage"""
    chunks = text_splitter.split_text(content)
    return chunks

def research_agent(topic):
    """Multi-agent research system"""
    
    # Planner Agent
    planner_prompt = f"""Break down this research topic into key sub-questions: {topic}
    Return as a numbered list of questions."""
    questions = llm.invoke([HumanMessage(content=planner_prompt)]).content.split("\n")
    st.session_state.research_data = {"topic": topic, "questions": [], "sources": []}

    # Researcher Agent
    for q in questions[:5]:
        if not q.strip():
            continue
            
        # Web Search
        search_results = google_search(q)
        sources = []
        
        for result in search_results.get("organic", [])[:2]:
            content = web_scraper(result["link"])
            if content:
                chunks = process_content(content)
                sources.append({
                    "title": result.get("title", ""),
                    "link": result["link"],
                    "content": chunks
                })
        
        # Store in vector DB
        if sources:
            docs = [f"Source: {s['title']}\nContent: {chunk}" 
                   for s in sources for chunk in s["content"]]
            Chroma.from_texts(docs, embeddings, collection_name="research_db", persist_directory="./chroma_db")
            
            st.session_state.research_data["questions"].append({
                "question": q,
                "sources": sources
            })

def generate_report():
    """Generate structured research report"""
    report_template = """
    # Research Report: {{ topic }}
    
    ## Overview
    {% for section in sections %}
    ### {{ section.title }}
    {{ section.content }}
    
    Sources:
    {% for source in section.sources %}
    - [{{ source.title }}]({{ source.link }})
    {% endfor %}
    {% endfor %}
    """
    
    sections = []
    for item in st.session_state.research_data["questions"]:
        # Retrieve relevant context from vector DB
        db = Chroma(collection_name="research_db", embedding_function=embeddings)
        docs = db.similarity_search(item["question"], k=3)
        
        # Writer Agent
        prompt = f"""Compile research findings for: {item['question']}
        Context: {[d.page_content for d in docs]}
        Create a comprehensive section with references."""
        
        section_content = llm.invoke([HumanMessage(content=prompt)]).content
        
        sections.append({
            "title": item["question"],
            "content": section_content,
            "sources": item["sources"]
        })
    
    return Template(report_template).render(
        topic=st.session_state.research_data["topic"],
        sections=sections
    )

def create_pdf(content):
    """Generate PDF without using pdfkit and save to system"""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)

    # Check if the file exists and delete it first
    if os.path.exists("research_report.pdf"):
        os.remove("research_report.pdf")
    
    pdf.output("research_report.pdf")

    st.success("PDF saved as 'research_report.pdf' in the current directory.")

def main():
    st.title("🤖 AI Research Agent")
    st.markdown("💡 Enter a topic to generate a comprehensive research report.")
    st.markdown("""
    ### 🛠️ How to Use:
    1. **📝 Enter a Research Topic**: Provide a clear and concise topic in the input box.
    2. **🚀 Start Research**: Click the "Start Research" button to begin the process.
    3. **👀 View Results**: Once the research is complete, preview the generated report.
    4. **📥 Download Report**: Download the report as a PDF for offline use.
    
    **⚠️ Note**: The AI will:
    - Break down the topic into sub-questions.
    - Search for relevant information.
    - Compile a structured report with references.
    """)
    topic = st.text_input("Research Topic:", placeholder="Climate change impacts on biodiversity...")
    
    if st.button("Start Research"):
        if topic:
            with st.spinner("🔍 Conducting research..."):
                research_agent(topic)
                report_content = generate_report()
                
            st.success("✅ Research complete!")
            
            create_pdf(report_content)
            
            with open("research_report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_file,
                    file_name="research_report.pdf",
                    mime="application/pdf"
                )
                
            st.markdown("### Research Preview")
            st.markdown(report_content, unsafe_allow_html=True)
                        
        else:
            st.warning("Please enter a research topic")

if __name__ == "__main__":
    main()