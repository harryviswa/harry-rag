import os
import tempfile
from fpdf import FPDF
import base64

import chromadb
import ollama
import streamlit as st

from streamlit.runtime.uploaded_file_manager import UploadedFile
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

qa_prompt = """
You are a Quality Assurance AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use Table format or bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


qa_strategy_prompt = """
You are a Quality Assurance AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
    Provide the test strategy in a well-structured Markdown format.
    Use H2 (##) for main section titles (e.g., "## Test Objectives").
    Use H3 (###) for sub-section titles if any.
    Use bold emphasis (**) for important terms or phrases where appropriate.
    Ensure lists are formatted with standard Markdown list syntax (e.g., "- List item" or "* List item").
    **For tabular data such as lists of test types, environments, tools, or risk matrices, YOU MUST use standard Markdown pipe table syntax for better readability and structure. Example:
    | Header 1 | Header 2 | Header 3 |
    |---|---|---|
    | Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 |
    | Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 |
    Ensure table headers are separated from rows by a line of hyphens and pipes (e.g., |---|---|).**

    The test strategy must include the following sections (formatted as H2 headings):
    - Test Objectives
    - Scope
    - Test Approach
    - Test Environment
    - Test Deliverables
    - Entry and Exit Criteria
    - Risk Assessment
    - Test Schedule
    - Tools & Technologies
    - Metrics and Reporting
    - Test Automation Strategy

    Additionally, provide an 'effortDistribution' array. This array should contain objects, one for each week of the project timeline (up to the total 'XX' weeks). Each object must have:
    1. 'week': The week number (integer, starting from 1).
    2. 'estimatedTasks': An estimated count (integer) of high-level tasks or major test activities planned for that specific week. This is a numeric representation of effort/activity for charting purposes.

    Furthermore, provide a 'dailyTaskBreakdown' array. For each week present in the 'effortDistribution', distribute its 'estimatedTasks' over 5 working days (Monday to Friday, represented as day 1 to 5).
    Each object in 'dailyTaskBreakdown' must have:
    1. 'week': The week number (integer, corresponding to a week in 'effortDistribution').
    2. 'day': The day of the week (integer, 1 for Monday, 2 for Tuesday, ..., 5 for Friday).
    3. 'tasks': The number of tasks planned for this specific day. The sum of 'tasks' for all 5 days in a given week should ideally equal the 'estimatedTasks' for that week from 'effortDistribution'.

    After the main strategy, provide a 'recommendations' section in Markdown. If you identify that the timeline or scope is extensive for the project, this section should offer actionable advice or highlight areas of concern (e.g., suggest phasing, descoping, resource adjustments). If no specific recommendations are needed due to an appropriate timeline/scope, this field can be omitted or contain a brief statement like "The project timeline and scope appear reasonable."

    Finally, assess and provide the 'testingComplexity' as one of 'Low', 'Medium', 'High' based on the overall project details, features, and timeline. Consider factors like the number of features, integration points, use of new technologies, and the tightness of the schedule relative to the scope.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
   # os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def process_url(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def generate_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./harry-rag-chroma-db")
    return chroma_client.get_or_create_collection(
        name="harry_rag",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Yes, I have successfully analyzed your document and memorized it for future reference!")


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm_strategy(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": qa_strategy_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": qa_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    # Document Upload Area
    st.set_page_config(
            page_title="☄️ Quality Engineering AI Assistant by Harry ☄️", 
            page_icon="☄️"
            layout="wide",
            initial_sidebar_state="expanded"
        )

    with st.sidebar:
        st.set_page_config(page_title="QE AI Assistant", layout="wide")
        st.header("Configuration Settings for your Quality Engineering AI Assistant")

        with st.expander("Add your PDF Resource:"):
            uploaded_file = st.file_uploader(
                "Add your source to help you better", type=["pdf"], accept_multiple_files=False
            )
            process = st.button(
                "➕ Add PDF Content",
            )

        with st.expander("Add your Webpage Resource:"):
            url_input = st.text_input("OR Enter a web page URL to analyze")
            process_url_btn = st.button("➕ Add Web Content")
    
        
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            with st.spinner("Analyzing your document... Please allow me sometime..."):
                all_splits = process_document(uploaded_file)
                add_to_vector_collection(all_splits, normalize_uploaded_file_name)

        if url_input and process_url_btn:
            with st.spinner("Fetching and analyzing web page..."):
                all_splits = process_url(url_input)
                add_to_vector_collection(all_splits, url_input.replace("https://", "").replace("http://", "").replace("/", "_"))

    left, center, right = st.columns([1, 3, 1])

    st.markdown("""
    <style>
    .stApp {
        background-color: black;
        text-align: center;

    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #90caf9;
        background: #fff;
        font-size: 1.1em;
        color: black;
        padding: 1em;
    }
    .stButton button {
        background-color: white;
        color: maroon;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1em;
        font-size: 1em;
        margin: 0.2em;
        width: 200px;
        height: 50px;
        transition: background 0.5s;
    }
    .stButton button:hover {
        background-color: teal;
    }
    .stHeader, .stTitle {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    with center:
        st.title("☄️ Quality Engineering AI Assistant ☄️")
        st.header("Your on-demand AI assistant for Quality Engineering")
        prompt = st.text_area("**Provide your context or questions on the source provided**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ask = st.button("💡 Enlight me")
        with col2:
            summary = st.button("⏱  Quick Summary")
        with col3:
            test_strategy = st.button("💻  Generate Test Strategy")
        with col4:
            test_case = st.button("📤  Generate Test Cases")
    if ask and prompt:
        with st.spinner("ah okay, let me think..."):
            results = query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
            response_text = "".join([chunk for chunk in call_llm(context=relevant_text, prompt=prompt)])
            st.write(response_text)

            pdf_bytes = generate_pdf(response_text)
            st.download_button("Download Result as PDF", data=pdf_bytes, file_name="result.pdf", mime="application/pdf")

            with st.expander("Analyzed documents references:"):
                st.write(results)

            with st.expander("Considered document references:"):
                st.write(relevant_text_ids)
                st.write(relevant_text)
    
    if summary and prompt:
        with st.spinner("Sure, I'm on it... Curating a summary for you..."):
            results = query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
            response = call_llm(context=relevant_text, prompt="Provide a detailed summary on the given requirement."+prompt)
            st.write_stream(response)

           # pdf_bytes = generate_pdf(response)
           # st.download_button("Download Result as PDF", data=pdf_bytes, file_name="AI-Summary.pdf", mime="application/pdf")

            with st.expander("Analyzed documents references:"):
                st.write(results)

            with st.expander("Considered document references:"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

    if test_strategy and prompt:
        with st.spinner("Test strategies are essential in the testing phase... Let me help you with that..."):
            results = query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
            spl_prompt="Provide a detailed test strategy. Also mention the risks, assumptions and estimated efforts required to complete the testing."
            response = call_llm_strategy(context=relevant_text, prompt=spl_prompt)
            st.write_stream(response)

            with st.expander("Analyzed documents references:"):
                st.write(results)

            with st.expander("Considered document references:"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

    if test_case and prompt:
        with st.spinner("Test cases are crucial for ensuring quality... Let me assist you with that..."):
            results = query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
            spl_prompt="List all possible usecases in the table format containing the fields 'S.no, Testcase Summary, Test Conditions, Business Usecase' along with assumptions if any."
            response = call_llm(context=relevant_text, prompt=spl_prompt)
            st.write_stream(response)

            with st.expander("Analyzed documents references:"):
                st.write(results)

            with st.expander("Other Related document references:"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

    if not prompt and (ask or  summary or  test_strategy or  test_case):
        st.error(
            "Please upload a document and provide a question or context to help you better"
        )
        
    st.markdown(
        """
        ---
        Made with ❤️ by [Harry](harry.viswa@gmail.com)
        """
    )