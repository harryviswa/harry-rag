import os
import tempfile
import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from openai import OpenAI
import streamlit as st
from prompts import qa_testcase_prompt, qa_prompt, qa_strategy_prompt

#client = OpenAI(api_key="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")

offline_models = ["gpt-oss:20b","llama3.2:3b", "deepseek-r1:latest","qwen3:30b","mistral:7b"]
active_model = offline_models[1]  # Default offline model
embedding_model = "nomic-embed-text:latest"
cross_encoder_model=["cross-encoder/ms-marco-MiniLM-L-12-v2","cross-encoder/ms-marco-MiniLM-L-6-v2"]
def process_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile("wb", delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    if suffix.lower() == ".pdf":
        loader = PyMuPDFLoader(temp_file.name)
    elif suffix.lower() in [".txt"]:
        loader = TextLoader(temp_file.name)
    elif suffix.lower() in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(temp_file.name)
    else:
        st.error("Unsupported file type!")
        return []
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def process_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection():
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=embedding_model,
    )
    chroma_client = chromadb.PersistentClient(path="./harry-rag-chroma-db")
    return chroma_client.get_or_create_collection(
        name="harry_rag",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits, file_name, user_id):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        split.metadata["file_name"] = file_name
        split.metadata["user_id"] = user_id  # Add user/session ID
        metadatas.append(split.metadata)
        ids.append(f"{user_id}_{file_name}_{idx}")
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Yes, I have successfully analyzed your document and memorized it for future reference!")

def query_collection(prompt, user_id, n_results=10):
    collection = get_vector_collection()
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
        where={"user_id": user_id}  # Filter by user/session
    )
    return results

def call_llm(context,sysprompt, prompt, spl_prompt, mode, client):
    if prompt not in st.session_state["prompts"]:
        st.session_state["prompts"].append(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    if mode == "Offline":
        response = ollama.chat(
            model=active_model,
            stream=True,
            messages=[
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": f"Context: {context}, Question: {prompt}, Requirements: {spl_prompt}"},
            ],
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": f"Context: {context}, Question: {prompt}, Requirements: {spl_prompt}"},
            ],
            stream=True,
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk.choices[0].message.content
            else:
                break

# def call_llm(context, prompt, spl_prompt):
#     if prompt not in st.session_state["prompts"]:
#         st.session_state["prompts"].append(prompt)
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     response = ollama.chat(
#         model=active_model,
#         stream=True,
#         messages=[
#             {"role": "system", "content": qa_prompt},
#             {"role": "user", "content": f"Context: {context}, Question: {prompt}, Requirements: {spl_prompt}"},
#         ],
#     )
#     for chunk in response:
#         if chunk["done"] is False:
#             yield chunk["message"]["content"]
#         else:
#             break

def re_rank_cross_encoders(documents, prompt):
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder(cross_encoder_model[0])
    ranks = encoder_model.rank(prompt, documents, top_k=2)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids

def list_vector_sources(userid):
    collection = get_vector_collection()
    all_metadatas = collection.get(include=["metadatas"])["metadatas"]
    sources = set()
    for meta in all_metadatas:
        if(meta.get("user_id") != userid):
            continue
        source = meta.get("source") or meta.get("file_name") or meta.get("url")
        if source:
            sources.add(source)
    return list(sources)