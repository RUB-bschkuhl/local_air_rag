#!/usr/bin/env python3
import os
import time
import tempfile
import streamlit as st
from streamlit_chat import message

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Color palette and custom CSS
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

st.set_page_config(page_title="ChatPDF with DeepSeek R1 & Ollama")
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stAppViewContainer {{
     background-color: rgb(14, 17, 23);
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# Define a class that handles PDF ingestion and answering
class ChatPDF:
    def __init__(self):
        self.qa_chain = None
        self.ready = False

    def ingest(self, file_path):
        # Load the PDF and create documents
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        # Split the documents using semantic chunking
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        documents = text_splitter.split_documents(docs)

        # Create the vector store from document chunks
        embedder = HuggingFaceEmbeddings()
        vector = FAISS.from_documents(documents, embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Define LLM and prompt template for QA
        llm = Ollama(model="deepseek-r1:1.5b")
        prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
        3. Keep the answer crisp and limited to 3-4 sentences.
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
        
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            verbose=True,
        )
        
        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            retriever=retriever,
            verbose=True,
            return_source_documents=True
        )
        self.ready = True

    def ask(self, question):
        if not self.ready:
            return "Please upload and ingest a document first."
        result = self.qa_chain(question)
        return result["result"]

    def clear(self):
        self.qa_chain = None
        self.ready = False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chatpdf" not in st.session_state:
    st.session_state["chatpdf"] = ChatPDF()
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Functions to display messages and process input
def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"].strip() != "":
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, True))
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["chatpdf"].ask(user_text)
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""

def read_and_save_file():
    # Clear previous state on new file ingestion
    st.session_state["chatpdf"].clear()
    st.session_state["messages"] = []
    uploaded_files = st.session_state["file_uploader"]
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        with st.spinner(f"Ingesting {file.name}"):
            start_time = time.time()
            st.session_state["chatpdf"].ingest(file_path)
            elapsed = time.time() - start_time
        st.session_state["messages"].append((f"Ingested {file.name} in {elapsed:.2f} seconds", False))
        os.remove(file_path)

# Page layout
st.header("FAISS PDF Chat with Deepseek-r1")
st.subheader("Upload a PDF Document")
st.file_uploader(
    "Upload PDF document",
    type=["pdf"],
    key="file_uploader",
    on_change=read_and_save_file,
    accept_multiple_files=True,
)

display_messages()
st.text_input("Message", key="user_input", on_change=process_input)
