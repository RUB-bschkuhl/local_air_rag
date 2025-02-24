import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # Using Chroma instead of FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- App Setup ---
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; color: #4561e9; }
    .stButton>button { background-color: #1E90FF; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
st.title("AirRAG with Chroma and MCTS")

# --- PDF Processing ---
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    
    embedder = HuggingFaceEmbeddings()
    # Create the vector store using Chroma
    vector = Chroma.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # --- Initialize LLM ---
    llm = Ollama(model="deepseek-r1:7b")
    
    # --- Define Reasoning Actions ---
    def system_analysis(query):
        prompt = PromptTemplate.from_template(
            "Given the query: '{query}', rephrase it or decompose it for better clarity. If no changes are needed, return the original query."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def retrieval_answer(query):
        # Retrieve context from Chroma
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = PromptTemplate.from_template(
            "Using the following context:\n{context}\nAnswer the question: {query}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=context, query=query)
    
    def query_transformation(query):
        prompt = PromptTemplate.from_template(
            "Transform the following query to improve retrieval: '{query}'"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def direct_answer(query):
        prompt = PromptTemplate.from_template(
            "Answer the following question directly using your internal knowledge: '{query}'"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def summary_answer(history, context):
        prompt = PromptTemplate.from_template(
            "Given the reasoning history:\n{history}\nand the context:\n{context}\nSummarize and provide a final concise answer."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(history=history, context=context)
    
    # --- Simple MCTS-like Search ---
    def mcts_search(root_query, iterations=3):
        # Root node is the initial query.
        root = root_query
        candidate_paths = []
        
        for _ in range(iterations):
            reasoning_history = []
            
            # Action 1: System Analysis
            analysis = system_analysis(root)
            reasoning_history.append(("SAY", analysis))
            
            # Action 2: Retrieval Answer based on analysis
            retrieval = retrieval_answer(analysis)
            reasoning_history.append(("RA", retrieval))
            
            # Action 3: Query Transformation to refine the original query
            refined_query = query_transformation(analysis)
            reasoning_history.append(("QT", refined_query))
            
            # Action 4: Direct Answer on the refined query
            direct = direct_answer(refined_query)
            reasoning_history.append(("DA", direct))
            
            # Action 5: Summary Answer combines the reasoning steps
            final_answer = summary_answer(
                history="\n".join([f"{act}: {out}" for act, out in reasoning_history]),
                context=retrieval
            )
            reasoning_history.append(("SA", final_answer))
            
            candidate_paths.append((reasoning_history, final_answer))
        
        # Here, a more advanced implementation would verify self-consistency.
        # For now, we simply return the final answer from the first rollout.
        return candidate_paths[0][1]
    
    # --- AirRAG Execution ---
    user_input = st.text_input("Ask a question related to the PDF:")
    if user_input:
        with st.spinner("Running AirRAG reasoning..."):
            final_response = mcts_search(user_input, iterations=3)
            st.write("Final Response:")
            st.write(final_response)
else:
    st.write("Please upload a PDF file to proceed.")
