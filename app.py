import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- Custom CSS and App Setup ---
st.markdown("""
    <style>
    .stApp { background-color: #F5F5F5; color: #4561e9; }
    .stButton>button { background-color: #1E90FF; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
st.title("AirRAG: RAG with Tree-Based Reasoning")

# --- PDF Loading and Processing ---
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # --- Initialize LLM ---
    llm = Ollama(model="deepseek-r1:7b")
    
    # --- Define Reasoning Actions ---
    def system_analysis(query):
        # For example, rephrase or decompose the query.
        prompt = PromptTemplate.from_template(
            "Given the query: '{query}', rephrase it to a clearer version or decompose it if needed. If no change is needed, return the original query."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def direct_answer(query):
        # Answer directly based on LLM's parametric knowledge.
        prompt = PromptTemplate.from_template(
            "Answer the following question directly using your own knowledge: '{query}'"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def retrieval_answer(query):
        # Retrieve relevant docs and answer based on context.
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = PromptTemplate.from_template(
            "Using the following context:\n{context}\nAnswer the question: {query}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(context=context, query=query)
    
    def query_transformation(query):
        # Refine or transform the query to improve retrieval.
        prompt = PromptTemplate.from_template(
            "Transform the following question for better retrieval performance: '{query}'"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(query=query)
    
    def summary_answer(history, final_context):
        # Summarize multiple reasoning paths into a final crisp answer.
        prompt = PromptTemplate.from_template(
            "Given the following history of reasoning steps:\n{history}\nand context:\n{final_context}\n"
            "Provide a concise final answer."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(history=history, final_context=final_context)
    
    # --- Simple MCTS Implementation ---
    class MCTSNode:
        def __init__(self, state, parent=None):
            self.state = state              # e.g. current query or reasoning step
            self.parent = parent
            self.children = []
            self.visits = 0
            self.reward = 0.0

    def mcts_search(root_query, iterations=3):
        # Start with the root node representing the user query.
        root = MCTSNode(state=root_query)
        candidate_paths = []
        
        # Each iteration simulates a reasoning path using a fixed sequence of actions.
        for _ in range(iterations):
            current_node = root
            reasoning_history = []
            
            # Selection and Expansion: apply reasoning actions sequentially.
            # Here we use a simple fixed action sequence as a demo.
            # Action 1: System Analysis
            analysis = system_analysis(current_node.state)
            reasoning_history.append(("SAY", analysis))
            
            # Action 2: Retrieval Answer
            retrieval = retrieval_answer(analysis)
            reasoning_history.append(("RA", retrieval))
            
            # Action 3: Query Transformation to refine the query based on RA output.
            refined_query = query_transformation(analysis)
            reasoning_history.append(("QT", refined_query))
            
            # Action 4: Direct Answer (could be combined with RA or invoked separately).
            direct = direct_answer(refined_query)
            reasoning_history.append(("DA", direct))
            
            # Action 5: Summary Answer to combine steps.
            final_answer = summary_answer(
                history="\n".join([f"{act}: {out}" for act, out in reasoning_history]),
                final_context=retrieval
            )
            reasoning_history.append(("SA", final_answer))
            
            # Save the final candidate answer along with its reasoning path.
            candidate_paths.append((reasoning_history, final_answer))
        
        # Here, one would typically run a self-consistency verification to select the best answer.
        # For this blueprint, we simply select the most frequent or the first answer.
        # (In practice, add clustering or scoring of candidate_paths.)
        return candidate_paths[0][1]  # Return final answer of the first rollout
    
    # --- AirRAG Execution ---
    user_input = st.text_input("Ask a question related to the PDF:")
    if user_input:
        with st.spinner("Activating AirRAG reasoning..."):
            final_response = mcts_search(user_input, iterations=3)
            st.write("Final Response:")
            st.write(final_response)
else:
    st.write("Please upload a PDF file to proceed.")
