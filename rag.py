from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain import hub

set_debug(True)
set_verbose(True)


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "deepseek-r1:7b"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = hub.pull("rlm/rag-prompt")

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )

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
    

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Please, add a PDF document first."

        # TODO Use air rag ansatz
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None