import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.1-8b-instant")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.

<context>
{context}
</context>

Question: {input}
""")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)
        vectors = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectors = vectors
        st.success("✅ Vector database created!")

st.title("📄 RAG Document Q&A with Groq ")

user_prompt = st.text_input("Ask a question from the research papers")

if st.button("Document Embedding"):
    create_vector_embedding()

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first.")
    else:
        retriever = st.session_state.vectors.as_retriever()

        rag_chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        start = time.process_time()
        response = rag_chain.invoke(user_prompt)
        elapsed = time.process_time() - start

        st.write("✅ Answer")
        st.write(response)

        st.caption(f"⏱ Response time: {elapsed:.2f}s")
