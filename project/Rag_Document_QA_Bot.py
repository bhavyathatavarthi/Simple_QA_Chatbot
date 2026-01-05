import streamlit as st
import os
import time

from langchain_groq import ChatGroq
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

## 1. ENV SETUP
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

# 2. STREAMLIT UI
st.set_page_config(page_title="RAG Q&A with Groq")
st.title("📄 RAG Q&A ChatBot with Groq ")


# 3. LLM & EMBEDDINGS
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(model_name="llama-3.1-8b-instant",temperature=0,streaming=True)


# 4. PROMPT TEMPLATE
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know".
    <context>
    {context}
    </context>
    Question:{question}

    """

)

# 5. FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)


# 6. CREATE VECTOR STORE
def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# 7. PROCESS UPLOADED FILE
if uploaded_file:
    with st.spinner("Processing PDF..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.vectors = create_vector_store(temp_path)
        st.success("✅ PDF processed and indexed!")


# 8. USER QUESTION INPUT
user_question=st.chat_input("Ask a question from the research papers")


# 9. RAG PIPELINE (LCEL)
if user_question:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 3}
    )

    rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    # 🔄 Spinner starts here
    with st.spinner("🤖 Generating answer..."):
        start = time.process_time()
        answer = rag_chain.invoke(user_question)
        elapsed = time.process_time() - start
    # 🔄 Spinner ends here


#10. DISPLAYING ANSWER
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"⏱ Response time: {elapsed:.2f} seconds")


# 11. SHOW RETRIEVED CONTEXT
    with st.expander("🔍 Retrieved Context"):
        docs = retriever.invoke(user_question)
        for i,doc in enumerate(docs,1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content)
            st.divider()






