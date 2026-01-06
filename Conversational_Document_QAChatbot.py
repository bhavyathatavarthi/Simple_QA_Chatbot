#RAG CONVERSATIONAL Q&A CHATBOT WITH DOCUMENT UPLOAD AND CHAT HISTORY

# IMPORTING REQUIRED LIBRARIES
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv

# SETTING UP ENV
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

# STREAMLIT UI
st.title("Conversational RAG With PDF upload and chat history")
st.set_page_config(page_title="Rag QA Bot")
st.write("Upload Pdf's and chat with their content")

#LLM AND EMBED
llm=ChatGroq(model_name="llama-3.1-8b-instant")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#CREATING SESSION
session_id=st.text_input("Session ID",value="default session")
if 'store' not in st.session_state:
    st.session_state.store={}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

#PROCESSING UPLOADED FILES
uploaded_files=st.file_uploader("Upload a pdf",type="pdf",accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload at least one PDF to start chat")
    st.stop()
with st.spinner("📄Processing PDF"):
    documents=[]
    for file in uploaded_files:
        with open("temp.pdf","wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        documents.extend(docs)

    # SPLIT AND STORE EMBEDDINGS
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    vec_store=Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever=vec_store.as_retriever()
st.success("✅ PDFs processed and indexed!")

#PROMPT TO CONTEXT QUESTION BASED ON CHAT HISTORY
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
rewrite_question = (
        contextualize_prompt
        | llm
        | (lambda msg: msg.content)
)

#RETRIEVAL
retrieve_docs = rewrite_question | retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#ANSWERING FOR QUESTION
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        #MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
rag_runnable = (
    {
        "context": retrieve_docs | format_docs,
        "input": RunnablePassthrough(),
    }
    | qa_prompt
    | llm
)

# ADD MEMORY
conversational_rag = RunnableWithMessageHistory(
    rag_runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# CHAT INPUT
user_input = st.text_input("Ask a question")

if user_input:
    with st.spinner("🤖 Thinking and generating answer..."):
        response = conversational_rag.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.markdown("🧠 Assistant")
        st.write(response.content)

        with st.expander("🔍 Chat History (Debug)"):
            st.write(get_session_history(session_id).messages)








