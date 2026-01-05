# 📄 RAG Document Q&A Bot (Streamlit + LangChain)

A **Retrieval-Augmented Generation (RAG)** based Streamlit application that allows users to upload PDF documents (research papers) and ask questions.  
The system retrieves the most relevant document chunks using **FAISS vector similarity search** and generates accurate, context-aware answers using an **LLM**.

## 🚀 Features

- 📤 Upload PDF documents
- 📚 Automatic document chunking
- 🔍 Semantic search using FAISS
- 🤖 Context-aware answers using RAG
- ⚡ Fast responses using session caching
- 🧠 No reprocessing for multiple questions
- 🔎 View retrieved context chunks
- 🌐 Interactive UI using Streamlit

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **PyPDF Loader**
- **HuggingFace / OpenAI Embeddings**
- **Groq / OpenAI LLM**

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
 git clone https://github.com/bhavyathatavarthi/Simple_QA_Chatbot.git
 cd Simple_QA_Chatbot
```
### 2️⃣ Create Virtual Environment
```
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 🔑 Environment Variables
Create a .env file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
OR
GROQ_API_KEY=your_groq_api_key
```
### ▶️ Run the Application
```
streamlit run Rag_Document_QA_Bot.py
```




