# 📄 Conversational RAG Document Q&A Bot With Chat History (Streamlit + LangChain)

A Streamlit-based Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with them through conversational Q&A. The system uses semantic search over embedded document chunks and an LLM to generate accurate, context-aware responses with persistent chat history.

## 📸 Screenshots
I have uploaded the screenshots of the other 3 chat bots above here is the modified chat bot 
[chatBot]()

## 🚀 Features

- 📚 Automatic document chunking and embedding generation
- 🔍 Semantic search over documents using Chroma vector database
- 🤖 Context-aware question answering using Retrieval-Augmented Generation (RAG)
- 🧠 History-aware question reformulation for better follow-up answers
- ⚡ Fast responses with session-based chat memory
- 🔎 View retrieved context chunks and chat history for debugging
- 🌐 Interactive web UI built using Streamlit

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




