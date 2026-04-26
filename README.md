# 🏥 Hospital Operations Knowledge Bot

A smart AI-powered hospital assistant that helps users interact with hospital services, understand medical operations, and get guidance on healthcare-related queries. It combines **Retrieval-Augmented Generation (RAG)** with a **machine learning-based disease prediction system** to provide intelligent and context-aware responses.

---

## 🚀 Features

### 💬 AI Hospital Assistant
- Answers hospital-related queries such as:
  - OPD information
  - Doctor availability
  - Appointment guidance
  - Emergency assistance
- Uses **LLM + FAISS vector database** for accurate responses from hospital data

### 🧠 Disease Prediction System (Add-on Feature)
- Users can input symptoms (e.g., fever, cough, headache)
- Machine Learning model predicts possible diseases
- Acts as a **support tool** when users are unsure about their condition

> ⚠️ This is an **add-on intelligence layer**, not a replacement for medical diagnosis.

### 📚 Context-Aware Chatbot
- Maintains conversation history
- Provides medium-length, clear, and structured responses
- Avoids hallucination by relying on hospital knowledge base

---

## 🏗️ Tech Stack

- Python
- Streamlit (UI)
- LangChain
- FAISS (Vector Database)
- HuggingFace Embeddings
- Groq LLM (LLaMA 3)
- Scikit-learn (ML model)
- Joblib (Model storage)

---

## 🧠 System Architecture

1. User Query → Streamlit UI  
2. Query processed by LLM + RAG system  
3. FAISS retrieves hospital context  
4. LLM generates response  
5. Optional: ML model predicts disease from symptoms  
6. Prediction enhances assistant response (if applicable)
