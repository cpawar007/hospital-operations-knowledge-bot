import streamlit as st
import numpy as np
import random
import joblib
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

load_dotenv()

st.set_page_config(page_title="Hospital Operations Knowledge Bot")
st.title("🏥 Hospital Operations Knowledge Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mode" not in st.session_state:
    st.session_state.mode = "NORMAL"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "Database_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = init_chat_model("groq:llama-3.3-70b-versatile")

def is_greeting(q):
    return q.lower().strip() in ["hi", "hello", "hey", "hii"]

def is_noise(q):
    return q.lower().strip() in ["sun", "moon", "sky", "ok", "thanks"]

def is_opd_query(q):
    return any(k in q.lower() for k in ["opd", "doctor", "appointment", "book", "consult"])

def is_emergency_query(q):
    return any(k in q.lower() for k in ["chest pain", "breathing", "unconscious", "emergency"])

def format_history(history):
    return "\n".join([f"{r}: {m}" for r, m in history[-6:]])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# INPUT BOX
# ----------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(("User", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    query = user_input
    response = ""

    # ----------------------------
    # SIMPLE RESPONSES
    # ----------------------------
    if is_noise(query):
        response = "Please ask medical or hospital-related questions."

    elif is_greeting(query):
        response = "Hello! 👋 How can I help you today regarding hospital services?"

    # ----------------------------
    # OPD / EMERGENCY LOGIC
    # ----------------------------
    else:
        if is_opd_query(query):
            query += " opd doctors only"

        if is_emergency_query(query):
            query += " emergency department only"

        docs = retriever.invoke(query)
        docs = [d for d in docs if len(d.page_content.strip()) > 30]

        if not docs:
            response = "I don’t have this information in hospital records."
        else:
            context = "\n".join(d.page_content for d in docs)
            history = format_history(st.session_state.chat_history)

            prompt = f"""
You are a helpful hospital assistant.

Be friendly, clear, and medium-length.

If general hospital concept like OPD -> explain properly.

If data missing -> say not available.

Conversation:
{history}

Context:
{context}

User:
{query}

Answer:
"""

            response = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(("Bot", response))

    with st.chat_message("assistant"):
        st.markdown(response)
