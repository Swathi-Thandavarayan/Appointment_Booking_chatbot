import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mem0 import MemoryClient
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
import chromadb
from uuid import uuid4
import re
from datetime import datetime


# Load env variables
load_dotenv()

# Initialize mem0 client
MEMORY_CLIENT_API_KEY = os.getenv("MEM0_API_KEY")
memory_client = MemoryClient(api_key=MEMORY_CLIENT_API_KEY)

# Setup Groq LLaMA-3 via LangChain
llm = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    streaming=True,
)

# Setup ChromaDB
api_key = os.getenv("CHROMA_API_KEY")
tenant_id = os.getenv("CHROMA_TENANT_ID")

client = chromadb.CloudClient(
    api_key=api_key,
    tenant=tenant_id,
    database="hospital_conversations"
)


embedding_func = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="hospital_conversations",
    embedding_function=embedding_func
)




# Streamlit config
st.set_page_config(page_title="Appointment Bot", layout="centered")
st.title("🏥 Hospital Appointment Booking Assistant")
st.write("👋 Welcome! I'm here to help with hospital appointments and remember your preferences.")

# ------------------------------
# 📌 Restore full history for backend memory
# ------------------------------
def load_full_chat_history(user_name):
    results = collection.get(
        where={"user_id": user_name}
    )
    # Sort by timestamp to preserve order
    messages = list(zip(results["metadatas"], results["documents"]))
    messages.sort(key=lambda x: x[0].get("timestamp", ""))
    return [(meta["role"], doc) for meta, doc in messages]

# ------------------------------
# 📌 Store message in ChromaDB
# ------------------------------
def log_to_chroma(role, content, user_name):
    doc_id = f"{user_name}_{uuid4()}"
    collection.add(
        documents=[content],
        metadatas=[{
            "role": role,
            "user_id": user_name,
            "timestamp": datetime.utcnow().isoformat()
        }],
        ids=[doc_id]
    )

# ------------------------------
# 📌 Recall memory snippets (semantic)
# ------------------------------
def recall_memory(user_input, user_name, top_k=5):
    results = collection.query(
        query_texts=[user_input],
        n_results=top_k,
        where={"user_id": user_name}
    )
    return results["documents"]

# ------------------------------
# 📌 Generate response
# ------------------------------
def get_response(user_query, memory_docs, user_name, chat_history):
    memory_context = "\n".join([f"- {doc}" for doc in memory_docs])
    template = """
    You are a helpful hospital appointment assistant. Keep your responses short (2-3 lines max).
    Avoid repeating the user's name in every message.
    Ask one question at a time. Greet only once.
    Capture patient needs, medical history, and preferences.
    Show concern if relevant based on memory.

    Chat history: {chat_history}

    User query: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query
    })

# ------------------------------
# ✅ Init session state
# ------------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "temp"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_full_chat_history(st.session_state.user_name)

if "display_history" not in st.session_state:
    st.session_state.display_history = []

# ------------------------------
# 💬 Chat input
# ------------------------------
user_input = st.chat_input("💬 Type your message here...")
if user_input:
    # Append to backend memory
    st.session_state.chat_history.append(("user", user_input))

    # Recall memory snippets
    memory = recall_memory(user_input, st.session_state.user_name, top_k=5)

    # Convert chat history to string
    chat_as_text = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat_history])

    # Get response from LLM
    response_stream = get_response(user_input, memory, st.session_state.user_name, chat_as_text)
    response = "".join(response_stream)

    # Append to backend + display chat
    st.session_state.chat_history.append(("assistant", response))
    st.session_state.display_history.append(("user", user_input))
    st.session_state.display_history.append(("assistant", response))

    # Try to extract name
    if st.session_state.user_name == "temp" and "name is" in response.lower():
        match = re.search(r"name is ([A-Za-z]+)", response)
        if match:
            st.session_state.user_name = match.group(1).strip()

    # Save to ChromaDB
    user_key = st.session_state.user_name
    log_to_chroma("user", user_input, user_key)
    log_to_chroma("assistant", response, user_key)

# ------------------------------
# 💬 Display only current session chat
# ------------------------------
for role, msg in st.session_state.display_history:
    if role == "user":
        st.markdown(f"🧑‍💬  {msg}")
    else:
        st.markdown(f"💼  {msg}")
