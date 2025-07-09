# Appointment_Booking_chatbot
A Smart Appointment booking bot with semantic memory

🏥 Hospital Appointment Booking Assistant
This is a smart chatbot built using Streamlit, LangChain, and ChromaDB (Cloud) to help users book hospital appointments and provide assistance based on their medical history and past conversations.

The assistant not only books appointments but also remembers your preferences and health details using a memory system powered by vector embeddings.

📌 Features
💬 Conversational chatbot UI built with Streamlit

🧠 Memory retention using ChromaDB Cloud (vector store)

🔐 PII-safe with no repetition of user data

🤖 Responses generated via LLaMA-3 (Groq API) using LangChain

📂 Medical history, preferences, and symptoms are recalled across sessions

☁️ Fully integrated with Chroma Cloud (no local DB needed)

🧱 Tech Stack
Component	Purpose
Streamlit	Frontend interface
LangChain	Prompt handling + response chain
ChromaDB Cloud	Vector memory storage
Groq (LLaMA-3)	Chat model for appointment logic
dotenv	Secrets management via .env file
