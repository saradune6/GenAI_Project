import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

st.title("ChatGPT-like Clone with Gemini Pro")

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # To store chat history

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = chat.invoke(prompt)
            clean_response = response if isinstance(response, str) else response.content  
            st.markdown(clean_response)
        except Exception as e:
            st.error(f"Error fetching response: {e}")
            print("Error:", e)  

    st.session_state["messages"].append({"role": "assistant", "content": clean_response})
