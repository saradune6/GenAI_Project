from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq


st.title("Langchain-DeepSeek-R1 app")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


model = ChatGroq(model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

chain = prompt | model


question = st.chat_input("Enter your question here")
if question: 
    response = chain.invoke({"question": question})
    clean_response = response.content.split("</think>")[-1].strip()
    st.write(clean_response)