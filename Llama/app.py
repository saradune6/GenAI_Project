from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.title("LAngchain-LLama3.1 app")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = ChatGroq(groq_api_key=groq_api_key,model='llama3-8b-8192')

chain = prompt | model


question = st.chat_input("Enter your question here")
# if question: 
#     st.write(chain.invoke({"question": question}))

if question:
    response = chain.invoke({"question": question})  # Response is an AIMessage object

    # Extract content properly
    content = getattr(response, "content", None)  # Use getattr to avoid AttributeError

    if content:
        st.markdown(content)
    else:
        st.error("Unexpected response format from model.")


