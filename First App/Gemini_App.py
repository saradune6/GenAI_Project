from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")


# import getpass
# import os


# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key= google_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a chatbot"),
        ("human","Question:{question}")
    ]
)

st.title('Langchain Demo With Gemini')
input_text=st.text_input("Enter your question here")


output_parser=StrOutputParser()


chain=prompt|llm|output_parser   
  
if input_text:
    st.write(chain.invoke({'question':input_text})) 
