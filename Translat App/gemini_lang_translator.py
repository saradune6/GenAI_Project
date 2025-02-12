
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key= google_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)


st.title('Langchain Demo With Gemini (language translator)')
input_text=st.text_input("Write the sentence in english and it will be translated in Hindi")



output_parser=StrOutputParser()

chain=prompt|llm|output_parser  

if input_text:
    st.write(chain.invoke(
    {
        "input_language": "English",
        "output_language": "Hindi",
        "input": input_text, 
    }
))

      
