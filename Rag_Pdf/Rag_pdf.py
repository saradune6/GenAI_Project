import streamlit as st
import time
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY")

st.title("RAG Application built on Gemini Model : (Interview Questions)")

app_dir = os.path.dirname(os.path.abspath(__file__))  
pdf_folder = os.path.join(app_dir, "Pdf_Files")  

# Ensure the folder exists
if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"Folder not found: {pdf_folder}")

pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

data = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf)  # Store only filename
    data.extend(documents)  

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

persist_directory = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model, 
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0, max_tokens=None, timeout=None, google_api_key=api_key)

query = st.chat_input("Say something: ") 
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    sources = set(os.path.basename(doc.metadata.get("source", "Unknown")) for doc in response["context"])

    st.write(response["answer"])
    st.write(f"**Source(s):** {', '.join(sources)}")
