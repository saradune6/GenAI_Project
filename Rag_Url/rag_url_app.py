import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in your environment variables.")

st.title("🔍 Smart Web Knowledge Assistant")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# List of URLs to process
urls = [
    "https://www.victoriaonmove.com.au/local-removalists.html",
    "https://victoriaonmove.com.au/index.html",
    "https://victoriaonmove.com.au/contact.html",
]

# Load documents from URLs
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

if not data:
    st.error("Error: No data was loaded from the provided URLs.")
    st.stop()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

if not docs:
    st.error("Error: No text splits were created. Check your data loader and text splitter.")
    st.stop()

# Initialize the embedding model (using Groq or Google, not Vertex AI)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Test embedding model
test_embedding = embedding_model.embed_query("Test query")
if not test_embedding or len(test_embedding) == 0:
    st.error("Error: Failed to generate embeddings. Check your API key and model settings.")
    st.stop()

# Initialize ChromaDB with persistent storage
persist_directory = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model, 
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize LLM (using Groq or another model)
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.4, max_tokens=500, google_api_key=api_key)

# Define prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Chat input
query = st.chat_input("Ask a question: ") 
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    answer = response.get("answer", "No response generated.")

    # Save query and response to session history
    st.session_state.history.append({"question": query, "answer": answer})

    # Display response
    # st.write(f"**Answer:** {answer}")

# Display chat history
# st.subheader("📜 Chat History")
for entry in st.session_state.history:
    st.write(f"**Q:** {entry['question']}")
    st.write(f"**A:** {entry['answer']}")
    st.write("---")
