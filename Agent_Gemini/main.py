import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq  # Import Groq model
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Debugging
print("Groq API Key:", groq_api_key)

# Initialize LLM with DeepSeek via Groq
llm = ChatGroq(model_name="gemma2-9b-it", api_key=groq_api_key)  # Updated to DeepSeek

# Initialize search tool
search_tool = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to perform web searches."
    )
]

# Initialize LangChain agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def main():
    st.set_page_config(page_title="AI Web Search Agent", page_icon="🔍")
    st.title("🌍 AI Web Search Assistant")
    st.write("Ask anything, and the AI will fetch relevant answers, including web-based information when needed.")

    # User input
    user_query = st.text_input("Enter your query:")
    if st.button("Ask AI") and user_query:
        with st.spinner("Fetching answer..."):
            response = agent.run(user_query)
        st.subheader("Response:")
        st.write(response)

if __name__ == "__main__":
    main()
