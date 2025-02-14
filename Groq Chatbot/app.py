import streamlit as st
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import AIMessage
import re 

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)

# Define State class
class State(Dict):
    messages: List[Dict[str, str]]

# Initialize StateGraph
graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    if isinstance(response, AIMessage):
        response = response.content  
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()  # Remove <think> tags
    state["messages"].append({"role": "assistant", "content": response})
    return {"messages": state["messages"]}
   

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(page_title="Chatbot")
st.title("💬 Groq-Powered Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_graph_updates(user_input: str):
    state = {"messages": st.session_state.messages + [{"role": "user", "content": user_input}]}
    for event in graph.stream(state):
        for value in event.values():
            response_content = value["messages"][-1]["content"]
            if isinstance(response_content, AIMessage):
                response_content = response_content.content  # Ensure string content
            yield response_content

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        response = st.empty()
        assistant_response = ""
        for chunk in stream_graph_updates(user_input):
            assistant_response += chunk
            response.write(assistant_response)
        
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})