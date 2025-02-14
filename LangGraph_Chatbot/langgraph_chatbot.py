import streamlit as st
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Step 1: Define State
class State(Dict):
    messages: List[Dict[str, str]]

# Step 2: Initialize StateGraph
graph_builder = StateGraph(State)

# Initialize the LLM
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)

# Define chatbot function (DO NOT modify state["messages"] inside this function)
def chatbot(state: State):
    response = llm.invoke(state["messages"]).content
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}  # Append outside

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")
st.title("🤖 LangGraph Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the response and capture only the first valid output
    state = {"messages": st.session_state.messages}
    assistant_response = None  # Placeholder

    for event in graph.stream(state):
        for value in event.values():
            assistant_response = value["messages"][-1]["content"]
            break  # Stop after first response

    if assistant_response:
        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
