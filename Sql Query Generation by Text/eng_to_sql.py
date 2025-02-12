import os
from urllib import response
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
load_dotenv() 

# Database connection parameters
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Initialize SQLDatabase
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# Initialize LLM
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
# Create SQL query chain
chain = create_sql_query_chain(llm, db)

def execute_query(question):
    try:
        # Generate SQL query from question
        response = chain.invoke({"question": question})

        # Clean the response to remove any unwanted characters
        cleaned_query = response.strip().replace("```sql", "").replace("```", "").strip()
        cleaned_query = " ".join(cleaned_query.split())  # Remove newlines and extra spaces

        # Execute the cleaned query
        result = db.run(cleaned_query)
                
        # Return the cleaned query and the result
        return cleaned_query, result
    except ProgrammingError as e:
        st.error(f"An error occurred: {e}")
        return None, None



# Streamlit interface
st.title("Question Answering App")

# Input from user
question = st.text_input("Enter your question:")

if st.button("Execute"):
    if question:
        cleaned_query, query_result = execute_query(question)
        
        if cleaned_query and query_result is not None:
            st.write("Generated SQL Query:")
            st.code(cleaned_query, language="sql")  # ✅ Correct variable
            st.write("Query Result:")
            st.write(query_result)
        else:
            st.write("No result returned due to an error.")
    else:
        st.write("Please enter a question.")

        