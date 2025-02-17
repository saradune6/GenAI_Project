import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import tempfile

load_dotenv()  # Load environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini Pro Response
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

# Extract and concatenate text from all pages of the given PDF file
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experienced ATS(Application Tracking System) with a deep understanding of the tech field, software engineering, data science, data analysis, and big data engineering. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide the best assistance for improving their resumes. Assign the percentage Matching based on the Job description and the missing keywords with high accuracy.  
resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}}
"""

# Streamlit app
st.title("Resume Match With JD using Gemini 📄")
st.text("Boost Your Resume's ATS Compatibility")

# Sidebar - File Uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")

# Input for Job Description
jd = st.text_area("Paste the Job Description", height=300) 
# Button to submit
submit = st.button("Submit")

# Response Container
response_container = st.container()
# Input Container
container = st.container()

if submit:
    if uploaded_file is not None:
        # Extract text from the uploaded resume
        text = input_pdf_text(uploaded_file)

        # Send the extracted text and job description to Gemini for analysis
        response = get_gemini_response(input_prompt.format(text=text, jd=jd))

        # Display the result in a response container
        with response_container:
            st.subheader("ATS Analysis Result:")
            st.text(response)

