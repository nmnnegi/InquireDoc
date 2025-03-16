import streamlit as st
import os
import time
import fitz  # PyMuPDF for PDF handling
from io import BytesIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fpdf import FPDF

# ============================ INITIAL SETUP ============================

# Load environment variables
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Set page config with wide layout
st.set_page_config(page_title="Gemma Doc Q&A", layout="wide", initial_sidebar_state="expanded")

# ====== Session State Initialization ======
session_defaults = {
    "vectors": None,
    "docs": [],
    "summaries": [],
    "chat_history": [],
    "confirm_clear": None,
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================ FUNCTIONS ============================

def extract_text_from_pdf(pdf_bytes):
    """ Extract text from PDFs using PyMuPDF. """
    pdf_text = ""
    with fitz.open("pdf", pdf_bytes) as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text


def summarize_text(text):
    """ Simple summarization by taking the first 500 characters. """
    return text[:500] + "..." if len(text) > 500 else text


def vector_embedding(uploaded_files):
    """ Process PDFs and create embeddings. """
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.docs = []
    st.session_state.summaries = []

    # Extract text and summarize PDFs
    for uploaded_file in uploaded_files:
        pdf_bytes = BytesIO(uploaded_file.read())
        pdf_text = extract_text_from_pdf(pdf_bytes)
        st.session_state.docs.append({"filename": uploaded_file.name, "content": pdf_text})
        st.session_state.summaries.append({"filename": uploaded_file.name, "summary": summarize_text(pdf_text)})

    # Split documents into chunks, maintaining file names
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_documents = []
    for doc in st.session_state.docs:
        chunks = text_splitter.create_documents([doc["content"]])
        for chunk in chunks:
            chunk.metadata = {"filename": doc["filename"]}
        all_documents.extend(chunks)

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(all_documents, st.session_state.embeddings)
    st.success("✅ Vector Store is Ready!")


def generate_pdf(chat_history):
    """ Generate a PDF with chat history. """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Gemma Model Document Q&A - Chat History", ln=True, align='C')
    pdf.ln(10)

    for qa in chat_history:
        pdf.multi_cell(0, 10, f"Q: {qa['question']}", border=0)
        pdf.multi_cell(0, 10, f"A: {qa['answer']}", border=0)
        pdf.ln(5)

    # Save to a temporary file
    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    return pdf_output

# ============================ LLM INITIALIZATION ============================

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# ============================ SIDEBAR ============================

with st.sidebar:
    st.title("📄 Document Uploader")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    # Embed Button
    if st.button("📥 Embed Documents"):
        with st.spinner("Processing PDFs..."):
            vector_embedding(uploaded_files)

    # Clear Button
    if st.button("🗑️ Clear Data"):
        if st.session_state.confirm_clear is None:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Are you sure you want to clear all data?")
        else:
            st.session_state.clear()
            st.success("✅ Data Cleared. Upload new PDFs!")
            time.sleep(1)
            st.rerun()

# ============================ MAIN AREA ============================

st.markdown("<h1 style='text-align: center;'>🕯️ InquireDoc</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>Uncover the Secrets Hidden in Your Documents...</h5>", unsafe_allow_html=True)

# Show document summaries
if st.session_state.summaries:
    st.subheader("📋 Document Summaries:")
    for summary in st.session_state.summaries:
        st.write(f"📄 **{summary['filename']}**")
        st.write(summary["summary"])
        st.write("---")

# Question Input
prompt1 = st.text_input("💬 Ask a question across all documents:")
submit_button = st.button("Get Answer 🚀")

# ============================ PROCESS QUESTION ============================

if submit_button and prompt1 and "vectors" in st.session_state:
    try:
        with st.spinner("Fetching Answer..."):
            # Create Document Chain
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Retrieve relevant documents
            retriever = st.session_state.vectors.as_retriever()
            docs = retriever.invoke(prompt1)

            # Get the response
            start = time.process_time()
            response = document_chain.invoke({"context": docs, "input": prompt1})
            end_time = time.process_time()

            # Add to chat history
            st.session_state.chat_history.append({"question": prompt1, "answer": response})

            # Display response
            st.subheader("🤖 AI Answer:")
            st.write(response)
            st.success(f"Response time: {end_time - start:.2f} seconds")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================ CHAT HISTORY ============================

if st.session_state.chat_history:
    st.subheader("📜 Chat History:")
    for i, qa in enumerate(st.session_state.chat_history):
        with st.container():
            st.write(f"**Q{i+1}:** {qa['question']}")
            st.write(f"**A{i+1}:** {qa['answer']}")
            st.write("---")

    # PDF Download Button
    pdf_data = generate_pdf(st.session_state.chat_history)
    st.download_button("📥 Download Chat as PDF", pdf_data, "chat_history.pdf")

# ============================ Made by Naman ============================
st.markdown(
    """
    <style>
        .made-by {
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: gray;
            font-size: 14px;
        }
    </style>
    <div class='made-by'>Made by Naman</div>
    """,
    unsafe_allow_html=True
)











