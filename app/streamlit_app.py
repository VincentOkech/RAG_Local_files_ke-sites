import sys
from pathlib import Path

# Add the project root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import streamlit as st
from src.agent import RAGAgent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import asyncio
import os
from src.storage_manager import StorageManager
from src.document_processor import DocumentProcessor

# Page config
st.set_page_config(
    page_title="Kenyan RAG Agent",
    page_icon="🇰🇪",
    layout="wide"
)

st.title("🇰🇪 Kenyan RAG Agent")

# Initialize agent
@st.cache_resource
def get_agent():
    return RAGAgent()

agent = get_agent()

# Initialize storage manager
storage = StorageManager()

# Create all necessary directories
required_directories = [
    "data/documents",
    "data/models",
    "data/vector_store",
    "data/vector_store/backups"
]

for directory in required_directories:
    os.makedirs(directory, exist_ok=True)

# File upload section
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                bytes_data = uploaded_file.read()
                file_path = f"data/documents/{uploaded_file.name}"
                os.makedirs("data/documents", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
                
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                    
                documents = loader.load()
                if agent.add_documents(documents):
                    st.success("Document processed successfully!")
                else:
                    st.error("Error processing document")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# Chat interface
st.header("Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about Kenya?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = asyncio.run(agent.get_response(prompt))
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Display system status
with st.sidebar:
    st.header("System Status")
    try:
        stats = agent.vector_store.get_stats()
        st.write("Documents in store:", stats.get("total_documents", "N/A"))
        st.write("Vector dimension:", stats.get("vector_dimension", "N/A"))
    except Exception as e:
        st.write("Status unavailable")

# Add a document listing section
if st.sidebar.checkbox("Show uploaded documents"):
    documents = storage.list_documents()
    if documents:
        st.sidebar.write("Uploaded documents:")
        for doc in documents:
            st.sidebar.write(f"- {doc.name}")
    else:
        st.sidebar.write("No documents uploaded yet") 