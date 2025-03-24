import os
import time  # import time module to generate timestamp
import streamlit as st
import requests  # used to check the status of the Ollama host
from session_persistence import PersistentStorage
from initialization import get_retriever

storage = PersistentStorage()
storage.load_state()

# Ensure llm_type is initialized in session state
if "llm_type" not in st.session_state:
    st.session_state.llm_type = "local"


def update_cache_file(vectorstore_name):
    st.session_state.active_directory = vectorstore_name
    storage.save_state()


def check_ollama_host(host_url):
    """
    Checks if the Ollama host is accessible by sending a GET request.
    Returns True if the response contains 'Ollama is running', else returns False.
    """
    try:
        response = requests.get(host_url, timeout=5)
        if response.status_code == 200 and "Ollama is running" in response.text:
            return True
    except Exception:
        return False
    return False


def main():
    st.title("RAG Techniques Explorer")
    st.session_state.chunk_size = 1200
    st.session_state.chunk_overlap = 200
    storage.save_state()

    st.write(
        "Welcome! This interactive application allows you to explore various RAG techniques."
    )
    st.header("Upload Documents or Use Existing Vectorstore")

    option = st.selectbox(
        "Do you want to use your own documents as vectorstore?",
        ("No", "Yes"),
    )

    if option == "No":
        update_cache_file("default")
        st.write(":green[Using pre-existing vectorstore (default).]")
    elif option == "Yes":
        upload_type = st.radio(
            "Do you want to upload a single file or multiple files (folder)?",
            ("Single File", "Multiple Files"),
        )

        st.subheader("Chunk Settings")
        st.session_state.chunk_size = st.number_input(
            "Set Chunk Size", min_value=1, value=st.session_state.chunk_size
        )
        st.session_state.chunk_overlap = st.number_input(
            "Set Chunk Overlap", min_value=0, value=st.session_state.chunk_overlap
        )

        if upload_type == "Single File":
            uploaded_file = st.file_uploader(
                "Upload a document (PDF, TXT)", type=["pdf", "txt"]
            )

            if uploaded_file:
                base_folder = "uploaded_files"
                os.makedirs(base_folder, exist_ok=True)
                file_path = os.path.join(base_folder, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                update_cache_file(file_path)
                st.write(f"File uploaded successfully and saved to {file_path}.")
                st.write(f"Active vectorstore set to: {file_path}")
                st.write(
                    f":blue[Chunk size: {st.session_state.chunk_size}, Chunk overlap: {st.session_state.chunk_overlap}]"
                )
            else:
                st.write("No file uploaded.")
        elif upload_type == "Multiple Files":
            folder_name = str(int(time.time() * 1000))
            st.write(f"Generated folder name: {folder_name}")

            uploaded_files = st.file_uploader(
                "Upload documents (PDF, TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                base_folder = "uploaded_files"
                os.makedirs(base_folder, exist_ok=True)
                save_folder = os.path.join(base_folder, folder_name)
                os.makedirs(save_folder, exist_ok=True)
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(save_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                vectorstore_path = save_folder
                update_cache_file(vectorstore_path)
                st.write("Files uploaded successfully and saved to:")
                for path in file_paths:
                    st.write(f"- {path}")
                st.write(f"Active vectorstore set to folder: {vectorstore_path}")
                st.write(
                    f":blue[Chunk size: {st.session_state.chunk_size}, Chunk overlap: {st.session_state.chunk_overlap}]"
                )
                st.write(
                    "Please note that to change to another vectorstore afterwards, please close and reopen the application again."
                )
            else:
                st.write("No files uploaded.")

    # New Section: Ollama Deployment Selection
    st.header("LLM Deployment Selection")

    # Check default HPC availability using default host address
    default_hpc_available = check_ollama_host("http://localhost:8888")
    default_option = "HPC" if default_hpc_available else "Local Ollama"

    # Set the radio button default index based on default_option
    host_option = st.radio(
        "Select LLM deployment",
        ("HPC", "Local Ollama", "API"),
        index=0 if default_option == "HPC" else 1,
    )

    if host_option == "HPC":
        st.session_state.llm_type = "hpc"
        storage.save_state()
        host_address = st.text_input("Enter HPC host address", "http://localhost:8888")
        if check_ollama_host(host_address):
            st.success(f"Ollama is accessible at {host_address}!")
        else:
            st.error(f"Ollama is not accessible at {host_address}!")
    elif host_option == "Local Ollama":
        st.session_state.llm_type = "local"
        storage.save_state()
        if not default_hpc_available:
            st.info("Local Ollama selected because HPC is not available.")
        else:
            st.info("Local Ollama selected.")
    elif host_option == "API":
        st.session_state.llm_type = "api"
        storage.save_state()
        st.info("API selected. Please ensure your API credentials are configured.")


main()
