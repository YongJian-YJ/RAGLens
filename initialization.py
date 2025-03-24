import os
import streamlit as st  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from session_persistence import PersistentStorage

load_dotenv()


storage = PersistentStorage()
storage.load_state()


@st.cache_resource
def initialize_embeddings_and_llm():
    llm_selection = str(st.session_state.llm_type)
    print("Selected LLM is: ", llm_selection)
    embedding = OllamaEmbeddings(model="bge-m3:latest")

    if llm_selection == "hpc":
        # HPC/host
        llm = ChatOllama(
            model="llama3.3:70b",
            temperature=0.0,
            base_url="http://localhost:8888",
        )
    elif llm_selection == "api":
        # OpenAI Implementation
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
        DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

        llm = AzureChatOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            openai_api_version=OPENAI_API_VERSION,
            model_name=DEPLOYMENT_NAME,
            temperature=0.0,
        )
    else:
        # Ollama
        llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

    return embedding, llm


# Function to create a retriever for a specific file
# @st.cache_resource
def create_file_retriever(file_path):
    # Determine the file extension
    ext = os.path.splitext(file_path)[1].lower()

    # Use PyPDFLoader for PDFs; otherwise, use TextLoader
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and build an in-memory retriever
    embeddings = OllamaEmbeddings(model="bge-m3:latest")  # Use your embedding model
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=None
    )
    return vectorstore.as_retriever()


# Function to create a retriever for files in a directory
# @st.cache_resource
def create_folder_retriever(folder_path):
    # Get a list of all files in the folder
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Load and process all documents from the files
    documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        # Choose the appropriate loader based on file extension
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())  # Add the documents from this file

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and build an in-memory retriever
    embeddings = OllamaEmbeddings(model="bge-m3:latest")  # Use your embedding model
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=None
    )

    return vectorstore.as_retriever()


# Function to get the appropriate retriever based on the provided input
# @st.cache_resource
def get_retriever():
    print("\n\n----Initializing Retriever---\n\n")
    print(
        "The current chunk_size is ",
        st.session_state.chunk_size,
        " and the current chunk_overlap is ",
        st.session_state.chunk_overlap,
    )
    value = st.session_state.active_directory
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    persist_directory = "chroma"

    if value == "default":
        if not os.path.exists(persist_directory):
            print("\n\nDetected Nothing \n")
            loader = TextLoader("../util/data/books/alice_in_wonderland.md")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            vectorstore.persist()  # Save the vectorstore if it's new
            vectorstore = None
        else:
            print("\n\nRetrieving Directly \n")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
        return vectorstore.as_retriever()
    else:
        # Handle other cases like file or folder retrievers
        if os.path.isdir(value):
            return create_folder_retriever(value)
        else:
            return create_file_retriever(value)


####
# Function to create a retriever for a specific file
# @st.cache_resource
def pc_create_file_retriever(file_path):
    # Determine the file extension
    ext = os.path.splitext(file_path)[1].lower()
    # Use PyPDFLoader for PDFs; otherwise, use TextLoader
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and build an in-memory retriever
    embeddings = OllamaEmbeddings(model="bge-m3:latest")  # Use your embedding model
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=None
    )

    return vectorstore, docs


# Function to create a retriever for files in a directory
# @st.cache_resource
def pc_create_folder_retriever(folder_path):
    # Get a list of all files in the folder
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Load and process all documents from the files
    documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        # Choose the appropriate loader based on file extension
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())  # Add the documents from this file

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(st.session_state.chunk_size),
        chunk_overlap=int(st.session_state.chunk_overlap),
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and build an in-memory retriever
    embeddings = OllamaEmbeddings(model="bge-m3:latest")  # Use your embedding model
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=None
    )

    return vectorstore, docs


# Function to get the appropriate retriever based on the provided input
# @st.cache_resource
def pc_get_retriever():
    print("\n\n----Initializing Retriever---\n\n")
    print(
        "The current chunk_size is ",
        st.session_state.chunk_size,
        " and the current chunk_overlap is ",
        st.session_state.chunk_overlap,
    )
    value = st.session_state.active_directory
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    persist_directory = "chroma"

    if value == "default":
        loader = TextLoader("util/data/books/alice_in_wonderland.md")
        docs = loader.load()
        if os.path.exists(persist_directory):
            client_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=False,  # Prevents creation of new collections
            )
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                client_settings=client_settings,
            )
        else:
            raise FileNotFoundError(
                f"Vector store directory '{persist_directory}' does not exist"
            )
        return vectorstore, docs
    else:
        # Handle other cases like file or folder retrievers
        if os.path.isdir(value):
            return pc_create_folder_retriever(value)
        else:
            return pc_create_file_retriever(value)


def create_doc():
    value = st.session_state.active_directory
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    persist_directory = "chroma"

    if value == "default":
        loader = TextLoader("util/data/books/alice_in_wonderland.md")
        docs = loader.load()
        return docs
    else:
        if os.path.isdir(value):
            file_paths = [
                os.path.join(value, f)
                for f in os.listdir(value)
                if os.path.isfile(os.path.join(value, f))
            ]

            # Load and process all documents from the files
            documents = []
            for file_path in file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                # Choose the appropriate loader based on file extension
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents.extend(loader.load())  # Add the documents from this file

            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(st.session_state.chunk_size),
                chunk_overlap=int(st.session_state.chunk_overlap),
            )
            docs = text_splitter.split_documents(documents)

            return docs
        else:
            # Determine the file extension
            ext = os.path.splitext(file_path)[1].lower()
            # Use PyPDFLoader for PDFs; otherwise, use TextLoader
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(st.session_state.chunk_size),
                chunk_overlap=int(st.session_state.chunk_overlap),
            )
            docs = text_splitter.split_documents(documents)
            return docs
