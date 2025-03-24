# conda activate chat-with-website
# streamlit run flashRank.py

import bs4
import os
import streamlit as st  # type: ignore
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import AzureOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_loaders import TextLoader
from technique.monoT5 import monoT5_main
from langchain_openai import AzureChatOpenAI
from initialization import get_retriever
from session_persistence import PersistentStorage

storage = PersistentStorage()
storage.load_state()

st.cache_resource.clear()
load_dotenv()
CHROMA_PATH = "chroma"

# Initialize session state for dialog control
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

# !-Streamlit Part-!
st.set_page_config(
    page_title="RAG Evaluator", page_icon="X", layout="wide"
)  # Set layout to wide
st.title(" ")

# Add settings button
if st.sidebar.button("Chunk Configuration"):
    st.session_state.show_settings = not st.session_state.show_settings

# Create the settings section that appears/disappears
if st.session_state.show_settings:
    with st.container():
        st.markdown("### Chunk Configuration")
        st.write(
            "The current chunk_size is ",
            st.session_state.chunk_size,
            " and the current chunk_overlap is ",
            st.session_state.chunk_overlap,
        )
        st.write("For Custom Knowledge Base Only")
        chunk_size = st.text_input("Chunk Size", value=st.session_state.chunk_size)
        chunk_overlap = st.text_input(
            "Chunk Overlap", value=st.session_state.chunk_overlap
        )

        if st.button("Update and Close"):
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            storage.save_state()
            st.session_state.show_settings = False
            st.rerun()


def display_docs(docs):
    """
    Display multiple documents, separated by a full-width horizontal rule.
    """
    separator = """<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />"""
    output = []
    for index, doc in enumerate(docs):
        doc_id = doc.metadata.get("doc_id", "N/A")  # Use 'N/A' if 'doc_id' is missing
        formatted_doc = (
            f"{doc.page_content.strip()}<br><br>"
            f"Source: {doc.metadata.get('source', 'N/A')}"
        )
        output.append(formatted_doc)
    # Join documents with the full-width horizontal rule
    return separator.join(output)


def display_scored_docs(scored_docs):
    """
    Display scored documents in descending order of score,
    separated by a full-width horizontal rule.
    """
    # Sort by score (descending)
    sorted_scored_docs = sorted(scored_docs, key=lambda tup: tup[1], reverse=True)

    separator = """<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />"""
    output = []
    for index, (doc, score) in enumerate(sorted_scored_docs):
        doc_id = doc.metadata.get("doc_id", "N/A")
        formatted_doc = (
            f"{doc.page_content.strip()}<br><br>"
            f"Score: {score:.4f}<br><br>"
            f"Source: {doc.metadata.get('source', 'N/A')}"
        )
        output.append(formatted_doc)
    # Join documents with the full-width horizontal rule
    return separator.join(output)


def get_response(user_input):

    result = monoT5_main(user_input)

    # Check if result is a tuple with 5 elements
    if isinstance(result, tuple) and len(result) == 4:
        response, reranked_docs, scored_doc, docs = result
    else:
        # Assume result is an error message
        response = result
        reranked_docs = scored_doc = docs = None

    return response, scored_doc, docs


#############################################################
# Initialize an empty list to store conversation history
if "history" not in st.session_state:
    st.session_state.history = []


# Create two equal columns with no fixed width
col1, col2 = st.columns(2)  # Two columns, automatically dividing the screen equally

# Custom CSS for full-height columns and separator line
st.markdown(
    """
    <style>
    /* Ensure that both columns take up full height */
    div[data-testid="column"] {
        height: 90vh;
    }
    /* Stretch the block container */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        height: 100vh;
    }
    /* Border between the two columns */
    div[data-testid="column"]:nth-of-type(1) {
        border-right: 2px solid black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Left panel: chatbot interaction
with col1:
    st.header("RAG Chatbot")

    # Display conversation history
    chat_area = st.empty()  # Placeholder to keep the chatbox static
    with chat_area.container():
        for sender, message in st.session_state.history:
            with st.chat_message(sender):
                st.write(message)

    # Keep the input box at the bottom of the left panel
    user_query = st.text_input("Your Input Over Here...", key="input_box")

    if user_query:
        response, scored_docs, docs = get_response(user_query)

        # Update conversation history
        st.session_state.history.append(("Human", user_query))
        st.session_state.history.append(("AI", response))
        st.write(f"Response from AI: {response}")


# Right panel: result display
def display_vectorstore_documents(docs):
    """
    Display documents retrieved from the vectorstore.
    Applies formatting using display_docs and renders in a styled container.
    """
    st.markdown(
        "<h4>Documents retrieved from the vectorstore</h4>", unsafe_allow_html=True
    )
    formatted_docs = display_docs(docs)
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{formatted_docs}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_reranked_documents(scored_docs):
    """
    Display the order after the reranker.
    Applies formatting using display_scored_docs and renders in a styled container.
    """
    st.markdown("<h4>Order after Reranker</h4>", unsafe_allow_html=True)
    formatted_docs = display_scored_docs(scored_docs)

    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{formatted_docs}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_ai_response(response):
    """
    Display the AI response in a styled container.
    """
    st.markdown("<h4>AI Response</h4>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border: 2px solid #007BFF; border-radius: 5px; padding: 10px; background-color: #f0f8ff;">
            <pre>{response}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Right panel: result display
with col2:
    st.header("Behind-the-Scenes")

    if user_query:
        # Container for documents retrieved from the vectorstore
        with st.container():
            display_vectorstore_documents(docs)

        # Container for order after reranker
        with st.container():
            display_reranked_documents(scored_docs)

        # Container for the AI response
        with st.container():
            display_ai_response(response)
