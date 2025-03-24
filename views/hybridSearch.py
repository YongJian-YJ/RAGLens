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
from technique.hyde_hybridSearch import hybrid_main
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


def format_documents(documents):
    # Use an <hr> element with width: 100% to span the entire container
    separator = """<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />"""
    formatted_output = []

    for i, doc in enumerate(documents):
        formatted_output.append(
            f"Document {i + 1}:<br>" f"Content:<br>{doc.page_content}<br>"
        )

    # Join the documents with a full-width horizontal rule
    return separator.join(formatted_output)


def get_response(user_input):
    chunk_size = st.session_state.chunk_size
    chunk_overlap = st.session_state.chunk_overlap

    result = hybrid_main(user_input, chunk_size, chunk_overlap)

    # Check if result is a tuple with 5 elements
    if isinstance(result, tuple) and len(result) == 5:
        response, bm25_results, semantic_results, ensemble_results, hypothetical_doc = (
            result
        )
    else:
        # Assume result is an error message
        response = result
        bm25_results = semantic_results = ensemble_results = hypothetical_doc = None

    return response, bm25_results, semantic_results, ensemble_results, hypothetical_doc


############################################################################
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
        response, bm25_results, semantic_results, ensemble_results, hypothetical_doc = (
            get_response(user_query)
        )

        # Update conversation history
        st.session_state.history.append(("Human", user_query))
        st.session_state.history.append(("AI", response))
        st.write(f"Response from AI: {response}")


# Right panel: result display
def display_hypothetical_answer(hypothetical_doc):
    """
    Display the hypothetical answer generated to enhance semantic search.
    """
    st.markdown(
        "<h4>Hypothetical Answer generated to enhance semantic search</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="border: 2px solid #007BFF; border-radius: 5px; padding: 10px; background-color: #f0f8ff;">
            <pre>{hypothetical_doc}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_bm25_results(bm25_results):
    """
    Display documents retrieved from the BM25 Retriever.
    """
    st.markdown(
        "<h4>Documents retrieved from the BM25 Retriever</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="border: 2px solid #007BFF; border-radius: 5px; padding: 10px; background-color: #f0f8ff;">
            <pre>{format_documents(bm25_results)}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_semantic_results(semantic_results):
    """
    Display documents retrieved from the Semantic Retriever.
    """
    st.markdown(
        "<h4>Documents retrieved from the Semantic Retriever</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="border: 2px solid #007BFF; border-radius: 5px; padding: 10px; background-color: #f5cbf0;">
            <pre>{format_documents(semantic_results)}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_ai_response(response):
    """
    Display the AI response.
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
        with st.container():
            display_hypothetical_answer(hypothetical_doc)

        with st.container():
            display_bm25_results(bm25_results)

        with st.container():
            display_semantic_results(semantic_results)

        with st.container():
            display_ai_response(response)
