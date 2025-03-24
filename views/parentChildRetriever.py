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
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ParentDocumentRetriever
from technique.parentChild import parentChild_main
from langchain_openai import AzureChatOpenAI
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


def format_text_for_streamlit(text):
    """
    Replace newlines in text with <br> for HTML rendering in Streamlit.

    - If the input is a list, join its items with a full-width <hr> separator.
    - For non-string items, attempt to use their 'page_content' attribute or convert to string.
    """
    # Full-width horizontal rule for separating list items
    hr_separator = """<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />"""

    if isinstance(text, list):
        # Convert each element to a string or use 'page_content' if available
        items_as_strings = []
        for item in text:
            if isinstance(item, str):
                items_as_strings.append(item)
            else:
                items_as_strings.append(getattr(item, "page_content", str(item)))
        # Join all list items with the horizontal rule
        text = hr_separator.join(items_as_strings)

    # Finally, replace newlines with <br>
    return text.replace("\n", "<br>")


def format_docs(docs):
    """
    Combine each document's content into one string,
    separated by a full-width horizontal rule.
    """
    # Full-width horizontal rule
    separator = '\n<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />\n'

    formatted_output = []
    for i, doc in enumerate(docs, start=1):
        # Optionally label each document if you like
        doc_str = f"Document {i}:\n{doc.page_content}"
        formatted_output.append(doc_str)

    # Join documents with the full-width horizontal rule in between
    return separator.join(formatted_output)


def get_response(user_input):

    result = parentChild_main(user_input)

    if isinstance(result, tuple) and len(result) == 3:
        response, parent_docs, child_docs = result
    else:

        response = result
        parent_docs = child_docs = None

    return response, parent_docs, child_docs


####################################################################
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
        response, parent_docs, child_docs = get_response(user_query)

        # Update conversation history
        st.session_state.history.append(("Human", user_query))
        st.session_state.history.append(("AI", response))
        st.write(f"Response from AI: {response}")


# Right panel: functions to display result
def display_child_docs(child_docs):
    """
    Display the documents retrieved from the vectorstore.
    Applies formatting using format_docs and format_text_for_streamlit.
    """
    st.markdown(
        "<h4>Documents retrieved from the Child Splitter</h4>", unsafe_allow_html=True
    )

    # 1. Create a single string with docs separated by <hr>
    combined_docs_str = format_docs(child_docs)

    # 2. Convert newlines to <br> for HTML display
    formatted_child_docs = format_text_for_streamlit(combined_docs_str)

    # 3. Render in a styled container
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{formatted_child_docs}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_parent_docs(parent_docs):
    """
    Display the documents retrieved from the parent chunk.
    Applies formatting using format_text_for_streamlit.
    """
    st.markdown(
        "<h4>Documents retrieved from Parent Splitter</h4>", unsafe_allow_html=True
    )
    formatted_parent_docs = format_text_for_streamlit(parent_docs)
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{formatted_parent_docs}</pre>
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
            display_child_docs(child_docs)

        # Container for documents retrieved from the parent chunk
        with st.container():
            display_parent_docs(parent_docs)

        # Container for displaying the AI response
        with st.container():
            display_ai_response(response)
