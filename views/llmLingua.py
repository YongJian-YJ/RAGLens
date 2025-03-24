# conda activate chat-with-website
# streamlit run llmLingua.py

import streamlit as st  # type: ignore
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
from langchain_community.document_compressors import LLMLinguaCompressor
from llmlingua import PromptCompressor
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from technique.llmlingua import llmlingua_main
from initialization import get_retriever
from session_persistence import PersistentStorage

storage = PersistentStorage()
storage.load_state()

st.cache_resource.clear()
global shorter_prompt
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
    If text is not a string, convert it to a string before replacing.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\n", "<br>")


def format_docs(docs):
    """
    Return a single string with each document separated by a full-width horizontal rule
    and labeled with its index.
    """
    # We'll insert an HTML <hr> element for separation.
    # This line will span 100% width of the container.
    separator = """\n<hr style="border: none; border-top: 2px solid #4CAF50; width: 100%; margin: 1em 0;" />\n"""

    formatted_output = []
    for i, doc in enumerate(docs, start=1):
        # Label each document, then add its page_content
        doc_str = f"Document {i}:\n\n{doc.page_content}"
        formatted_output.append(doc_str)

    # Join all documents with the full-width horizontal rule in between
    return separator.join(formatted_output)


def display_retrieved_documents(docs):
    """
    Display the documents retrieved from the vectorstore.
    Applies formatting using format_docs and format_text_for_streamlit.
    """
    st.markdown(
        "<h4>Documents retrieved from the vectorstore</h4>", unsafe_allow_html=True
    )

    # Format the documents first, then replace newlines with <br>
    combined_docs_str = format_docs(docs)
    html_docs_str = format_text_for_streamlit(combined_docs_str)

    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{html_docs_str}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_response(user_input):

    llm_lingua = PromptCompressor(
        model_name="openai-community/gpt2",
        device_map="cpu",
    )
    # compress
    compressed_prompt = llm_lingua.compress_prompt(
        user_input,
        rate=0.33,
        drop_consecutive=True,
    )
    shorter_prompt = "\n\n".join([compressed_prompt["compressed_prompt"], user_input])

    result = llmlingua_main(shorter_prompt)

    if isinstance(result, tuple) and len(result) == 3:
        response, compressed_docs, docs = result
    else:

        response = result
        compressed_docs = docs = None

    return response, compressed_prompt, compressed_docs, docs


#######################################################
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
        response, shorter_prompt, compressed_docs, docs = get_response(user_query)

        # Update conversation history
        st.session_state.history.append(("Human", user_query))
        st.session_state.history.append(("AI", response))
        st.write(f"Response from AI: {response}")


# Right panel: result display
def display_user_query_and_prompt(user_query, shorter_prompt):
    """
    Display the user's original query along with a shorter prompt.
    Applies formatting via format_text_for_streamlit.
    """
    st.markdown("<h4>User Query</h4>", unsafe_allow_html=True)
    formatted_user_query = format_text_for_streamlit(user_query)
    formatted_shorter_prompt = format_text_for_streamlit(shorter_prompt)
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
            <pre>{formatted_user_query}</pre>
            <pre>{formatted_shorter_prompt}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_retrieved_documents(docs):
    """
    Display the documents retrieved from the vectorstore.
    Applies formatting using format_docs and format_text_for_streamlit.
    """
    st.markdown(
        "<h4>Documents retrieved from the vectorstore</h4>", unsafe_allow_html=True
    )
    # Format the documents first via format_docs, then apply streamlit formatting.
    formatted_docs = format_docs(docs)
    formatted_docs = format_text_for_streamlit(formatted_docs)
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
        # Display the user query and prompt in one container.
        with st.container():
            display_user_query_and_prompt(user_query, shorter_prompt)

        # Display the retrieved documents in another container.
        with st.container():
            display_retrieved_documents(docs)

        # Display the AI response.
        with st.container():
            display_ai_response(response)
