# conda activate chat-with-website
# streamlit run main.py

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


# Load environment variables from .env file and initialize llm
load_dotenv()

# Specify the path for the Chroma DB
CHROMA_PATH = "chroma"

# llm = AzureOpenAI(deployment_name="gpt-35-turbo-instruct")
embedding = OllamaEmbeddings(model="bge-m3:latest")
llm = ChatOllama(model="llama3.2", temperature=0)

# retrieve vectorstore from Chroma
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def get_response(user_input):
    answer = rag_chain.invoke(str(user_input))
    return answer


def format_text_for_streamlit(text):
    """Replace newlines in text with <br> for HTML rendering in Streamlit."""
    return text.replace("\n", "<br>")


# !-Streamlit Part-!
# App configuration
st.set_page_config(
    page_title="RAG Evaluator", page_icon="X", layout="wide"
)  # Set layout to wide
st.title(" ")

# Initialize an empty list to store conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

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
        response = get_response(user_query)

        # Update conversation history
        st.session_state.history.append(("Human", user_query))
        st.session_state.history.append(("AI", response))
        st.write(f"Response from AI: {response}")

# Right panel: result display
with col2:
    st.header("Behind-the-scenes")

    if user_query:
        # Retrieve the relevant documents from the retriever
        docs = retriever.get_relevant_documents(user_query)

        # Format and display the concatenated document contents
        formatted_docs = format_docs(docs)

        # Create a container for the process behind the RAG
        with st.container():
            st.markdown(
                "<h4>Documents retrieved from the vectorstore</h4>",
                unsafe_allow_html=True,
            )
            formatted_docs = format_text_for_streamlit(formatted_docs)
            st.markdown(
                f"""
                <div style="border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
                    <pre>{formatted_docs}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Create a container for the AI response
        with st.container():
            st.markdown("<h4>AI Response</h4>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="border: 2px solid #007BFF; border-radius: 5px; padding: 10px; background-color: #f0f8ff;">
                    <pre>{response}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )
