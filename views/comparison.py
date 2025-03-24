# Streamlit app
import streamlit as st
from time import time
import bs4
import os
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
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
from functools import partial
import logging
import time
from initialization import get_retriever, initialize_embeddings_and_llm
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from session_persistence import PersistentStorage
from initialization import get_retriever

# 1. Update the new RAG pipeline here
from technique.llmlingua import llmlingua_main
from technique.reciprocal import reciprocal_main
from technique.hyde_hybridSearch import hybrid_main
from technique.monoT5 import monoT5_main
from technique.bge_reranker_large import bge_main
from technique.parentChild import parentChild_main
from technique.default import default_main


global shorter_prompt
storage = PersistentStorage()
storage.load_state()

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
st.set_page_config(page_title="RAG Techniques Comparison", layout="wide")
st.title("Compare RAG Techniques")

logging.basicConfig(
    filename="app_debug.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_error(message):
    """Log an error message to a file and Streamlit."""
    logging.error(message)
    st.error(f"An error occurred: {message}")


# 2. Create a function that calls the new RAG pipeline
def default_retrieval(query):
    try:
        answer = default_main(query)
        return {"response": answer, "time": time}
    except Exception as e:
        log_error(f"Error in Default Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def llmlingua_retrieval(query):
    try:
        llm_lingua = PromptCompressor(
            model_name="openai-community/gpt2",
            device_map="cpu",
        )

        compressed_prompt = llm_lingua.compress_prompt(
            query,
            rate=0.33,
            drop_consecutive=True,
        )
        shorter_prompt = "\n\n".join([compressed_prompt["compressed_prompt"], query])
        response, compressed_docs = llmlingua_main(shorter_prompt)

        return {"response": response, "time": time}
    except Exception as e:
        log_error(f"Error in LLMLingua Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def parentChild_retrieval(query):
    try:

        answer, parent_docs, child_docs = parentChild_main(query)

        return {"response": answer, "time": time}
    except Exception as e:
        log_error(f"Error in Parent Child Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def reciprocal_retrieval(query):
    try:
        response, generated_queries, stored_results = reciprocal_main(query)

        return {"response": response, "time": time}
    except Exception as e:
        log_error(f"Error in Reciprocal Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def hyde_hybridSearch_retrieval(query):
    try:
        chunk_size = st.session_state.chunk_size
        chunk_overlap = st.session_state.chunk_overlap

        (
            response,
            bm25_results,
            semantic_results,
            ensemble_results,
            hypothetical_prompt,
        ) = hybrid_main(query, chunk_size, chunk_overlap)

        return {"response": response, "time": time}
    except Exception as e:
        log_error(f"Error in Reciprocal Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def monoT5_retrieval(query):
    try:
        response, reranked_docs, scored_docs, docs = monoT5_main(query)

        return {"response": response, "time": time}
    except Exception as e:
        log_error(f"Error in Reciprocal Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


def bgeRerankerLarge_retrieval(query):
    try:
        response, reranked_docs_with_scores, docs = bge_main(query)

        return {"response": response, "time": time}
    except Exception as e:
        log_error(f"Error in Reciprocal Retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


# 3. Update the dictionary to reflect the changes in the application
techniques = {
    "Default Retrieval": default_retrieval,
    "LLMLingua": llmlingua_retrieval,
    "ParentChild": parentChild_retrieval,
    "Reciprocal": reciprocal_retrieval,
    "HyDE + Hybrid Search": hyde_hybridSearch_retrieval,
    "monoT5 Reranker": monoT5_retrieval,
    "BGE Reranker": bgeRerankerLarge_retrieval,
}


@st.cache_resource
def meta_evaluation(query, responses, techniques):
    """
    Meta-agent to evaluate the results from different RAG techniques
    based on relevance, completeness, clarity, and novelty.

    Args:
        query (str): The user query.
        responses (dict): The output from each RAG technique.
        techniques (list): The list of techniques used for retrieval.

    Returns:
        str: The meta-agent's evaluation and recommendation.
    """

    evaluation_prompt = """
        Example Evaluation:
        Query: "What are the main causes of climate change?"

        Response A: "Climate change is primarily driven by greenhouse gas emissions from industrial activities and deforestation."
        Response B: "Climate change is caused by human actions such as burning fossil fuels and deforestation, as well as natural processes like volcanic eruptions."

        Evaluation:

        Response A:
        - Relevance: Directly addresses the query by mentioning human-induced factors.
        - Completeness: Mentions key factors like industrial emissions and deforestation but omits natural causes.
        - Clarity and Structure: Clear and concise explanation.
        - Accuracy: The information is factually correct.

        Response B:
        - Relevance: Fully addresses the query by including both human and natural causes.
        - Completeness: Covers both human actions (burning fossil fuels, deforestation) and natural processes (volcanic eruptions) for a comprehensive view.
        - Clarity and Structure: Well-organized and easy to follow.
        - Accuracy: The content is factually correct.

        Final Recommendation:
        After evaluating both responses, Response B is recommended because it provides a more comprehensive answer by incorporating both human-induced and natural factors.

        Now, using the example above as a guide, evaluate the following responses for the query: "{query}"

        Responses:
        {responses}

        Please evaluate each response based on the following criteria:
        - Relevance to the query
        - Completeness of the answer
        - Clarity and structure
        - Accuracy/factual correctness

        After evaluating all responses, provide a final conclusive recommendation and detailed explanation of your reasoning.
    """

    # Prepare the responses for the agent in the required format
    formatted_responses = ""
    for technique in techniques:
        formatted_responses += (
            f"Technique: {technique}\nResponse: {responses[technique]['response']}\n\n"
        )

    # Use PromptTemplate to format the final prompt
    prompt = PromptTemplate(
        input_variables=["query", "responses"],
        template=evaluation_prompt,
    )

    # Initialize the LLMChain with the prompt and the model
    embedding, model = initialize_embeddings_and_llm()

    llm_chain = LLMChain(prompt=prompt, llm=model)

    # Run the chain with the query and formatted responses
    try:
        meta_agent_response = llm_chain.invoke(
            {"query": query, "responses": formatted_responses}
        )
        evaluation_text = meta_agent_response.get(
            "text", "Error: No evaluation text returned."
        )
    except Exception as e:
        meta_agent_response = f"An error occurred while evaluating: {e}"

    return evaluation_text


@st.cache_resource
def recommend_question(_retriever):
    try:
        # Retrieve all documents from the retriever to generate a question
        docs = retriever.get_relevant_documents("")

        # Combine documents into a single context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Define a prompt template for question generation
        recommendation_prompt = f"""
           Based on the following context, generate a set of meaningful and insightful questions
           designed to test the Retrieval-Augmented Generation (RAG) system's ability to retrieve
           relevant, complete, and accurate information. Each question should be followed by a
           short description of what the question aims to test.

           Focus on these categories, if they are applicable to the context. If applicable, but not limited to these categories:
           - Factual Precision: Questions requiring specific details or complete lists.
           - Temporal Context: Questions addressing dates, historical sequences, or events.
           - Hallucination: Questions that evaluate incorrect or misleading results that AI models generate, test for hallucination.
           - Aggregation: Questions that evaluate grouping or combining related entities.
           - Specific Entity Retrieval: Questions that require retrieving specific names, milestones, or entities.
           - Multi-Faceted Queries: Questions combining multiple aspects or requiring more complex reasoning.
           - Edge Cases: Questions testing the system's ability to handle granularity and completeness.

           For each category, generate 2 questions. Include a short explanation after each question
           to specify what the question aims to test.

           Context:
           {context}

           Generate the questions and explanations in this format:
           Category: <Category Name>
           Question: <Generated Question>
           Explanation: <What the question aims to test>
       """

        # Set up a prompt with the template
        prompt = PromptTemplate(
            input_variables=["context"],
            template=recommendation_prompt,
        )

        # Create the LLM chain
        embedding, model = initialize_embeddings_and_llm()

        llm_chain = LLMChain(prompt=prompt, llm=model)

        # Generate the question using the chain
        recommended_question = llm_chain.invoke({"context": context})

        return recommended_question.get("text", "Error: No question returned.")

    except Exception as e:
        log_error(f"Error in recommending question: {e}")
        return "An error occurred while generating a recommendation."


@st.cache_resource
def retrieve_with_metrics(query, technique):
    """
    Call the appropriate RAG technique's retrieval function and capture response
    along with performance metrics like time and relevance.

    Args:
        query (str): The user query.
        technique (str): The name of the selected RAG technique.

    Returns:
        dict: The response from the RAG technique along with performance metrics.
    """
    start_time = time.time()  # Start timing the retrieval

    # Retrieve the result from the appropriate technique function
    try:
        # Call the specific retrieval function based on the selected technique
        result = techniques[technique](
            query=query, retriever=retriever
        )  # Pass retriever as a keyword argument
        end_time = time.time()  # End timing the retrieval

        # Calculate time taken
        time_taken = end_time - start_time
        relevance = result.get(
            "relevance", 0.0
        )  # You can modify how relevance is calculated

        return {
            "response": result["response"],
            "time": time_taken,
            "relevance": relevance,
        }

    except Exception as e:
        log_error(f"Error in {technique} retrieval: {e}")
        return {
            "response": "An error occurred during retrieval.",
            "time": 0,
            "relevance": 0,
        }


# Assign the retriever
retriever = get_retriever()


# Sidebar Management
st.sidebar.header("Select Techniques")
selected_techniques = st.sidebar.multiselect(
    "Choose RAG techniques to compare:",
    options=list(techniques.keys()),
    default=["Default Retrieval"],
)

# Add settings button
if st.sidebar.button("Chunk Configuration"):
    st.session_state.show_settings = not st.session_state.show_settings

# Create the settings section that appears/disappears
if st.session_state.show_settings:
    with st.container():
        st.markdown("### Settings")
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


# Define the dialog for recommending a question
@st.dialog("Recommended Question")
def show_recommended_question(question):
    st.write(question)
    if st.button("Close"):
        # Clear the session state for the dialog
        del st.session_state.recommended_question
        st.rerun()


# Sidebar button to recommend a question
if st.sidebar.button("Recommend a Question"):
    recommended_question = recommend_question(
        retriever
    )  # Generate the recommended question

    if recommended_question:
        # Store the question in session state
        st.session_state.recommended_question = recommended_question

# Display the dialog if a recommended question exists
if "recommended_question" in st.session_state:
    show_recommended_question(st.session_state.recommended_question)


# Toggle between single query and bulk input
input_mode = st.sidebar.radio("Select Input Mode", ["Single Query", "Bulk Input"])

if input_mode == "Single Query":
    query = st.text_input("Enter your query:")
    if query and selected_techniques:
        st.header("Comparison Results")
        columns = st.columns(len(selected_techniques))

        responses = {}
        for i, technique in enumerate(selected_techniques):
            with columns[i]:
                st.subheader(technique)
                result = techniques[technique](query)
                responses[technique] = result

                st.write("**AI Response:**")
                st.write(result["response"])
                # st.write("**Metrics:**")
                # st.write(f"Time Taken: {result['time']} seconds")
                # st.write(f"Relevance Score: {result['relevance']}")

        st.header("Meta-Evaluation")
        meta_result = meta_evaluation(query, responses, selected_techniques)
        st.write("**Meta-Agent's Evaluation:**")
        st.write(meta_result)
    else:
        st.write("Enter a query and select at least one technique to start.")

elif input_mode == "Bulk Input":
    st.write("Enter up to 8 queries. Leave unused fields blank.")

    queries = [st.text_input(f"Query {i+1}") for i in range(8)]
    filled_queries = [q for q in queries if q.strip()]

    if filled_queries and selected_techniques:
        if st.button("Run Queries"):
            for query in filled_queries:
                st.subheader(f"Results for Query: {query}")
                columns = st.columns(len(selected_techniques))

                responses = {}
                for i, technique in enumerate(selected_techniques):
                    with columns[i]:
                        st.subheader(technique)
                        result = techniques[technique](query)
                        responses[technique] = result

                        st.write("**AI Response:**")
                        st.write(result["response"])
                        # st.write("**Metrics:**")
                        # st.write(f"Time Taken: {result['time']} seconds")
                        # st.write(f"Relevance Score: {result['relevance']}")

                st.subheader("Meta-Evaluation")
                meta_result = meta_evaluation(query, responses, selected_techniques)
                st.write("**Meta-Agent's Evaluation:**")
                st.write(meta_result)
    else:
        st.write("Enter at least one query and select at least one technique to start.")
