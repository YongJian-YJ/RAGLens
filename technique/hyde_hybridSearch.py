import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st  # type: ignore
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from session_persistence import PersistentStorage
from initialization import get_retriever, initialize_embeddings_and_llm
from langchain.prompts import ChatPromptTemplate

storage = PersistentStorage()
storage.load_state()


def HH_generate_hypothetical_document(llm, question):
    """Generate a hypothetical document that would answer the question."""
    hyde_prompt = ChatPromptTemplate.from_template(
        """Given this question: "{question}"
        Write a detailed passage that would answer this question directly and factually.
        The passage should be clear and informative, containing specific details that would be found in a reference document.
        Keep the response under 3 sentences.
        
        Passage:"""
    )

    hyde_chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"question": question})
    return hypothetical_doc


def HH_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()
    return embedding, llm


def HH_get_context():
    value = st.session_state.active_directory

    if value == "default":
        loader = TextLoader("util/data/books/alice_in_wonderland.md")
        documents = loader.load()
    else:
        if os.path.isdir(value):
            file_paths = [
                os.path.join(value, f)
                for f in os.listdir(value)
                if os.path.isfile(os.path.join(value, f))
            ]
            documents = []
            for file_path in file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents.extend(loader.load())
        else:
            ext = os.path.splitext(value)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(value)
            else:
                loader = TextLoader(value)
            documents = loader.load()
    return documents


def HH_get_hybrid_retriever(
    context_text, chunk_size, chunk_overlap, hypothetical_prompt
):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    # Split long text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = text_splitter.create_documents([context_text])

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Get BM25 retriever results
    bm25_results = bm25_retriever.get_relevant_documents(hypothetical_prompt)

    # Initialize semantic search retriever with enhanced document set
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Get semantic search retriever results
    semantic_results = semantic_retriever.get_relevant_documents(hypothetical_prompt)

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6]
    )

    # Get ensemble retriever results
    ensemble_results = ensemble_retriever.get_relevant_documents(hypothetical_prompt)

    return ensemble_retriever, bm25_results, semantic_results, ensemble_results


def HH_setup_rag_chain(retriever, hyde_chain, llm):
    # Enhanced prompt that considers potentially hypothetical content
    template = """Using the following context, answer the question. Note that the context may include both actual documents 
    and hypothetical examples. Focus on factual information that answers the question directly.

    Context: {context}
    
    Question: {question}

    CRITICAL INSTRUCTIONS:
            - Provide ONLY the most direct answer.
            - Your answer must be exactly as brief as the ground truth.
            - Do NOT include any explanation, context, or additional information.
            - Respond with ONLY the key identifying information.

    Example:
            Question: How does Percival get even with O'Gallagher after he takes all of the boy's fireworks?
            Ground truth: Answer: He sets them on fire with the teacher sitting on them
    
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": {"question": RunnablePassthrough()} | hyde_chain | retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def hybrid_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def hybrid_main(question, chunk_size, chunk_overlap, use_hyde=True):
    try:

        # Initialize components
        embedding, llm = HH_initialize_embeddings_and_llm()

        # Get relevant documents using hybrid search
        context = HH_get_context()

        # Generate hypothetical document if HyDE is enabled
        hypothetical_prompt = None
        if use_hyde:
            hypothetical_prompt = HH_generate_hypothetical_document(llm, question)

        # Wrap the hypothetical_prompt in a Runnable so it can be piped
        if use_hyde and hypothetical_prompt is not None:
            hyde_chain = RunnableLambda(lambda _: hypothetical_prompt)
        else:
            # Fallback: use the question as the context if no HyDE result is available
            hyde_chain = RunnableLambda(lambda _: question)

        # Get hybrid retriever instead of single retriever
        retriever, bm25_results, semantic_results, ensemble_results = (
            HH_get_hybrid_retriever(
                context, chunk_size, chunk_overlap, hypothetical_prompt
            )
        )

        # Setup and run RAG chain
        rag_chain = HH_setup_rag_chain(retriever, hyde_chain, llm)
        response = hybrid_get_response(rag_chain, question)
        # print("\n\nhh: ", response)
        return (
            response,
            bm25_results,
            semantic_results,
            ensemble_results,
            hypothetical_prompt,
        )
    except Exception as e:
        return f"Error in main function: {str(e)}"
