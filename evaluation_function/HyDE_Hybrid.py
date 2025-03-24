import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import hub
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from initialization import get_retriever, initialize_embeddings_and_llm
import sys


def filter_unique_contexts(documents):
    unique_texts = set()
    unique_docs = []

    for doc in documents:
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs


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
    try:
        embedding, llm = initialize_embeddings_and_llm()

        return embedding, llm
    except Exception as e:
        print("Initialization error:", e)
        raise


def HH_get_hybrid_retriever(context_text, hypothetical_doc):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    # Split long text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    documents = text_splitter.create_documents([context_text])

    documents = filter_unique_contexts(documents)

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4

    # Initialize semantic search retriever with enhanced document set
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create ensemble retriever with adjusted weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], weights=[0.4, 0.6]
    )

    return ensemble_retriever


def HH_setup_rag_chain(retriever, llm):
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
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def HH_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def HH_main(question, context, use_hyde=True):
    try:
        # Initialize components
        embedding, llm = HH_initialize_embeddings_and_llm()

        # Generate hypothetical document if HyDE is enabled
        hypothetical_doc = None
        if use_hyde:
            hypothetical_doc = HH_generate_hypothetical_document(llm, question)

        # Get hybrid retriever with optional HyDE document
        retriever = HH_get_hybrid_retriever(context, hypothetical_doc)

        # Get relevant documents using hybrid search
        relevant_docs = retriever.get_relevant_documents(hypothetical_doc)
        relevant_docs = remove_duplicates(relevant_docs)

        # Remove the hypothetical document from results if it was added
        if use_hyde:
            relevant_docs = [
                doc for doc in relevant_docs if doc.page_content != hypothetical_doc
            ]

        # Setup and run RAG chain
        rag_chain = HH_setup_rag_chain(retriever, llm)
        response = HH_get_response(rag_chain, question)

        return response, relevant_docs
    except Exception as e:
        return f"Error in main function: {str(e)}", [], None
