import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.storage import InMemoryStore
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from initialization import (
    get_retriever,
    create_doc,
    initialize_embeddings_and_llm,
)


def deduplicate_documents(documents):
    """
    Deduplicate a list of Document objects based on their page_content.
    Returns a new list with only unique documents.
    """
    seen_contents = set()
    unique_docs = []
    for doc in documents:
        # Extract the page content (strip whitespace for consistency)
        content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc)
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append(doc)
    return unique_docs


def parentChild_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()
    return embedding, llm


def parentChild_setup_rag_chain(compression_retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def parentChild_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def pc_initialize_retriever():
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    vectorstore = get_retriever()  # Load or create the vectorstore
    vectorstore = vectorstore.vectorstore
    docs = create_doc()
    store = InMemoryStore()

    # Initialize ParentDocumentRetriever only once
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )
    retriever.add_documents(docs)  # Add documents only once

    return retriever, vectorstore


def parentChild_main(question):
    try:
        # Initialize components
        embedding, llm = parentChild_initialize_embeddings_and_llm()

        # get retriever and vectorstore
        retriever, vectorstore = pc_initialize_retriever()

        # Prepare child_docs
        raw_child_docs = vectorstore.similarity_search(question)
        # Deduplicate the child documents
        child_docs = deduplicate_documents(raw_child_docs)

        # Set up the RAG chain
        rag_chain = parentChild_setup_rag_chain(retriever, llm)

        # Prepare parent_docs
        parent_docs = retriever.invoke(question)

        # Get response
        response = parentChild_get_response(rag_chain, question)

        return response, parent_docs, child_docs
    except Exception as e:
        return f"Error in main function: {str(e)}", []
