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
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from initialization import get_retriever, initialize_embeddings_and_llm


def flashRank_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def flashRank_create_compression_retriever(retriever):
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


def flashRank_setup_rag_chain(compression_retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def flashRank_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def flashRank_main(question):
    try:
        embedding, llm = flashRank_initialize_embeddings_and_llm()
        # initialize retriever
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(question)
        compression_retriever = flashRank_create_compression_retriever(retriever)
        compressed_docs = compression_retriever.invoke(question)
        rag_chain = flashRank_setup_rag_chain(compression_retriever, llm)

        # Generate response
        response = flashRank_get_response(rag_chain, question)
        return response, compressed_docs, docs
    except Exception as e:
        return f"Error in main function: {str(e)}"
