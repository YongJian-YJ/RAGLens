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
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
from langchain_community.document_compressors import LLMLinguaCompressor
from langchain_openai import AzureChatOpenAI
from initialization import get_retriever, initialize_embeddings_and_llm


def default_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def default_setup_rag_chain(compression_retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def default_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def default_main(question):
    try:
        # initialize retriever
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(question)
        embedding, llm = default_initialize_embeddings_and_llm()
        rag_chain = default_setup_rag_chain(retriever, llm)

        # Generate response
        response = default_get_response(rag_chain, question)
        return response
    except Exception as e:
        return f"Error in main function: {str(e)}"
