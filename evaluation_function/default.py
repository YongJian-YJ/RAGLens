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
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from llmlingua import PromptCompressor
from initialization import get_retriever, initialize_embeddings_and_llm
import sys

sys.stderr = open(os.devnull, "w")


def remove_duplicates(documents):
    seen_content = set()
    unique_docs = []

    for doc in documents:
        if isinstance(doc, Document):
            content = doc.page_content
        else:
            content = str(doc)

        if content not in seen_content:
            seen_content.add(content)
            unique_docs.append(doc)

    return unique_docs


def default_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def default_get_retriever(context_text):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    # Split long text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    documents = text_splitter.create_documents([context_text])

    documents = remove_duplicates(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 7}  # Return top 3 most relevant chunks
    )
    return retriever


def default_setup_rag_chain(compression_retriever, llm):
    template = """Answer the following question based solely on the provided context.
    
                Context:
                {context}
                
                Question:
                {question}
                
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


def default_main(question, context):
    try:
        # Initialize components
        embedding, llm = default_initialize_embeddings_and_llm()
        retriever = default_get_retriever(context)
        context = retriever.get_relevant_documents(question)
        context = remove_duplicates(context)
        rag_chain = default_setup_rag_chain(retriever, llm)

        # Generate response
        response = default_get_response(rag_chain, question)

        return response, context
    except Exception as e:
        return f"Error in main function: {str(e)}"
