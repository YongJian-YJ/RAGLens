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
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from initialization import get_retriever, initialize_embeddings_and_llm


def reciprocal_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()
    return embedding, llm


def reciprocal_setup_rag_chain(compression_retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def reciprocal_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def reciprocal_main(question):
    try:
        # Initialize components
        embedding, llm = reciprocal_initialize_embeddings_and_llm()

        # Initialize retriever
        retriever = get_retriever()

        ##
        # create a chain that generate 4 queries from question
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        #         template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        # Please generate multiple search queries that are **directly relevant** to the original question and **maintain the core meaning** of the question. \n
        # Do not generate queries that are significantly different from the original question. The goal is to provide related search queries that will help gather **relevant information** for answering the question. \n
        # Output (4 queries):"""

        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries_chain = (
            prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.split("\n"))
        )

        # Chain for extracting relevant documents for all 4 queries
        retrieval_chain_rag_fusion = generate_queries_chain | retriever.map()
        results = retrieval_chain_rag_fusion.invoke({"question": question})
        # print("\nresults: ", results)
        # deduplicate the output from retrieval_chain_rag_fusion
        lst = []
        for ddxs in results:
            for ddx in ddxs:
                if ddx.page_content not in lst:
                    lst.append(ddx.page_content)

        # calculate score for each doc retrieved
        fused_scores = {}
        k = 60
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                # print('\n')
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # sort the result
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # print the queries generated from the original questions
        generated_queries = generate_queries_chain.invoke({"question": question})

        # print the score of the results
        stored_results = []
        for x in reranked_results:
            stored_results.append(f"Content: {x[0].page_content}, Score: {x[1]}")
        # print("\n".join(stored_results))

        # final RAG
        template = """Answer the following question based on this context:
        
                    {context}
    
                    Question: {question}

                    Please provide a response in plain text, avoiding the use of bullet points, numbered lists, or any special formatting.
                   """

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = prompt | llm | StrOutputParser()

        ##

        # Generate response
        response = final_rag_chain.invoke(
            {"context": reranked_results, "question": question}
        )
        # context_removed_numerical = [doc for doc, _ in reranked_results]
        return response, generated_queries, stored_results
    except Exception as e:
        return f"Error in main function: {str(e)}"
