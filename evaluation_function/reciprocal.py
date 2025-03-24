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


def reciprocal_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def reciprocal_get_retriever(context_text):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    # Split long text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    documents = text_splitter.create_documents([context_text])

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    return retriever


def reciprocal_setup_rag_chain(compression_retriever, llm):
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


def reciprocal_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def reciprocal_main(question, context):
    try:
        embedding, llm = reciprocal_initialize_embeddings_and_llm()
        retriever = reciprocal_get_retriever(context)

        template = """
            You are a highly capable AI assistant tasked with generating relevant search queries for a given question.

            **Instructions:**

            1. **Generate exactly 4 distinct search queries** that are highly relevant to answering the following question.
            2. **Avoid redundancy, ambiguity, or overly broad queries.** Focus on specific and clear keywords, phrasing the queries naturally as if they were real user searches.
            3. **Format your response as plain text,** listing each query on a new line, with no additional formatting, special characters, or punctuation beyond the queries themselves.
                * Query 1: <Query 1>
                * Query 2: <Query 2>
                * Query 3: <Query 3>
                * Query 4: <Query 4>
            4. **Important:** Ensure that the output format is simple, clean, and suitable for processing by a parser. The response should only include the queries in the format listed above and nothing else. No extra explanations, commentary, or text should be included.
            5. **Each query should follow a straightforward and natural phrasing,** asking for specific information related to the question. Avoid long, complicated, or vague phrases.

            **Question:** {question}
        """

        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries_chain = (
            prompt_rag_fusion
            | llm
            | StrOutputParser()
            | (lambda x: [query.strip() for query in x.split("\n") if query.strip()])
        )

        retrieval_chain_rag_fusion = generate_queries_chain | retriever.map()
        results = retrieval_chain_rag_fusion.invoke({"question": question})
        results = remove_duplicates(results)

        lst = []
        for ddxs in results:
            for ddx in ddxs:
                if ddx.page_content not in lst:
                    lst.append(ddx.page_content)

        fused_scores = {}
        k = 60
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)

                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0

                previous_score = fused_scores[doc_str]

                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        generated_queries = generate_queries_chain.invoke({"question": question})

        stored_results = []
        for x in reranked_results:
            stored_results.append(f"Content: {x[0].page_content}, Score: {x[1]}")

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

        final_rag_chain = prompt | llm | StrOutputParser()

        ##

        # Generate response
        try:
            response = final_rag_chain.invoke(
                {"context": reranked_results, "question": question}
            )

        except Exception as e:
            print(f"Error parsing final response: {e}\n")
            # Attempt to clean the output (example: remove special characters)
            cleaned_output = "".join(c for c in str(e) if c.isalnum() or c.isspace())

        context_removed_numerical = [doc for doc, _ in reranked_results]

        return response, context_removed_numerical
    except Exception as e:
        return f"Error in main function: {str(e)}"
