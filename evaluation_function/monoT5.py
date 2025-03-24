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
from langchain.schema import Document
from langchain.schema.runnable import RunnableConfig
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from initialization import get_retriever, initialize_embeddings_and_llm
import torch

from typing import List, Any, Optional


class MonoT5Reranker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "castorini/monot5-base-msmarco"
        )
        self.model.to(self.device)

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        scores = []
        for doc in documents:
            # MonoT5 expects input in the format: "Query: [query] Document: [document] Relevant:"
            input_text = f"Query: {query} Document: {doc.page_content} Relevant:"

            # Tokenize and generate relevance score
            inputs = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=2,  # We only need "true" or "false"
                    min_length=1,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                # Get the probability of "true" prediction
                relevant_token_id = self.tokenizer.encode("true")[0]
                first_token_logits = outputs.scores[0][0]
                score = torch.softmax(first_token_logits, dim=-1)[
                    relevant_token_id
                ].item()
                scores.append(score)

        return scores

    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []

        scores = self.compute_scores(query, documents)
        scored_documents = list(zip(documents, scores))
        reranked_documents = [
            doc
            for doc, score in sorted(scored_documents, key=lambda x: x[1], reverse=True)
        ]
        return reranked_documents


class RerankedRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        return self.documents

    def get_relevant_documents(self, query):
        return self.documents


def monoT5_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def monoT5_get_retriever(context_text):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    documents = text_splitter.create_documents([context_text])

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever


def monoT5_setup_rag_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def monoT5_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def monoT5_main(question, context):
    try:
        # Initialize components
        embedding, llm = monoT5_initialize_embeddings_and_llm()
        retriever = monoT5_get_retriever(context)

        # Get initial documents
        initial_docs = retriever.get_relevant_documents(question)

        # Rerank documents using MonoT5 reranker
        reranker = MonoT5Reranker()
        reranked_docs = reranker.rerank(initial_docs, question)

        # Create retriever with reranked documents
        final_retriever = RerankedRetriever(reranked_docs)

        # Setup and run RAG chain
        rag_chain = monoT5_setup_rag_chain(final_retriever, llm)
        response = monoT5_get_response(rag_chain, question)

        return response, reranked_docs
    except Exception as e:
        return f"Error in main function: {str(e)}"
