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
from langchain.schema.runnable import RunnableConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
from typing import List, Any, Optional
from initialization import get_retriever, initialize_embeddings_and_llm

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


class BGEReranker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-large"
        )
        self.model.to(self.device)

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        pairs = []
        for doc in documents:
            pairs.append({"text_pairs": [[query, doc.page_content]]})

        scores = []
        for pair in pairs:
            inputs = self.tokenizer(
                pair["text_pairs"][0][0],
                pair["text_pairs"][0][1],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores.append(outputs.logits.squeeze().cpu().item())

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


def bge_initialize_embeddings_and_llm():
    embedding, llm = initialize_embeddings_and_llm()

    return embedding, llm


def bge_get_retriever(context_text):
    if not isinstance(context_text, str):
        context_text = str(context_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    documents = text_splitter.create_documents([context_text])

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    return retriever


def bge_setup_rag_chain(retriever, llm):
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


def bge_get_response(rag_chain, question):
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def bge_main(question, context):
    try:
        # Initialize components
        embedding, llm = bge_initialize_embeddings_and_llm()
        retriever = bge_get_retriever(context)

        # Get initial documents
        initial_docs = retriever.get_relevant_documents(question)
        initial_docs = remove_duplicates(initial_docs)

        # Rerank documents using BGE reranker
        reranker = BGEReranker()
        reranked_docs = reranker.rerank(initial_docs, question)

        # Create retriever with reranked documents
        final_retriever = RerankedRetriever(reranked_docs)

        # Setup and run RAG chain
        rag_chain = bge_setup_rag_chain(final_retriever, llm)
        response = bge_get_response(rag_chain, question)

        return response, reranked_docs
    except Exception as e:
        return f"Error in main function: {str(e)}"
