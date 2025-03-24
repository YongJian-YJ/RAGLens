import streamlit as st
from datasets import load_dataset
import pandas as pd
import time
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from session_persistence import PersistentStorage

storage = PersistentStorage()
storage.load_state()

# 1. Update new RAG pipeline here:
from evaluation_function.monoT5 import monoT5_main
from evaluation_function.default import default_main
from evaluation_function.bge import bge_main
from evaluation_function.reciprocal import reciprocal_main
from evaluation_function.HyDE_Hybrid import HH_main

# 2. Update the dictionary to reflect the new RAG pipeline:
RAG_TECHNIQUES = {
    "Default": default_main,
    "MonoT5": monoT5_main,
    "BGE Reranker": bge_main,
    "Reciprocal": reciprocal_main,
    "HyDE + Hybrid": HH_main,
}


# Ollama wrapper class
class Ollama(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "llama3.1:8b"


def load_evaluation_dataset(num_samples=2, user_csv=None):
    try:
        if user_csv is not None:
            df = pd.read_csv(user_csv)
            if {"question", "answer", "context"}.issubset(df.columns):
                max_rows = len(df)
                if num_samples > max_rows:
                    st.warning(
                        f"You have chosen to load {num_samples} samples, however, the maximum row of the dataset is only {max_rows}, hence only {max_rows} has been loaded."
                    )
                    num_samples = max_rows
                questions = df["question"].tolist()[:num_samples]
                answers = df["answer"].tolist()[:num_samples]
                contexts = df["context"].tolist()[:num_samples]
                return questions, answers, contexts
            else:
                st.error(
                    "CSV file must contain 'question', 'answer', and 'context' columns."
                )
                return None, None, None
        else:
            ragas_dataset = load_dataset("neural-bridge/rag-dataset-1200")
            if "train" in ragas_dataset:
                train_data = ragas_dataset["train"]
                max_rows = len(train_data)
                if num_samples > max_rows:
                    st.warning(
                        f"You have chosen to load {num_samples} samples, however, the maximum row of the dataset is only {max_rows}, hence only {max_rows} has been loaded."
                    )
                    num_samples = max_rows
                questions = [item["question"] for item in train_data][:num_samples]
                answers = [item["answer"] for item in train_data][:num_samples]
                contexts = [item["context"] for item in train_data][:num_samples]
                return questions, answers, contexts
            else:
                return None, None, None
    except Exception as e:
        return None, None, None


def run_evaluation(questions, answers, contexts, rag_technique, progress_bar=None):
    # custom_model = Ollama(model="llama3.3:70b", temperature=0, base_url="http://127.0.0.1:8883")
    custom_model = ChatOllama(model="llama3.2:latest")
    ollama_model = Ollama(model=custom_model)

    correctness_metric = GEval(
        name="Correctness",
        criteria="Evaluate whether the actual output semantically aligns with the expected output.",
        evaluation_steps=[
            "Ensure the actual output conveys the same meaning as the expected output.",
            "Penalize only if factual inaccuracies or contradictions are introduced.",
            "Consider context and completeness of the answer.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=ollama_model,
    )

    results = []
    total = len(questions)

    # Get the appropriate RAG technique function
    rag_function = RAG_TECHNIQUES[rag_technique]

    for i in range(total):
        if progress_bar:
            progress_bar.progress((i + 1) / total)

        try:
            result = rag_function(questions[i], contexts[i])

            # Check if the result is a tuple (valid output)
            if isinstance(result, tuple) and len(result) == 2:
                answer_chunk, context_chunk = result
            else:
                # If result is not a tuple, assume it's an error message
                st.error(f"Error processing question {i}: {result}")
                continue

            test_case = LLMTestCase(
                input=questions[i],
                actual_output=answer_chunk,
                expected_output=answers[i],
            )

            correctness_metric.measure(test_case)

            results.append(
                {
                    "Question": questions[i],
                    "Generated Answer": answer_chunk,
                    "Reference Answer": answers[i],
                    "Correctness Score": round(correctness_metric.score, 3),
                    "Reasoning": correctness_metric.reason,
                    "RAG Technique": rag_technique,
                }
            )

        except Exception as e:
            st.error(f"Unexpected error processing question {i}: {str(e)}")
            continue

    return pd.DataFrame(results)


def main():
    st.title("RAG Pipeline Evaluation Dashboard")
    try:
        if "dataset_loaded" not in st.session_state:
            st.session_state.dataset_loaded = False

        st.sidebar.header("Configuration")
        num_samples = st.sidebar.slider("Number of samples", 1, 100, 2)

        dataset_source = st.sidebar.radio(
            "Choose Dataset Source", ["Hugging Face Dataset", "Upload CSV File"]
        )

        user_csv = None
        if dataset_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file is not None:
                user_csv = uploaded_file

        selected_technique = st.sidebar.selectbox(
            "Select RAG Technique",
            options=list(RAG_TECHNIQUES.keys()),
            help="Choose the RAG technique to use for evaluation",
        )

        if st.sidebar.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                questions, answers, contexts = load_evaluation_dataset(
                    num_samples, user_csv
                )
                if questions:
                    st.session_state.questions = questions
                    st.session_state.answers = answers
                    st.session_state.contexts = contexts
                    st.session_state.dataset_loaded = True
                    st.success(f"Dataset loaded with {len(questions)} samples")
                else:
                    st.error("Failed to load dataset")

        if st.session_state.dataset_loaded:
            st.header("Dataset Preview")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Question": st.session_state.questions,
                        "Reference Answer": st.session_state.answers,
                    }
                )
            )

            if st.button("Run Evaluation"):
                progress_bar = st.progress(0)
                results_df = run_evaluation(
                    st.session_state.questions,
                    st.session_state.answers,
                    st.session_state.contexts,
                    selected_technique,
                    progress_bar,
                )
                st.header("Evaluation Results")
                st.dataframe(results_df)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


main()
