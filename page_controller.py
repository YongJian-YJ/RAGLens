# conda activate chat-with-website
# streamlit run page_controller.py

# conda activate chat-with-website && streamlit run page_controller.py

import streamlit as st

homepage = st.Page(
    page="views/homepage.py",
    title="Home Page",
    icon="🏡",
    default=True,
)
llmLingua_page = st.Page(
    page="views/llmLingua.py",
    title="LLMLingua",
    icon="👨‍👧‍👦",
)
parentChild_page = st.Page(
    page="views/parentChildRetriever.py",
    title="ParentChild Retriever",
    icon="👩‍👦",
)
reciprocal_page = st.Page(
    page="views/reciprocal.py",
    title="Reciprocal Retriever",
    icon="🔁",
)
monoT5_page = st.Page(
    page="views/monoT5.py",
    title="monoT5 reranker",
    icon="📊",
)
bge_page = st.Page(
    page="views/bge_reranker_large.py",
    title="BGE Reranker",
    icon="🚥",
)
hyde_with_hybridSearch_page = st.Page(
    page="views/hybridSearch.py",
    title="HyDE + Hybrid Search",
    icon="🔀",
)
comparison_page = st.Page(
    page="views/comparison.py",
    title="Comparison Page",
    icon="⚖",
)
evaluation_page = st.Page(
    page="views/evaluation.py",
    title="Evaluate Rag",
    icon="🔍",
)

pg = st.navigation(
    pages=[
        homepage,
        llmLingua_page,
        parentChild_page,
        reciprocal_page,
        monoT5_page,
        hyde_with_hybridSearch_page,
        bge_page,
        comparison_page,
        evaluation_page,
    ]
)

pg.run()
