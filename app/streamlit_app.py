from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# Make src/ importable when running Streamlit from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from rag_pipeline import ResponsibleRAGPipeline  # noqa: E402
from config import (  # noqa: E402
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    ENABLE_OPENAI_GENERATION,
    OPENAI_MODEL,
    OPENAI_API_KEY_AVAILABLE,
)


st.set_page_config(
    page_title="Responsible AI RAG Assistant",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource
def load_rag_pipeline() -> ResponsibleRAGPipeline:
    """
    Load the RAG pipeline once and cache it for the Streamlit session.
    """
    return ResponsibleRAGPipeline()


def display_retrieved_sources(retrieved_df: pd.DataFrame) -> None:
    """
    Display retrieved chunks in a readable Streamlit table.
    """
    if retrieved_df.empty:
        st.warning("No retrieved chunks were found.")
        return

    display_columns = [
        "rank",
        "distance",
        "source_id",
        "source_name",
        "publisher",
        "chunk_index",
        "word_count",
        "preferred_source",
    ]

    available_columns = [
        column for column in display_columns if column in retrieved_df.columns
    ]

    st.dataframe(
        retrieved_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("Responsible AI RAG Assistant")
    st.caption(
        "A source-grounded RAG prototype for EU AI Act, GDPR, and responsible AI risk-management questions."
    )

    with st.sidebar:
        st.header("Project Configuration")

        st.write("**Embedding model**")
        st.code(EMBEDDING_MODEL_NAME)

        st.write("**Chroma collection**")
        st.code(CHROMA_COLLECTION_NAME)

        st.write("**Configured LLM model**")
        st.code(OPENAI_MODEL)

        st.write("**OpenAI API key detected**")
        st.code(str(OPENAI_API_KEY_AVAILABLE))

        st.write("**OpenAI generation enabled**")
        st.code(str(ENABLE_OPENAI_GENERATION))

        st.divider()

        st.header("Responsible-Use Notice")
        st.info(
            "This assistant is an educational and portfolio-ready prototype. "
            "It does not provide legal advice. Real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals."
        )

    st.subheader("Ask a question")

    default_question = "What is the risk-based approach of the EU AI Act?"

    question = st.text_area(
        label="Question",
        value=default_question,
        height=100,
        help="Ask about the EU AI Act, GDPR, NIST AI RMF, high-risk AI, transparency, personal data, or AI risk management.",
    )

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        n_results = st.slider(
            "Retrieved chunks",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )

    with col_2:
        n_initial_results = st.slider(
            "Initial retrieval pool",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
        )

    with col_3:
        max_context_chunks = st.slider(
            "Context chunks used",
            min_value=3,
            max_value=8,
            value=5,
            step=1,
        )

    run_button = st.button("Ask the assistant", type="primary")

    if run_button:
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Loading RAG pipeline and retrieving relevant sources..."):
            try:
                rag_pipeline = load_rag_pipeline()

                result = rag_pipeline.answer_question(
                    question=question,
                    n_results=n_results,
                    n_initial_results=n_initial_results,
                    max_context_chunks=max_context_chunks,
                    max_characters_per_chunk=1000,
                )

            except Exception as error:
                st.error("The RAG pipeline could not complete the request.")
                st.write("**Error type:**", type(error).__name__)
                st.write("**Error message:**", str(error))
                return

        st.success("Answer generated.")

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

        with metric_col_1:
            st.metric("Answer method", result["answer_method"])

        with metric_col_2:
            st.metric("Retrieved chunks", result["retrieved_chunks"])

        with metric_col_3:
            st.metric(
                "Top distance",
                f"{result['top_distance']:.4f}"
                if result["top_distance"] is not None
                else "N/A",
            )

        with metric_col_4:
            st.metric(
                "Retrieved sources",
                len(result["retrieved_sources"]),
            )

        st.subheader("Answer")
        st.markdown(result["answer"])

        st.subheader("Retrieved Sources")
        display_retrieved_sources(result["retrieved_df"])

        with st.expander("Show retrieved context"):
            st.text(result["formatted_context"])

        with st.expander("Show responsible RAG prompt"):
            st.text(result["prompt"])


if __name__ == "__main__":
    main()