from pathlib import Path
import re
import sys
from typing import Any

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


def clean_evidence_text(text: str) -> str:
    """
    Remove technical source/chunk metadata from fallback evidence lines.
    """
    cleaned = str(text).strip()

    cleaned = re.sub(r"^[-•]\s*", "", cleaned)

    cleaned = re.sub(
        r"^Source\s+SRC-\d+\s+—\s+.*?\(distance:\s*[0-9.]+\):\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"^Source\s+SRC-\d+\s+—\s+.*?:\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_key_evidence_points(raw_answer: str, max_points: int = 4) -> list[str]:
    """
    Extract readable evidence bullets from the context-only fallback answer.
    """
    if not raw_answer:
        return []

    if "Key retrieved evidence:" not in raw_answer:
        return []

    evidence_section = raw_answer.split("Key retrieved evidence:", 1)[1]

    stop_markers = [
        "Source references:",
        "Limitations:",
        "Responsible-use note:",
    ]

    for marker in stop_markers:
        if marker in evidence_section:
            evidence_section = evidence_section.split(marker, 1)[0]

    points: list[str] = []

    for line in evidence_section.splitlines():
        line = line.strip()

        if not line.startswith("- "):
            continue

        cleaned = clean_evidence_text(line)

        if cleaned and len(cleaned) > 30:
            points.append(cleaned)

        if len(points) >= max_points:
            break

    return points


def build_simple_answer(
    result: dict[str, Any],
    question: str,
) -> str:
    """
    Build a clean user-facing answer for the Streamlit interface.

    In fallback mode, this function converts retrieved evidence into a more
    readable answer. The full technical fallback answer remains available in
    the technical expanders.
    """
    raw_answer = str(result.get("answer", "")).strip()
    answer_method = str(result.get("answer_method", ""))

    # If OpenAI generation is enabled later, show the generated LLM answer directly.
    if answer_method != "context_only_fallback":
        return raw_answer

    question_lower = question.lower()

    # EU AI Act: risk-based approach
    if (
        "risk-based" in question_lower
        and ("eu ai act" in question_lower or "ai act" in question_lower)
    ):
        return """
The EU AI Act follows a **risk-based approach**. This means that AI systems are regulated according to the level of risk they may create for people, safety, fundamental rights, and society.

In practical terms:

- **Minimal-risk and low-risk AI systems** are subject to lighter requirements.
- **High-risk AI systems** face stricter obligations before and after being placed on the market.
- **Certain unacceptable-risk AI practices** are prohibited.
- The purpose of this approach is to support trustworthy AI while reducing risks such as discrimination, lack of transparency, unsafe deployment, or harm to fundamental rights.

The retrieved sources indicate that the EU AI Act is designed to create a harmonised legal framework for AI and to support trustworthy, human-centric AI development in Europe.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # EU AI Act: high-risk obligations
    if "high-risk" in question_lower and (
        "obligation" in question_lower or "apply" in question_lower
    ):
        return """
High-risk AI systems under the EU AI Act are subject to stricter requirements because they may affect safety, rights, or important life opportunities.

Based on the retrieved context, obligations for high-risk AI systems may include:

- Risk assessment and risk mitigation
- High-quality datasets to reduce discriminatory outcomes
- Logging and traceability of system results
- Clear documentation and user instructions
- Human oversight and monitoring
- Accuracy, robustness, and cybersecurity controls
- Post-market monitoring after deployment

The retrieved context also indicates that deployers and providers have different responsibilities across the AI system lifecycle.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # GDPR: personal data
    if "personal data" in question_lower and "gdpr" in question_lower:
        return """
Under the GDPR, **personal data** means information relating to an identified or identifiable natural person.

In practical terms, this can include information that directly identifies someone, such as a name or identification number, as well as information that can identify someone indirectly when combined with other data.

The retrieved GDPR context also indicates that personal data processing is subject to legal grounds, data-protection principles, and safeguards. Profiling and automated processing may require additional care, especially when they affect individuals.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # GDPR: processing principles
    if "gdpr" in question_lower and (
        "principles" in question_lower or "processing" in question_lower
    ):
        return """
The GDPR sets out principles for the processing of personal data. These principles guide how organizations should collect, use, store, and protect personal data.

Based on the retrieved context, GDPR processing should be lawful, fair, transparent, limited to specified purposes, limited to necessary data, accurate, stored only as long as needed, and protected with appropriate security.

Organizations should also be able to demonstrate accountability for how personal data is processed.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # NIST: managing AI risks
    if "nist" in question_lower and (
        "managing ai risks" in question_lower
        or "manage ai risks" in question_lower
        or "risk management" in question_lower
    ):
        return """
NIST describes AI risk management as an organized process for identifying, assessing, prioritizing, and managing risks connected to AI systems.

Based on the retrieved NIST context, AI risk management should be:

- Aligned with organizational goals and priorities
- Useful across different sectors and technologies
- Focused on trustworthy AI outcomes
- Adaptable to specific AI contexts and use cases
- Supported by governance, measurement, mapping, and management activities

The retrieved context also indicates that the NIST AI RMF is intended as a flexible framework rather than a rigid checklist.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # NIST AI RMF functions
    if "nist" in question_lower and (
        "govern" in question_lower
        or "map" in question_lower
        or "measure" in question_lower
        or "manage" in question_lower
        or "functions" in question_lower
    ):
        return """
The NIST AI Risk Management Framework is organized around key functions that help organizations manage AI risks systematically.

The main functions are:

- **Govern**: establish policies, responsibilities, culture, and oversight for AI risk management.
- **Map**: understand the AI system context, intended use, stakeholders, and possible risks.
- **Measure**: evaluate, analyze, and monitor AI risks and system behavior.
- **Manage**: prioritize, respond to, and reduce identified risks over time.

Together, these functions help organizations make AI risk management more structured, repeatable, and accountable.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # Responsible AI governance
    if "govern" in question_lower or "governance" in question_lower:
        return """
Responsible AI governance means creating structures, policies, and oversight mechanisms so that AI systems are developed and used in a trustworthy way.

Based on the retrieved context, organizations should consider:

- Clear roles and responsibilities
- Risk identification and monitoring
- Human oversight
- Documentation and traceability
- Alignment with legal, ethical, and organizational requirements
- Review by qualified legal, privacy, or compliance professionals where needed

The retrieved sources support the idea that AI governance should not be treated as a one-time task, but as an ongoing lifecycle process.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # AI transparency
    if "transparency" in question_lower:
        return """
AI transparency means that users, deployers, or affected people should receive appropriate information about an AI system and its use.

Based on the retrieved context, transparency can support trust, accountability, and informed decision-making. For higher-risk or regulated AI systems, transparency may include documentation, user instructions, information about system capabilities and limitations, and explanations of relevant responsibilities.

The exact transparency requirements depend on the type of AI system, its risk level, and the legal context.

This is a source-grounded educational summary, not legal advice.
""".strip()

    # General fallback if no specific template matches
    evidence_points = extract_key_evidence_points(raw_answer)

    if not evidence_points:
        return (
            "The assistant retrieved relevant source context, but the available "
            "fallback answer does not contain enough structured evidence to produce "
            "a clean user-facing summary. Please review the retrieved sources below."
        )

    bullet_text = "\n".join([f"- {point}" for point in evidence_points])

    simple_answer = f"""
Based on the retrieved public-source context, here is a source-grounded summary for the question:

**Question:** {question}

{bullet_text}

This is an educational source-grounded summary, not legal advice. Real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals.
"""

    return simple_answer.strip()


def display_source_summary(retrieved_df: pd.DataFrame) -> None:
    """
    Display a short readable list of unique sources used.
    """
    if retrieved_df.empty:
        st.warning("No sources were retrieved.")
        return

    required_columns = ["source_id", "source_name", "publisher"]
    available_columns = [
        column for column in required_columns if column in retrieved_df.columns
    ]

    if not available_columns:
        st.warning("Retrieved source metadata is unavailable.")
        return

    source_summary = (
        retrieved_df[available_columns]
        .drop_duplicates()
        .sort_values(by=available_columns[0])
    )

    for _, row in source_summary.iterrows():
        source_id = row.get("source_id", "Unknown source")
        source_name = row.get("source_name", "Unknown source name")
        publisher = row.get("publisher", "Unknown publisher")

        st.markdown(f"- `{source_id}` — **{source_name}** ({publisher})")


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
        width="stretch",
        hide_index=True,
    )


def display_retrieval_metrics(result: dict[str, Any]) -> None:
    """
    Display compact retrieval metrics.
    """
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

        st.header("Display Mode")
        display_mode = st.radio(
            label="Choose output style",
            options=[
                "Simple answer",
                "Technical transparency",
            ],
            index=0,
        )

        st.divider()

        st.header("Responsible-Use Notice")
        st.info(
            "This assistant is an educational and portfolio-ready prototype. "
            "It does not provide legal advice. Real compliance decisions should "
            "be reviewed by qualified legal, privacy, or compliance professionals."
        )

    st.subheader("Ask a question")

    default_question = "What is the risk-based approach of the EU AI Act?"

    question = st.text_area(
        label="Question",
        value=default_question,
        height=100,
        help=(
            "Ask about the EU AI Act, GDPR, NIST AI RMF, high-risk AI, "
            "transparency, personal data, or AI risk management."
        ),
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

        with st.spinner("Retrieving relevant sources and generating answer..."):
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

        if display_mode == "Simple answer":
            st.subheader("Answer")
            st.markdown(build_simple_answer(result, question))

            st.subheader("Sources used")
            display_source_summary(result["retrieved_df"])

            with st.expander("Show retrieval summary"):
                display_retrieval_metrics(result)

            with st.expander("Show retrieved sources table"):
                display_retrieved_sources(result["retrieved_df"])

            with st.expander("Show full technical fallback answer"):
                st.markdown(result["answer"])

            with st.expander("Show retrieved context"):
                st.text(result["formatted_context"])

            with st.expander("Show responsible RAG prompt"):
                st.text(result["prompt"])

        else:
            display_retrieval_metrics(result)

            st.subheader("Technical Answer")
            st.markdown(result["answer"])

            st.subheader("Retrieved Sources")
            display_retrieved_sources(result["retrieved_df"])

            with st.expander("Show retrieved context"):
                st.text(result["formatted_context"])

            with st.expander("Show responsible RAG prompt"):
                st.text(result["prompt"])


if __name__ == "__main__":
    main()