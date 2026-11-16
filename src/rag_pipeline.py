import os
import re
from typing import Any

import pandas as pd
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from config import (
    VECTOR_STORE_DIR,
    MULTISOURCE_CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    OPENAI_MODEL,
    ENABLE_OPENAI_GENERATION,
)


def clean_metadata_value(value: Any) -> str:
    """
    Convert metadata values into Chroma-safe primitive values.

    Same logic as notebook 05.
    """
    if pd.isna(value):
        return ""
    return str(value)


def build_chroma_metadata(row: pd.Series, embedding_model_name: str) -> dict:
    """
    Create the clean metadata dictionary for one chunk.

    Same fields as build_chroma_metadata() in notebook 05, so a rebuilt
    collection is identical to the one created in the notebook.
    """
    return {
        "source_id": clean_metadata_value(row["source_id"]),
        "source_name": clean_metadata_value(row["source_name"]),
        "publisher": clean_metadata_value(row["publisher"]),
        "source_type": clean_metadata_value(row["source_type"]),
        "main_topic": clean_metadata_value(row["main_topic"]),
        "url": clean_metadata_value(row["url"]),
        "planned_use": clean_metadata_value(row["planned_use"]),
        "raw_filename": clean_metadata_value(row["raw_filename"]),
        "processed_text_file": clean_metadata_value(row["processed_text_file"]),
        "chunk_index": int(row["chunk_index"]),
        "word_count": int(row["word_count"]),
        "character_count": int(row["character_count"]),
        "embedding_model": embedding_model_name,
    }


def clean_text_for_prompt(text: str) -> str:
    """
    Clean text before inserting it into a RAG prompt.
    """
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_text(text: str, max_characters: int = 1200) -> str:
    """
    Truncate long text to keep prompts and fallback answers readable.
    """
    cleaned = clean_text_for_prompt(text)

    if len(cleaned) <= max_characters:
        return cleaned

    return cleaned[:max_characters].rstrip() + "..."


def infer_preferred_sources(question: str) -> list[str]:
    """
    Infer preferred source IDs from the user question.

    This is a transparent heuristic used to improve retrieval behavior.
    It does not replace semantic search.
    """
    question_lower = question.lower()

    gdpr_terms = [
        "gdpr",
        "personal data",
        "data subject",
        "lawful basis",
        "processing personal data",
        "data protection",
        "controller",
        "processor",
    ]

    nist_terms = [
        "nist",
        "ai rmf",
        "risk management framework",
        "govern",
        "map",
        "measure",
        "manage",
        "trustworthy ai",
    ]

    ai_act_terms = [
        "ai act",
        "high-risk",
        "high risk",
        "prohibited",
        "transparency",
        "general-purpose ai",
        "gpai",
        "risk-based",
        "risk based",
        "provider obligations",
        "deployer obligations",
    ]

    if any(term in question_lower for term in gdpr_terms):
        return ["SRC-003"]

    if any(term in question_lower for term in nist_terms):
        return ["SRC-004", "SRC-005"]

    if any(term in question_lower for term in ai_act_terms):
        return ["SRC-001", "SRC-002"]

    return []


class ResponsibleRAGPipeline:
    """
    Reusable Responsible AI RAG pipeline.

    The pipeline:
    1. loads the embedding model
    2. connects to the local Chroma vector store
       (building it from multisource_chunks.csv if it does not exist yet)
    3. retrieves source-aware chunks
    4. builds a responsible RAG prompt
    5. optionally calls OpenAI
    6. falls back to context-only answer generation
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = CHROMA_COLLECTION_NAME,
        vector_store_dir=VECTOR_STORE_DIR,
        openai_model: str = OPENAI_MODEL,
        enable_openai_generation: bool = ENABLE_OPENAI_GENERATION,
        chunks_path=MULTISOURCE_CHUNKS_PATH,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.vector_store_dir = vector_store_dir
        self.openai_model = openai_model
        self.enable_openai_generation = enable_openai_generation
        self.chunks_path = chunks_path

        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_dir)
        )

        self.collection = self._load_or_build_collection()

    def _load_or_build_collection(self):
        """
        Return the Chroma collection, building it from the chunks CSV if it
        does not exist (e.g. on Streamlit Community Cloud, where
        data/vector_store/ is not in the repository).

        An existing collection is never deleted. If it exists but is empty
        or incomplete (e.g. an interrupted build), only the missing chunks
        are added.
        """
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        except (NotFoundError, ValueError):
            collection = None

        if collection is not None:
            if not os.path.exists(self.chunks_path):
                return collection

            expected_chunks = len(pd.read_csv(self.chunks_path, usecols=["chunk_id"]))
            if collection.count() >= expected_chunks:
                return collection

        return self.build_collection_from_chunks()

    def build_collection_from_chunks(self, batch_size: int = 100):
        """
        Build (or complete) the multi-source collection from
        multisource_chunks.csv, matching notebook 05: same embedding model,
        normalized embeddings, chunk IDs, metadata fields and collection
        metadata.
        """
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                f"Chroma collection '{self.collection_name}' was not found in "
                f"{self.vector_store_dir} and the chunks file needed to build "
                f"it is missing: {self.chunks_path}"
            )

        chunks_df = pd.read_csv(self.chunks_path)

        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": (
                    "Multi-source Responsible AI RAG chunks from EU AI Act, "
                    "GDPR, and NIST AI RMF sources"
                ),
                "embedding_model": self.embedding_model_name,
                "source_count": str(chunks_df["source_id"].nunique()),
                "chunk_count": str(len(chunks_df)),
            },
        )

        existing_ids = set(collection.get(include=[])["ids"])
        missing_df = chunks_df[
            ~chunks_df["chunk_id"].astype(str).isin(existing_ids)
        ].reset_index(drop=True)

        if missing_df.empty:
            return collection

        print(
            f"Building Chroma collection '{self.collection_name}': "
            f"embedding {len(missing_df)} chunks from {self.chunks_path}"
        )

        embeddings = self.embedding_model.encode(
            missing_df["chunk_text"].astype(str).tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for start_idx in range(0, len(missing_df), batch_size):
            end_idx = min(start_idx + batch_size, len(missing_df))
            batch_df = missing_df.iloc[start_idx:end_idx]

            collection.add(
                ids=batch_df["chunk_id"].astype(str).tolist(),
                documents=batch_df["chunk_text"].astype(str).tolist(),
                metadatas=[
                    build_chroma_metadata(row, self.embedding_model_name)
                    for _, row in batch_df.iterrows()
                ],
                embeddings=embeddings[start_idx:end_idx].tolist(),
            )

        print(
            f"Chroma collection '{self.collection_name}' ready with "
            f"{collection.count()} chunks."
        )

        return collection

    def search(
        self,
        query: str,
        n_results: int = 5,
        n_initial_results: int = 20,
    ) -> pd.DataFrame:
        """
        Retrieve relevant chunks from the multi-source Chroma vector store.
        """
        preferred_sources = infer_preferred_sources(query)

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_initial_results,
            include=["documents", "metadatas", "distances"],
        )

        records = []

        ids = query_results["ids"][0]
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            source_id = metadata.get("source_id", "")

            is_preferred_source = (
                source_id in preferred_sources if preferred_sources else False
            )

            records.append(
                {
                    "rank": rank,
                    "distance": float(distance),
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "source_name": metadata.get("source_name", ""),
                    "publisher": metadata.get("publisher", ""),
                    "source_type": metadata.get("source_type", ""),
                    "main_topic": metadata.get("main_topic", ""),
                    "url": metadata.get("url", ""),
                    "chunk_index": metadata.get("chunk_index", ""),
                    "word_count": metadata.get("word_count", ""),
                    "preferred_source": is_preferred_source,
                    "retrieved_text": document,
                }
            )

        retrieved_df = pd.DataFrame(records)

        if retrieved_df.empty:
            return retrieved_df

        preferred_df = retrieved_df[retrieved_df["preferred_source"]].copy()

        if len(preferred_df) >= n_results:
            final_df = preferred_df.sort_values("distance").head(n_results)

        elif len(preferred_df) > 0:
            remaining_df = retrieved_df[
                ~retrieved_df["chunk_id"].isin(preferred_df["chunk_id"])
            ].copy()

            final_df = pd.concat(
                [
                    preferred_df.sort_values("distance"),
                    remaining_df.sort_values("distance"),
                ],
                ignore_index=True,
            ).head(n_results)

        else:
            final_df = retrieved_df.sort_values("distance").head(n_results)

        final_df = final_df.reset_index(drop=True)
        final_df["rank"] = range(1, len(final_df) + 1)

        return final_df

    def format_retrieved_context(
        self,
        retrieved_df: pd.DataFrame,
        max_chunks: int = 5,
        max_characters_per_chunk: int = 1200,
    ) -> str:
        """
        Format retrieved chunks into a structured context block.
        """
        context_blocks = []

        for _, row in retrieved_df.head(max_chunks).iterrows():
            retrieved_text = truncate_text(
                row.get("retrieved_text", ""),
                max_characters=max_characters_per_chunk,
            )

            context_block = f"""
[Retrieved Chunk {row.get("rank", "")}]
Source ID: {row.get("source_id", "")}
Source Name: {row.get("source_name", "")}
Publisher: {row.get("publisher", "")}
Chunk Index: {row.get("chunk_index", "")}
Retrieval Distance: {row.get("distance", "")}
URL: {row.get("url", "")}

Text:
{retrieved_text}
""".strip()

            context_blocks.append(context_block)

        separator = "\n\n" + ("-" * 80) + "\n\n"
        return separator.join(context_blocks)

    def build_prompt(
        self,
        question: str,
        formatted_context: str,
    ) -> str:
        """
        Build a responsible RAG prompt.
        """
        prompt = f"""
You are a Responsible AI knowledge-support assistant.

You answer questions using only the retrieved public-source context provided below.

Important rules:
1. Do not provide legal advice.
2. Do not claim certainty beyond the retrieved context.
3. Do not use outside knowledge.
4. If the retrieved context is insufficient, say that the available sources do not support a confident answer.
5. Cite the source IDs and chunk indices used in the answer.
6. Mention that real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals.
7. Keep the answer clear, structured, and source-aware.

User question:
{question}

Retrieved context:
{formatted_context}

Answer format:

1. Short answer
Provide a concise answer grounded only in the retrieved context.

2. Source-grounded evidence
List the most relevant source IDs and chunk indices, with a short explanation of what each supports.

3. Limitations
State any limits of the retrieved context.

4. Responsible-use note
Clearly state that this is not legal advice.
""".strip()

        return prompt

    def estimate_retrieval_confidence(
        self,
        retrieved_df: pd.DataFrame,
    ) -> str:
        """
        Estimate retrieval confidence from distance, source coverage, and
        source-aware matching.

        This is a lightweight project heuristic, not a formal legal-confidence score.
        """
        if retrieved_df.empty:
            return "no_context"

        top_distance = float(retrieved_df["distance"].min())
        unique_sources = retrieved_df["source_id"].nunique()

        preferred_matches = 0
        if "preferred_source" in retrieved_df.columns:
            preferred_matches = int(retrieved_df["preferred_source"].sum())

        # Strongest case: close semantic match and at least one relevant source.
        if top_distance <= 0.60 and unique_sources >= 1:
            return "moderate_to_good"

        # Good practical case: source-aware retrieval strongly matched the expected source.
        if preferred_matches >= 3 and top_distance <= 0.90:
            return "source_matched_moderate"

        # Acceptable semantic match.
        if top_distance <= 0.80 and unique_sources >= 1:
            return "moderate"

        # Weak but still retrieved context.
        return "low"

    def generate_context_only_answer(
        self,
        question: str,
        retrieved_df: pd.DataFrame,
        max_evidence_items: int = 4,
    ) -> str:
        """
        Generate a structured fallback answer without calling an external LLM.
        """
        if retrieved_df.empty:
            return f"""
Question:
{question}

Context-grounded answer:
The available retrieved sources do not provide enough information to answer this question confidently.

Limitations:
No relevant context was retrieved from the vector store.

Responsible-use note:
This prototype does not provide legal advice. Real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals.
""".strip()

        confidence = self.estimate_retrieval_confidence(retrieved_df)

        evidence_lines = []

        for _, row in retrieved_df.head(max_evidence_items).iterrows():
            snippet = truncate_text(
                row["retrieved_text"],
                max_characters=350,
            )

            evidence_lines.append(
                f"- Source {row['source_id']} — {row['source_name']}, "
                f"chunk {row['chunk_index']} "
                f"(distance: {row['distance']:.4f}): {snippet}"
            )

        source_reference_lines = []

        for _, row in retrieved_df.head(max_evidence_items).iterrows():
            source_reference_lines.append(
                f"- {row['source_id']} — {row['source_name']} "
                f"({row['publisher']}), chunk {row['chunk_index']}: {row['url']}"
            )

        answer = f"""
Question:
{question}

Context-grounded answer:
The retrieved context contains potentially relevant information for this question. The answer below should be treated as a source-grounded evidence summary, not as legal advice.

Retrieval confidence:
{confidence}

Key retrieved evidence:
{chr(10).join(evidence_lines)}

Source references:
{chr(10).join(source_reference_lines)}

Limitations:
This fallback answer is generated only from retrieved text snippets and does not perform full legal interpretation. The retrieved context may not cover all relevant legal provisions, exceptions, definitions, or implementation guidance.

Responsible-use note:
This prototype does not provide legal advice. Real compliance decisions should be reviewed by qualified legal, privacy, or compliance professionals.
""".strip()

        return answer

    def generate_openai_answer(
        self,
        prompt: str,
    ) -> str | None:
        """
        Optional OpenAI answer generation.

        Returns None if generation is disabled, unavailable, or fails.
        """
        if not self.enable_openai_generation:
            return None

        if not os.getenv("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.responses.create(
                model=self.openai_model,
                input=prompt,
            )

            return response.output_text

        except Exception as error:
            print("OpenAI generation failed. Falling back to context-only answer.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error message: {error}")
            return None

    def answer_question(
        self,
        question: str,
        n_results: int = 5,
        n_initial_results: int = 20,
        max_context_chunks: int = 5,
        max_characters_per_chunk: int = 1200,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.
        """
        retrieved_df = self.search(
            query=question,
            n_results=n_results,
            n_initial_results=n_initial_results,
        )

        formatted_context = self.format_retrieved_context(
            retrieved_df=retrieved_df,
            max_chunks=max_context_chunks,
            max_characters_per_chunk=max_characters_per_chunk,
        )

        prompt = self.build_prompt(
            question=question,
            formatted_context=formatted_context,
        )

        llm_answer = self.generate_openai_answer(prompt)

        if llm_answer:
            answer_method = "openai_responses_api"
            answer = llm_answer
        else:
            answer_method = "context_only_fallback"
            answer = self.generate_context_only_answer(
                question=question,
                retrieved_df=retrieved_df,
            )

        result = {
            "question": question,
            "preferred_sources": infer_preferred_sources(question),
            "answer_method": answer_method,
            "retrieved_chunks": len(retrieved_df),
            "retrieved_sources": (
                retrieved_df["source_id"].unique().tolist()
                if not retrieved_df.empty
                else []
            ),
            "top_distance": (
                float(retrieved_df["distance"].min())
                if not retrieved_df.empty
                else None
            ),
            "formatted_context": formatted_context,
            "prompt": prompt,
            "answer": answer,
            "retrieved_df": retrieved_df,
        }

        return result