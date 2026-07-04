import argparse

from rag_pipeline import ResponsibleRAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Responsible AI RAG assistant from the command line."
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to ask the Responsible AI RAG assistant.",
    )

    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of retrieved chunks to use.",
    )

    args = parser.parse_args()

    rag_pipeline = ResponsibleRAGPipeline()

    result = rag_pipeline.answer_question(
        question=args.question,
        n_results=args.n_results,
        n_initial_results=20,
        max_context_chunks=args.n_results,
        max_characters_per_chunk=1000,
    )

    print("=" * 100)
    print("Responsible AI RAG Assistant")
    print("=" * 100)
    print(f"Question: {result['question']}")
    print(f"Answer method: {result['answer_method']}")
    print(f"Preferred sources: {result['preferred_sources']}")
    print(f"Retrieved sources: {result['retrieved_sources']}")
    print(f"Retrieved chunks: {result['retrieved_chunks']}")
    print(f"Top distance: {result['top_distance']}")
    print("=" * 100)
    print(result["answer"])
    print("=" * 100)


if __name__ == "__main__":
    main()