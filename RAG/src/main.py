"""
Main entry point for the RAG insurance QA system.

Provides two commands:
  1. index: Load data and index into Qdrant
  2. query: Interactive search interface
"""

import argparse
from pathlib import Path

from .data_loader import load_jsonl
from .embeddings import GigaEmbedder
from .vector_store import create_client, create_collection, add_documents
from .retrieval import retrieve_top_k


def cmd_index(args):
    """
    Index a JSONL file into Qdrant.

    Steps:
      1. Load documents from JSONL.
      2. Initialize embedding model.
      3. Encode all documents.
      4. Create Qdrant collection.
      5. Add documents to collection.
    """
    data_path = Path(args.data)
    collection_name = args.collection

    print(f"\n=== Indexing: {data_path} ===\n")

    # Step 1: Load data
    print("Step 1: Loading documents from JSONL...")
    documents = load_jsonl(data_path)
    if args.limit_docs:
        documents = documents[: args.limit_docs]
        print(f"Limiting to first {len(documents)} documents (limit-docs).")
    print(f"Loaded {len(documents)} documents.\n")

    # Step 2: Initialize embedding model
    print("Step 2: Initializing Giga Embeddings (infgrad/stella-base-en-v2)...")
    embedding_model = GigaEmbedder(
        model_name="infgrad/stella-base-en-v2",
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Embedding model ready. Vector size: {embedding_model.vector_size}\n")

    # Step 3: Encode documents
    print("Step 3: Encoding documents to vectors...")
    texts_to_encode = [doc.get("input", "") for doc in documents]
    vectors = embedding_model.encode(texts_to_encode)
    print(f"Encoded {len(vectors)} vectors.\n")

    # Step 4: Create Qdrant client and collection
    print(f"Step 4: Creating Qdrant collection (Storage: {args.storage})...")
    client = create_client(path=args.storage)
    create_collection(client, collection_name, embedding_model.vector_size)
    print()

    # Step 5: Add documents
    print("Step 5: Adding documents to collection...")
    add_documents(client, collection_name, documents, vectors)
    print()

    print("✓ Indexing complete!\n")


def cmd_query(args):
    """
    Interactive query mode.

    For each query:
      1. Load the indexed collection.
      2. Encode the query.
      3. Retrieve top-k results.
      4. Display results with scores.
    """
    data_path = Path(args.data) if args.data else None
    collection_name = args.collection
    top_k = args.top_k

    print(f"\n=== Query Mode: Collection '{collection_name}' ===")
    print(f"Retrieving top {top_k} results per query.\n")

    # Initialize embedding model
    print("Initializing Giga Embeddings (infgrad/stella-base-en-v2)...")
    embedding_model = GigaEmbedder(
        model_name="infgrad/stella-base-en-v2",
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print()

    # Create Qdrant client (in-memory, but would need to load if persistent)
    print(f"Connecting to Qdrant (Storage: {args.storage})...\n")
    client = create_client(path=args.storage)

    # If data path provided, index it first
    if data_path and data_path.exists():
        # Only auto-index if collection is empty or user explicitly wants to?
        # For now, keep behavior but warn if persistent
        if args.storage != ":memory:":
             print("Warning: Auto-indexing with persistent storage might duplicate data if run multiple times.")
        
        print(f"Auto-indexing from {data_path}...")
        documents = load_jsonl(data_path)
        texts = [doc.get("input", "") for doc in documents]
        vectors = embedding_model.encode(texts)
        create_collection(client, collection_name, embedding_model.vector_size)
        add_documents(client, collection_name, documents, vectors)
        print()

    # Interactive loop
    print("Enter queries (empty to exit):\n")
    while True:
        query_text = input("Query: ").strip()
        if not query_text:
            print("Exiting.\n")
            break

        # Retrieve
        results = retrieve_top_k(
            client,
            collection_name,
            embedding_model,
            query_text,
            top_k=top_k,
        )

        # Display
        print("\n--- Top Results ---")
        if not results:
            print("No results found.")
        else:
            for i, doc in enumerate(results, start=1):
                score = doc.get("score", 0)
                doc_id = doc.get("id", "?")
                question = doc.get("input", doc.get("question", ""))
                answer = doc.get("output", doc.get("answer", ""))

                print(f"\n{i}. (ID: {doc_id}, Score: {score:.3f})")
                print(f"   Question: {question[:100]}...")
                print(f"   Answer: {answer[:100]}...")
        print()


def cmd_evaluate(args):
    """
    Evaluate retrieval performance.
    """
    from src.retrieval import evaluate_dataset

    data_path = Path(args.data)
    collection_name = args.collection
    top_k = args.top_k

    print(f"\n=== Evaluation Mode: Collection '{collection_name}' ===")
    print(f"Data: {data_path}")
    print(f"Top-K: {top_k}\n")

    # Load documents
    print("Loading documents...")
    documents = load_jsonl(data_path)
    if args.limit_docs:
        documents = documents[: args.limit_docs]
        print(f"Limiting to first {len(documents)} documents.")
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant (Storage: {args.storage})...")
    client = create_client(path=args.storage)

    # Initialize embedding model
    print("Initializing Giga Embeddings...")
    embedding_model = GigaEmbedder(
        model_name="infgrad/stella-base-en-v2",
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Run evaluation
    print("\nStarting evaluation loop...")
    metrics = evaluate_dataset(
        client,
        collection_name,
        embedding_model,
        documents,
        top_k=top_k,
    )

    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()


def build_parser():
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Insurance QA RAG system with Qdrant and Giga Embeddings"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    parser_index = subparsers.add_parser("index", help="Index JSONL data into Qdrant")
    parser_index.add_argument(
        "--data",
        required=True,
        help="Path to JSONL file (e.g., data/train.jsonl)",
    )
    parser_index.add_argument(
        "--collection",
        default="insurance_qa",
        help="Collection name in Qdrant (default: insurance_qa)",
    )
    parser_index.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16; lower if CPU is slow)",
    )
    parser_index.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Token max length for embedding (default: 256 to speed up CPU runs)",
    )
    parser_index.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional: only index the first N docs (useful for quick tests)",
    )
    parser_index.add_argument(
        "--storage",
        default=":memory:",
        help="Path to persistent storage folder (default: :memory:)",
    )
    parser_index.set_defaults(func=cmd_index)

    # Query command
    parser_query = subparsers.add_parser("query", help="Interactive query mode")
    parser_query.add_argument(
        "--storage",
        default=":memory:",
        help="Path to persistent storage folder (default: :memory:)",
    )
    parser_query.add_argument(
        "--data",
        default=None,
        help="Optional: JSONL file to auto-index before querying",
    )
    parser_query.add_argument(
        "--collection",
        default="insurance_qa",
        help="Collection name in Qdrant (default: insurance_qa)",
    )
    parser_query.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to retrieve (default: 3)",
    )
    parser_query.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding queries (default: 16)",
    )
    parser_query.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Token max length for embedding (default: 256)",
    )
    parser_query.set_defaults(func=cmd_query)

    # Evaluate command
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate retrieval performance")
    parser_eval.add_argument(
        "--data",
        required=True,
        help="Path to JSONL file to evaluate on (must be the same file used for indexing for Self-Retrieval test)",
    )
    parser_eval.add_argument(
        "--collection",
        default="insurance_qa",
        help="Collection name in Qdrant (default: insurance_qa)",
    )
    parser_eval.add_argument(
        "--storage",
        default=":memory:",
        help="Path to persistent storage folder (default: :memory:)",
    )
    parser_eval.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to check (default: 3)",
    )
    parser_eval.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16)",
    )
    parser_eval.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Token max length for embedding (default: 256)",
    )
    parser_eval.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional: only evaluate the first N docs",
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    return parser


def main():
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
