"""
Simple Qdrant vector store using in-memory storage.

No Docker needed; Qdrant runs as a Python library in-memory.
"""

from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def create_client(path: str = ":memory:"):
    """
    Create a Qdrant client.

    Args:
        path: Path to persistent storage or ":memory:" for in-memory.

    Returns:
        QdrantClient: Client connected to storage.
    """
    # Use ":memory:" for in-memory storage, no Docker needed
    # Or provide a path (e.g. "./qdrant_data") for persistence
    client = QdrantClient(path=path)
    return client


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create a collection in Qdrant to store vectors.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        vector_size: Size of each vector (e.g., 768 for most embeddings).
    """
    # Check if collection already exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists, skipping creation.")
        return
    except Exception:
        # Collection doesn't exist, create it
        pass

    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created with vector size {vector_size}.")


def add_documents(
    client: QdrantClient,
    collection_name: str,
    documents: List[Dict[str, Any]],
    vectors: List[List[float]],
):
    """
    Add documents and their vectors to the collection.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        documents: List of document dicts with metadata (e.g., question, answer).
        vectors: List of vectors corresponding to documents.
    """
    points = []
    for idx, (doc, vec) in enumerate(zip(documents, vectors)):
        point = qm.PointStruct(
            id=idx,
            vector=vec,
            payload=doc,  # Store the entire document as payload
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Added {len(points)} documents to collection '{collection_name}'.")


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int = 3,
) -> List[qm.ScoredPoint]:
    """
    Search for the top-k most similar documents.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        query_vector: Query vector (embedding of the question).
        top_k: Number of results to return.

    Returns:
        List of ScoredPoint objects with results and similarity scores.
    """
    # Use query_points as search is deprecated/removed in newer versions
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points
    return results

