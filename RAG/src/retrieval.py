"""
Retrieval and evaluation functions for RAG system.

Handles retrieving documents from Qdrant and computing metrics like Precision@k, Recall@k, MRR@k.
"""

from typing import List, Dict, Any, Optional
from .vector_store import search as qdrant_search
from .embeddings import GigaEmbedder
from FlagEmbedding import FlagReranker

# Initialize reranker globally to avoid reloading (lazy load could be better but this is simple)
# use_fp16=False for CPU compatibility
try:
    # Swapped to lighter model 'BAAI/bge-reranker-base' for better CPU compatibility
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)
    print("Reranker BAAI/bge-reranker-base initialized.")
except Exception as e:
    print(f"Warning: Could not initialize reranker: {e}")
    reranker = None


def search_with_rerank(
    client,
    collection_name: str,
    query_text: str,
    embedding_model: GigaEmbedder,
    top_k: int = 3,
    initial_top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using Vector Search + BGE-M3 Reranking.
    """
    if reranker is None:
        raise RuntimeError("Reranker not initialized.")

    # 1. Initial Retrieval (Fetch more candidates)
    # We use the existing retrieve_top_k logic but just get the raw results first
    query_vector = embedding_model.encode([query_text])[0]
    hits = qdrant_search(client, collection_name, query_vector, top_k=initial_top_k)
    
    if not hits:
        return []

    # 2. Prepare Pairs for Reranker [Query, Answer]
    # We assume 'output' (answer) is stored in payload based on previous code
    pairs = []
    for hit in hits:
        answer = hit.payload.get('output', hit.payload.get('answer', ''))
        pairs.append([query_text, answer])
    
    # 3. Compute Scores
    scores = reranker.compute_score(pairs)
    
    # Handle case where scores is a single float (if only 1 pair)
    if isinstance(scores, float):
        scores = [scores]

    # 4. Attach scores and Sort
    ranked_results = []
    for hit, score in zip(hits, scores):
        doc = {
            "id": hit.id,
            "score": score, # Use reranker score
            "initial_score": hit.score,
            **hit.payload,
        }
        ranked_results.append(doc)
    
    # Sort descending by the new BGE score
    ranked_results.sort(key=lambda x: x['score'], reverse=True)
    
    return ranked_results[:top_k]


def retrieve_top_k(
    client,
    collection_name: str,
    embedding_model: GigaEmbedder,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant documents for a query.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        embedding_model: EmbeddingModel instance for encoding the query.
        query: Query text (question).
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: id, score, question, answer, input, output.
    """
    # Encode the query into a vector
    query_vector = embedding_model.encode([query])[0]

    # Search in Qdrant
    results = qdrant_search(client, collection_name, query_vector, top_k=top_k)

    # Format results
    retrieved = []
    for result in results:
        doc = {
            "id": result.id,
            "score": result.score,
            **result.payload,  # Include all metadata (question, answer, input, output, etc.)
        }
        retrieved.append(doc)

    return retrieved


def compute_precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Compute Precision@k: what fraction of top-k results are relevant?

    Formula: (# relevant in top-k) / k

    Args:
        retrieved_ids: List of document IDs retrieved.
        relevant_ids: List of relevant document IDs.
        k: Cutoff (consider only top-k).

    Returns:
        Precision@k as a float between 0 and 1.
    """
    if k == 0:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = len(set(top_k_ids) & set(relevant_ids))
    return hits / k


def compute_recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Compute Recall@k: what fraction of relevant documents are in top-k?

    Formula: (# relevant in top-k) / (# relevant total)

    Args:
        retrieved_ids: List of document IDs retrieved.
        relevant_ids: List of relevant document IDs.
        k: Cutoff (consider only top-k).

    Returns:
        Recall@k as a float between 0 and 1.
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    hits = len(set(top_k_ids) & set(relevant_ids))
    return hits / len(relevant_ids)


def compute_mrr_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Compute MRR@k: Mean Reciprocal Rank (average rank of first relevant item).

    Formula: 1 / rank of first relevant item (if found in top-k, else 0)

    Args:
        retrieved_ids: List of document IDs retrieved.
        relevant_ids: List of relevant document IDs.
        k: Cutoff (consider only top-k).

    Returns:
        MRR@k as a float between 0 and 1.
    """
    top_k_ids = retrieved_ids[:k]
    for rank, doc_id in enumerate(top_k_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


import math

def evaluate_dataset(
    client,
    collection_name: str,
    embedding_model: GigaEmbedder,
    documents: List[Dict[str, Any]],
    top_k: int = 3,
    use_rerank: bool = False
) -> Dict[str, float]:
    """
    Evaluate retrieval performance on a dataset (Self-Retrieval).

    Args:
        client: QdrantClient.
        collection_name: Collection name.
        embedding_model: Model to encode queries.
        documents: List of documents.
        top_k: Number of results to check.
        use_rerank: Whether to use BGE-M3 reranker.

    Returns:
        Dict with average Precision@k, Recall@k, MRR@k, NDCG@k.
    """
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    n = len(documents)

    print(f"Evaluating on {n} documents (Rerank={use_rerank})...")
    
    for i, doc in enumerate(documents):
        query = doc.get("input", "")
        ground_truth_answer = doc.get("output", "").strip()
        
        # Retrieve
        if use_rerank:
            results = search_with_rerank(client, collection_name, query, embedding_model, top_k=top_k)
        else:
            results = retrieve_top_k(client, collection_name, embedding_model, query, top_k=top_k)
        
        # Check relevance by Answer String Match
        is_relevant_list = []
        for res in results:
            retrieved_answer = res.get("output", res.get("answer", "")).strip()
            is_match = (retrieved_answer == ground_truth_answer) and (len(ground_truth_answer) > 0)
            is_relevant_list.append(is_match)
            
        # Compute metrics for this query
        hits = sum(is_relevant_list)
        
        # Precision@K: (Relevant items found / K)
        p_k = hits / top_k if top_k > 0 else 0.0
        
        # Recall@K: (1 if correct answer is in Top-K, else 0)
        r_k = 1.0 if hits > 0 else 0.0
        
        # MRR@K: (1/Rank of the first correct answer)
        mrr_k = 0.0
        for rank, is_rel in enumerate(is_relevant_list, start=1):
            if is_rel:
                mrr_k = 1.0 / rank
                break
        
        # NDCG@K: 1 / log2(rank + 1) if found, else 0
        ndcg_k = 0.0
        for rank, is_rel in enumerate(is_relevant_list, start=1):
            if is_rel:
                ndcg_k = 1.0 / math.log2(rank + 1)
                break

        total_precision += p_k
        total_recall += r_k
        total_mrr += mrr_k
        total_ndcg += ndcg_k
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n} queries...")

    return {
        f"Precision@{top_k}": total_precision / n,
        f"Recall@{top_k}": total_recall / n,
        f"MRR@{top_k}": total_mrr / n,
        f"NDCG@{top_k}": total_ndcg / n,
    }

