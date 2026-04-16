import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.vector_store import create_client
from src.embeddings import GigaEmbedder
from src.data_loader import load_jsonl
from src.retrieval import evaluate_dataset

def plot_comparison(baseline_metrics, rerank_metrics, save_path="reranking_impact.png"):
    # Extract metrics in order
    # Keys might be like "Precision@3", "Recall@3", etc.
    # We want to plot them in a specific order if possible, or just all of them.
    # Let's assume the keys are consistent.
    labels = list(baseline_metrics.keys()) 
    baseline_vals = [baseline_metrics[k] for k in labels]
    rerank_vals = [rerank_metrics[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Vector Only)')
    rects2 = ax.bar(x + width/2, rerank_vals, width, label='Reranked (BGE Base)')

    ax.set_ylabel('Score')
    ax.set_title('Impact of Reranking on Retrieval Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def run_comparison(
    data_path: str,
    collection_name: str,
    storage_path: str,
    limit_docs: int = 50,
    top_k: int = 3
):
    print(f"=== RAG Approach Comparison ===")
    print(f"Data: {data_path}")
    print(f"Limit: {limit_docs} docs")
    print(f"Top-K: {top_k}")
    
    # 1. Setup
    print("\nInitializing resources...")
    client = create_client(path=storage_path)
    embedding_model = GigaEmbedder(
        model_name="infgrad/stella-base-en-v2",
        batch_size=16,
        max_length=256
    )
    
    documents = load_jsonl(Path(data_path))
    if limit_docs:
        documents = documents[:limit_docs]
        print(f"Limiting to {len(documents)} documents.")

    # 2. Baseline Evaluation
    print("\n--- Running Baseline (Vector Search Only) ---")
    baseline_metrics = evaluate_dataset(
        client,
        collection_name,
        embedding_model,
        documents,
        top_k=top_k,
        use_rerank=False
    )

    # 3. Reranked Evaluation
    print("\n--- Running Reranked (Vector Search + BGE-M3) ---")
    rerank_metrics = evaluate_dataset(
        client,
        collection_name,
        embedding_model,
        documents,
        top_k=top_k,
        use_rerank=True
    )

    # 4. Comparison Output
    print("\n=== Results Comparison ===")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Reranked':<10} | {'Improvement':<10}")
    print("-" * 55)
    
    metrics_keys = [f"Precision@{top_k}", f"Recall@{top_k}", f"MRR@{top_k}", f"NDCG@{top_k}"]
    
    for key in metrics_keys:
        base_val = baseline_metrics.get(key, 0.0)
        rerank_val = rerank_metrics.get(key, 0.0)
        diff = rerank_val - base_val
        print(f"{key:<15} | {base_val:.4f}     | {rerank_val:.4f}     | {diff:+.4f}")

    # 5. Generate Plot
    plot_comparison(baseline_metrics, rerank_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RAG Approaches")
    parser.add_argument("--data", default="data/valid.jsonl", help="Path to validation data")
    parser.add_argument("--storage", default="qdrant_data", help="Path to Qdrant storage")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of docs")
    parser.add_argument("--k", type=int, default=3, help="Top-K for evaluation")
    
    args = parser.parse_args()
    
    run_comparison(
        data_path=args.data,
        collection_name="insurance_qa",
        storage_path=args.storage,
        limit_docs=args.limit,
        top_k=args.k
    )
