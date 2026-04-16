import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import time

from src.vector_store import create_client
from src.embeddings import GigaEmbedder
from src.data_loader import load_jsonl
from src.retrieval import evaluate_dataset

def run_benchmark(
    data_path: str,
    collection_name: str,
    storage_path: str,
    k_values: List[int],
    limit_docs: int = None,
    batch_size: int = 16,
    max_length: int = 256
):
    """
    Run evaluation for multiple k values and plot results.
    """
    print(f"=== Running Benchmark ===")
    print(f"Data: {data_path}")
    print(f"Storage: {storage_path}")
    print(f"K values: {k_values}")
    
    # 1. Setup
    print("\nInitializing resources...")
    client = create_client(path=storage_path)
    embedding_model = GigaEmbedder(
        model_name="infgrad/stella-base-en-v2",
        batch_size=batch_size,
        max_length=max_length
    )
    
    documents = load_jsonl(Path(data_path))
    if limit_docs:
        documents = documents[:limit_docs]
        print(f"Limiting to {len(documents)} documents.")

    # 2. Loop through K values
    results = {
        "k": [],
        "precision": [],
        "recall": [],
        "mrr": [],
        "time": []
    }

    for k in k_values:
        print(f"\n--- Evaluating @ k={k} ---")
        start_time = time.time()
        
        metrics = evaluate_dataset(
            client,
            collection_name,
            embedding_model,
            documents,
            top_k=k
        )
        
        elapsed = time.time() - start_time
        
        results["k"].append(k)
        results["precision"].append(metrics[f"Precision@{k}"])
        results["recall"].append(metrics[f"Recall@{k}"])
        results["mrr"].append(metrics[f"MRR@{k}"])
        results["time"].append(elapsed)
        
        print(f"P@{k}: {metrics[f'Precision@{k}']:.4f} | R@{k}: {metrics[f'Recall@{k}']:.4f} | MRR@{k}: {metrics[f'MRR@{k}']:.4f}")

    # 3. Plotting
    print("\nGenerating plots...")
    create_plots(results)
    print("Done! Plots saved to 'benchmark_results.png'")

def create_plots(results):
    """Generate and save plots using matplotlib."""
    plt.figure(figsize=(12, 5))

    # Plot 1: Metrics vs K
    plt.subplot(1, 2, 1)
    plt.plot(results["k"], results["precision"], marker='o', label='Precision')
    plt.plot(results["k"], results["recall"], marker='s', label='Recall')
    plt.plot(results["k"], results["mrr"], marker='^', label='MRR')
    
    plt.title("Retrieval Performance vs Top-K")
    plt.xlabel("Top-K (Number of Retrieved Docs)")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(results["k"])

    # Plot 2: Latency (Optional, just for info)
    plt.subplot(1, 2, 2)
    plt.bar([str(k) for k in results["k"]], results["time"], color='skyblue')
    plt.title("Total Evaluation Time vs K")
    plt.xlabel("Top-K")
    plt.ylabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG Retrieval")
    parser.add_argument("--data", required=True, help="Path to validation data")
    parser.add_argument("--storage", default="qdrant_data", help="Path to Qdrant storage")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    
    args = parser.parse_args()
    
    # Define K values to test
    K_VALUES = [1, 3, 5, 10]
    
    run_benchmark(
        data_path=args.data,
        collection_name="insurance_qa",
        storage_path=args.storage,
        k_values=K_VALUES,
        limit_docs=args.limit
    )
