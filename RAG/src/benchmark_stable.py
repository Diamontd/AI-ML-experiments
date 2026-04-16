import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from qdrant_client.http import models as qm
from src.data_loader import load_jsonl
from src.embeddings import GigaEmbedder
from src.vector_store import create_client, create_collection
from src.retrieval import evaluate_dataset

def index_data(client, collection_name, embedder, documents):
    """Index documents into Qdrant."""
    print(f"Indexing {len(documents)} documents...")
    
    # Create collection if not exists
    # We need vector size. Encode one to check.
    sample_vec = embedder.encode(["test"])[0]
    vector_size = len(sample_vec)
    
    # Recreate collection to ensure it's empty/clean
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )
    
    # Batch encode
    texts = [doc.get('input', '') for doc in documents]
    print("Encoding vectors...")
    vectors = embedder.encode(texts)
    
    # Prepare points
    points = []
    for i, (vector, doc) in enumerate(zip(vectors, documents)):
        points.append(qm.PointStruct(id=i, vector=vector, payload=doc))
        
    # Upsert
    client.upsert(collection_name=collection_name, points=points)
    print("Indexing complete.")

def plot_results(metrics, save_path="stable_benchmark_results.png"):
    """Plot the metrics as a bar chart."""
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    x = np.arange(len(labels))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(x, values, width, color='skyblue')
    
    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_title('Retrieval Performance: Stella-Base (Local CPU)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def main():
    print("=== Starting Safe Mode Benchmark ===")
    
    # 1. Configuration
    DATA_PATH = "data/valid.jsonl"
    LIMIT = 100
    COLLECTION_NAME = "benchmark_stable"
    MODEL_NAME = "infgrad/stella-base-en-v2"
    
    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    documents = load_jsonl(Path(DATA_PATH))
    if LIMIT:
        documents = documents[:LIMIT]
        print(f"Limited to {len(documents)} documents.")
        
    # 3. Initialize Components
    print(f"Initializing Embedder ({MODEL_NAME})...")
    embedder = GigaEmbedder(model_name=MODEL_NAME)
    
    print("Initializing Qdrant (Memory)...")
    client = create_client(":memory:")
    
    # 4. Index Data
    index_data(client, COLLECTION_NAME, embedder, documents)
    
    # 5. Evaluate
    print("\n--- Running Evaluation (Vector Search Only) ---")
    # We use use_rerank=False to ensure stability (Safe Mode)
    metrics = evaluate_dataset(
        client,
        COLLECTION_NAME,
        embedder,
        documents,
        top_k=3,
        use_rerank=False
    )
    
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 6. Plot
    plot_results(metrics)
    print("\nBenchmark finished successfully.")

if __name__ == "__main__":
    main()
