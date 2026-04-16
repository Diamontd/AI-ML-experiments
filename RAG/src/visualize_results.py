import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO

def main():
    print("loading data...")
    
    # try to find the file
    possible_paths = [
        "benchmark_results/h100_benchmark_results.csv"
    ]
    
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                print(f"read from {p}")
                break
            except:
                pass
    print("\nResults:")
    print(df)

    # plotting: separate charts for each metric
    try:
        metrics = [c for c in df.columns if c != "Model"]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
        colors = ['#4c72b0', '#55a868', '#c44e52'] # colors for models
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(df["Model"], df[metric], color=colors[:len(df)])
            
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            # ticks
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["Model"], rotation=30, ha='right', fontsize=9)
            
            # labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        print("\nopening plot window...")
        plt.show()
    except Exception as e:
        print(f"error plotting: {e}")

    # simple check
    try:
        print("\nQuick math:")
        base = df.iloc[0]
        new_model = df.iloc[1]
        
        diff = (new_model['MRR'] - base['MRR']) / base['MRR'] * 100
        print(f"MRR improvement: {diff:.1f}%")
        
        if df.iloc[1]['MRR'] == df.iloc[2]['MRR']:
            print("reranker didn't change top metrics (scores identical)")
            
    except:
        print("couldn't calc difference")

if __name__ == "__main__":
    main()
