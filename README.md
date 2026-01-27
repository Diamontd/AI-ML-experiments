# Insurance Benchmark


Latest version of the work is done in the notebook. To run it:

1.  Open **`benchmark_final.ipynb`**.
2.  Run the cells in order.
    *   It handles dependency installation and setup automatically.
    *   It's designed to run smoothly on Google Colab or a local machine with a good GPU (H100/A100 recommended for the Giga models).
    *   Sometimes you may need to restart the runtime after installing dependencies.
    *   Results of the latest experiments are saved in `latest_results/` folder.

**Dependencies (Installed Automatically):**
The notebook will automatically install the necessary libraries for you. For reference, the main ones are:
*   `qdrant-client` (Vector Database)
*   `sentence-transformers` & `FlagEmbedding` (Models)
*   `accelerate` & `bitsandbytes` (GPU Optimization)
*   `numpy` (pinned to <2.0), `pandas`, `matplotlib` (Data & Math)

**A couple of notes on the project structure:**
*   **Data**: The dataset files (`test.jsonl`, `train.jsonl`) are located in the `sample_data/` folder. The notebook looks for them there.
*   Data is downloaded from here: https://huggingface.co/datasets/deccan-ai/insuranceQA-v2/tree/main
*   Old versions of the code and experiments are in the `archive/` folder.
