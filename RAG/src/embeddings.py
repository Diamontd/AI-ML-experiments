"""
Giga Embeddings wrapper via Hugging Face transformers.

Uses mean pooling over token embeddings (SBERT-style) to get one vector per text.
Defaults to multilingual model for EN+RU: "infgrad/stella-base-en-v2".
Adds batching and max_length controls so CPU runs don’t hang.
"""

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "infgrad/stella-base-en-v2"


class GigaEmbedder:
    """Encode text to vectors using Giga embeddings (Hugging Face)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Choose device automatically if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.vector_size = self.model.config.hidden_size
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into vectors (mean pooled) with batching."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state  # [batch, seq, hidden]
                attention_mask = encoded["attention_mask"]  # [batch, seq]
                masked = hidden * attention_mask.unsqueeze(-1)
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean = summed / counts  # [batch, hidden]

            all_embeddings.extend(mean.cpu().numpy().tolist())

        return all_embeddings

