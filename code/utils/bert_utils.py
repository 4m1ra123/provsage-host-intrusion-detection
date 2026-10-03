"""BERT embedding via sentence-transformers (déterministe, frozen)."""
from sentence_transformers import SentenceTransformer
import torch


def load_bert(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
    """Charge un sentence-transformer (384-dim, ~80MB)."""
    return SentenceTransformer(model_name, device=device)


def encode_phrases(model, phrases, batch_size=256):
    """phrases = list[list[str]]. Joint chaque liste, encode → tensor (N, 384)."""
    sentences = [" ".join(p) if p else "<empty>" for p in phrases]
    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
