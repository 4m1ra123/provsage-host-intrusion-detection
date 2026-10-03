"""EmbeddingGemma — Google google/embeddinggemma-300m (768-dim)."""
from sentence_transformers import SentenceTransformer


def load_gemma(model_name="google/embeddinggemma-300m", device="cuda"):
    return SentenceTransformer(model_name, device=device)


def encode_phrases_raw(model, phrases, batch_size=64):
    sentences = [" ".join(p) if p else "<empty>" for p in phrases]
    return model.encode(sentences, batch_size=batch_size,
                        show_progress_bar=True, convert_to_tensor=True)


def encode_phrases_prompted(model, phrases, batch_size=64):
    """Version prompted : préfixe task-aware EmbeddingGemma."""
    sentences = [
        f"task: classification | query: {' '.join(p) if p else '<empty>'}"
        for p in phrases
    ]
    return model.encode(sentences, batch_size=batch_size,
                        show_progress_bar=True, convert_to_tensor=True)
