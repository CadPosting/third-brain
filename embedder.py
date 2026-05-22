from sentence_transformers import SentenceTransformer

# Downloads once (~130MB), runs fully offline after
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(text: str) -> list[float]:
    """Convert text to a 384-dim vector. Prepend instruction for BGE retrieval quality."""
    instruction = "Represent this sentence for searching relevant passages: "
    return _model.encode(instruction + text, normalize_embeddings=True).tolist()

def embed_passage(text: str) -> list[float]:
    """Embed a document passage (no instruction prefix needed for storage)."""
    return _model.encode(text, normalize_embeddings=True).tolist()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping word chunks for precise retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
