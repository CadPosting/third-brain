from flashrank import Ranker, RerankRequest

# ~4MB model, CPU, <20ms for 20 candidates
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank candidates using cross-encoder. Returns top_k most relevant."""
    if not candidates:
        return []

    passages = [{"id": i, "text": c.get("text", "")} for i, c in enumerate(candidates)]
    request = RerankRequest(query=query, passages=passages)
    results = _ranker.rerank(request)

    reranked = []
    for r in results[:top_k]:
        # FlashRank returns Passage objects — access as attributes
        original_id = r.id if hasattr(r, "id") else r.get("id", 0)
        score = r.score if hasattr(r, "score") else r.get("score", 0)
        if original_id < len(candidates):
            doc = candidates[original_id].copy()
            doc["rerank_score"] = score
            reranked.append(doc)

    return reranked
