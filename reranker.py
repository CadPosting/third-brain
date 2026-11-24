from flashrank import Ranker, RerankRequest
import onnxruntime as ort

# ~4MB model, CPU, <20ms for 20 candidates.
#
# 2026-11-24: this was the server's memory leak (2.4G anon RSS in 5h, then an
# OOM kill mid-`remember`). Two FlashRank defaults combined:
#   - max_length=512: 20 passages x 512 tokens x 12 layers of attention is a
#     few hundred MB of activations per call. Vault chunks are 200 words
#     (~260 tokens), so 256 covers nearly all of the text.
#   - ONNX Runtime's CPU memory arena keeps a buffer for every new input shape
#     and never returns memory to the OS. Every query/candidate mix pads to a
#     different length, so the arena kept growing. Turning the arena off costs
#     a little speed per call and keeps RSS flat.
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=256)
_opts = ort.SessionOptions()
_opts.enable_cpu_mem_arena = False
_opts.intra_op_num_threads = 1
_opts.inter_op_num_threads = 1
_ranker.session = ort.InferenceSession(
    _ranker.session._model_path, sess_options=_opts, providers=["CPUExecutionProvider"]
)

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
