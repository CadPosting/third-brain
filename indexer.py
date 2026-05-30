import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi
from pathlib import Path
from embedder import embed, embed_passage, chunk_text
import frontmatter
import json

DB_PATH = Path.home() / ".third-brain" / "lancedb"
TABLE_NAME = "chunks"

_db = None
_table = None
_bm25 = None
_bm25_docs: list[dict] = []

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("source_path", pa.string()),
    pa.field("topic", pa.string()),
    pa.field("subtopic", pa.string()),
    pa.field("tags", pa.string()),  # JSON array stored as string
    pa.field("chunk_index", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])

def _get_table():
    global _db, _table, _bm25
    if _table is not None:
        # Probe the handle with a cheap count — if the underlying files moved
        # (e.g. index was wiped and rebuilt while the process was running),
        # LanceDB raises an error. Catch it and reconnect.
        try:
            _table.count_rows()
        except Exception:
            _db = None
            _table = None
            _bm25 = None
    if _table is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(str(DB_PATH))
        if TABLE_NAME in _db.table_names():
            _table = _db.open_table(TABLE_NAME)
        else:
            _table = _db.create_table(TABLE_NAME, schema=SCHEMA)
    return _table

def _rebuild_bm25():
    """Full rebuild — called on startup or after deletions."""
    global _bm25, _bm25_docs
    table = _get_table()
    rows = table.to_pandas()
    if rows.empty:
        _bm25 = None
        _bm25_docs = []
        return
    _bm25_docs = rows.to_dict("records")
    tokenized = [row["text"].lower().split() for row in _bm25_docs]
    _bm25 = BM25Okapi(tokenized)

def _update_bm25(new_rows: list[dict]):
    """Incremental update — append new chunks without reloading everything."""
    global _bm25, _bm25_docs
    if not new_rows:
        return
    if _bm25 is None:
        _rebuild_bm25()
        return
    _bm25_docs.extend(new_rows)
    tokenized = [row["text"].lower().split() for row in _bm25_docs]
    _bm25 = BM25Okapi(tokenized)

def index_file(file_path: str) -> int:
    """Parse a markdown file, chunk it, embed each chunk, store in LanceDB."""
    path = Path(file_path)
    if not path.exists() or not path.suffix == ".md":
        return 0

    post = frontmatter.load(str(path))
    content = post.content.strip()
    if not content:
        return 0

    meta = post.metadata
    topic = meta.get("topic", _infer_topic(file_path))
    subtopic = meta.get("subtopic", _infer_subtopic(file_path))
    tags = json.dumps(meta.get("tags", []))

    # Remove existing chunks for this file before re-indexing
    table = _get_table()
    try:
        # Escape single quotes in path to prevent SQL injection
        safe_path = file_path.replace("'", "''")
        table.delete(f"source_path = '{safe_path}'")
    except Exception:
        pass

    chunks = chunk_text(content)
    rows = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_path}::chunk{i}"
        rows.append({
            "id": chunk_id,
            "text": chunk,
            "source_path": file_path,
            "topic": topic,
            "subtopic": subtopic,
            "tags": tags,
            "chunk_index": i,
            "vector": embed_passage(chunk),
        })

    if rows:
        table.add(rows)
        _update_bm25(rows)

    return len(rows)

def vector_search(query: str, top_k: int = 20, topic_filter: str = None) -> list[dict]:
    """Semantic search over all indexed chunks."""
    global _db, _table, _bm25
    q_vec = embed(query)
    for attempt in range(2):
        try:
            table = _get_table()
            search = table.search(q_vec).limit(top_k)
            if topic_filter:
                safe_topic = topic_filter.replace("'", "''")
                search = search.where(f"topic = '{safe_topic}'")
            results = search.to_pandas()
            return results.to_dict("records")
        except Exception:
            if attempt == 0:
                # Force reconnect on next _get_table() call
                _db = None
                _table = None
                _bm25 = None
            else:
                raise

def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """Keyword search using BM25."""
    if _bm25 is None:
        _rebuild_bm25()
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_bm25_docs[i] for i in top_indices if scores[i] > 0]

def rrf_merge(vec_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — combines vector and BM25 rankings without score normalization."""
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}

    for rank, doc in enumerate(vec_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        seen[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        seen[doc_id] = doc

    ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [seen[d] for d in ranked]

def _infer_topic(file_path: str) -> str:
    parts = Path(file_path).parts
    vault_idx = next((i for i, p in enumerate(parts) if p == "vault"), None)
    if vault_idx is not None and vault_idx + 1 < len(parts):
        return parts[vault_idx + 1]
    return "general"

def _infer_subtopic(file_path: str) -> str:
    parts = Path(file_path).parts
    vault_idx = next((i for i, p in enumerate(parts) if p == "vault"), None)
    if vault_idx is not None and vault_idx + 2 < len(parts):
        return parts[vault_idx + 2]
    return "general"
