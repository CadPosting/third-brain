"""
Third Brain — MCP server.
Exposes recall, remember, graph_traverse, and list_map to all connected agents.

Runs as a persistent HTTP server on http://127.0.0.1:7891/mcp
Managed by systemd: systemctl --user start third-brain
"""
from pathlib import Path
from datetime import datetime
import frontmatter as fm

from fastmcp import FastMCP
from indexer import index_file, vector_search, bm25_search, rrf_merge
from reranker import rerank
from graph import add_episode, search_graph, is_available as graph_available
from classifier import classify, resolve_vault_path, build_frontmatter

VAULT_PATH = Path.home() / "vault"

mcp = FastMCP("third-brain")


@mcp.tool()
async def recall(
    query: str,
    top_k: int = 5,
    topic_filter: str = None,
) -> list[dict]:
    """
    Search Third Brain for relevant knowledge.
    Combines semantic search + keyword search + knowledge graph, then reranks.

    Args:
        query: What you want to find. Natural language.
        top_k: Number of results to return (default 5).
        topic_filter: Limit to a domain e.g. 'machine-learning', 'hpc', 'web-development'.
    """
    vec = vector_search(query, top_k=20, topic_filter=topic_filter)
    bm25 = bm25_search(query, top_k=20)
    merged = rrf_merge(vec, bm25)

    # Add graph results if available
    if graph_available():
        graph_results = await search_graph(query, top_k=5)
        merged = merged + graph_results

    reranked = rerank(query, merged, top_k=top_k)

    return [
        {
            "text": r.get("text", ""),
            "source": r.get("source_path", ""),
            "topic": r.get("topic", ""),
            "score": round(float(r.get("rerank_score", 0)), 4),
        }
        for r in reranked
    ]


@mcp.tool()
async def remember(
    content: str,
    title: str = None,
    agent: str = "agent",
    tags: list[str] = None,
    domain: str = None,
) -> dict:
    """
    Save a piece of knowledge to Third Brain.

    The agent should supply `domain` whenever it knows which topic area this
    belongs to (e.g. 'networking', 'machine-learning', 'hpc'). If domain is
    a new one that doesn't exist yet, the vault folder and MOC file are created
    automatically and cross-linked to related existing domains.

    If domain is omitted, the classifier infers it from content similarity
    against existing domain MOC files.

    Args:
        content: The knowledge to store. Markdown supported.
        title:   Optional note title. Auto-generated from content if omitted.
        agent:   Which agent is writing (e.g. 'claude-code', 'gemini').
        tags:    Optional list of tags.
        domain:  Explicit domain name or subfolder (e.g. 'networking',
                 'machine-learning/alignment'). Preferred over auto-classify.
    """
    tags = tags or []

    if domain:
        # Agent knows the domain — use it directly.
        # classify() with a known domain still handles MOC bootstrapping
        # for new domains via the side-effect in _bootstrap_domain.
        folder = domain.strip().lower().replace(" ", "-")
        # Ensure MOC exists for this domain if it's new
        from classifier import _score_against_mocs, _bootstrap_domain, _find_moc, VAULT_PATH
        domain_dir = VAULT_PATH / folder.split("/")[0]
        if _find_moc(domain_dir) is None:
            related = [
                d for d, s in _score_against_mocs(content).items() if s > 0.15
            ]
            _bootstrap_domain(folder.split("/")[0], content, related)
    else:
        folder = classify(content)

    title = title or _auto_title(content)
    note_path = resolve_vault_path(folder, title)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing note or create new
    if note_path.exists():
        existing = fm.load(str(note_path))
        existing.content += f"\n\n---\n*{agent} — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n{content}"
        with open(note_path, "w") as f:
            f.write(fm.dumps(existing))
    else:
        frontmatter_str = build_frontmatter(folder, "", tags, agent)
        note_path.write_text(frontmatter_str + content)

    # Index the note
    chunks = index_file(str(note_path))

    # Add to knowledge graph
    if graph_available():
        await add_episode(content, source=agent)

    return {
        "status": "saved",
        "path": str(note_path),
        "folder": folder,
        "chunks_indexed": chunks,
        "new_domain_created": _find_moc(VAULT_PATH / folder.split("/")[0]) is None,
    }


@mcp.tool()
def capture(file_path: str) -> dict:
    """
    Index a file into Third Brain immediately.

    Claude Code does this automatically via a PostToolUse hook whenever it
    writes or edits a .md file. Agents without hook support (e.g. Antigravity)
    should call this manually after writing any file that contains knowledge
    worth remembering.

    Args:
        file_path: Absolute path to the .md file to index.
    """
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "reason": f"file not found: {file_path}"}
    if path.suffix != ".md":
        return {"status": "skipped", "reason": "only .md files are indexed"}

    chunks = index_file(file_path)
    return {
        "status": "indexed",
        "file": file_path,
        "chunks": chunks,
    }


@mcp.tool()
async def summarize_session(
    summary: str,
    agent: str = "agent",
    tags: list[str] = None,
) -> dict:
    """
    Write a structured session summary to vault/projects/ and index it.

    Claude Code does this automatically via a Stop hook at session end.
    Agents without hook support (e.g. Antigravity) should call this manually
    at the end of every session.

    The summary should cover what was done, decisions made, and any next steps.
    Markdown is supported.

    Args:
        summary: The session summary in markdown. Include ## sections for
                 What was done, Decisions made, Next steps.
        agent:   Which agent is writing (e.g. 'antigravity', 'claude-code').
        tags:    Optional extra tags beyond ['session', agent].
    """
    tags = tags or []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    note_path = VAULT_PATH / "projects" / f"session-{timestamp}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    all_tags = list({"session", agent} | set(tags))
    tag_block = "\n".join(f"  - {t}" for t in all_tags)

    note_path.write_text(f"""---
topic: projects
subtopic: sessions
tags:
{tag_block}
agent: {agent}
last_updated: {datetime.now().strftime('%Y-%m-%d')}
---

# Session {timestamp}

{summary}

---
*[[Projects MOC]]*
""")

    chunks = index_file(str(note_path))

    if graph_available():
        await add_episode(summary, source=agent)

    return {
        "status": "saved",
        "path": str(note_path),
        "chunks_indexed": chunks,
    }


@mcp.tool()
async def graph_traverse(entity: str, depth: int = 2) -> list[dict]:
    """
    Traverse the knowledge graph to find related concepts.
    Useful for 'what connects to X?' queries.

    Args:
        entity: The concept or entity to start from (e.g. 'PPO', 'NAMD3').
        depth: How many hops to traverse (default 2).
    """
    if not graph_available():
        return [{"text": "Knowledge graph not available. Install graphiti-core.", "source": "system"}]
    results = await search_graph(entity, top_k=depth * 5)
    return results


@mcp.tool()
def list_map(topic: str = None) -> dict:
    """
    List the vault structure — topics, subtopics, and note counts.
    Useful for orientation at session start.

    Args:
        topic: Optional — drill into a specific topic e.g. 'machine-learning'.
    """
    root = VAULT_PATH / topic if topic else VAULT_PATH
    if not root.exists():
        return {"error": f"Topic '{topic}' not found in vault."}

    structure = {}
    for folder in sorted(root.rglob("*")):
        if folder.is_dir():
            notes = list(folder.rglob("*.md"))
            rel = str(folder.relative_to(VAULT_PATH))
            structure[rel] = len(notes)

    return {"vault": str(root), "structure": structure}


@mcp.custom_route("/search", methods=["POST"])
async def search_endpoint(request) -> "Response":
    """
    Plain JSON search endpoint for hooks and scripts.
    POST {"query": "...", "top_k": 3} → JSON array of {source_path, text} dicts.
    Much faster than loading the embedding model in a subprocess.
    """
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
        query = body.get("query", "")
        top_k = int(body.get("top_k", 3))
        if not query:
            return JSONResponse([], status_code=200)
        vec = vector_search(query, top_k=top_k * 4)
        bm25 = bm25_search(query, top_k=top_k * 4)
        merged = rrf_merge(vec, bm25)
        results = rerank(query, merged, top_k=top_k)
        trimmed = [{"source_path": r.get("source_path", ""), "text": r.get("text", "")} for r in results]
        return JSONResponse(trimmed)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _auto_title(content: str) -> str:
    """Generate a short title from the first line of content."""
    first_line = content.strip().splitlines()[0] if content.strip() else "note"
    first_line = first_line.lstrip("#").strip()
    return first_line[:60] if first_line else "note"


def _startup_integrity_check():
    """
    Run before accepting any requests.
    Verifies the LanceDB index can actually serve a vector search.
    If it can't (corrupt manifest, missing data files, etc.), wipes the entire
    lancedb/ directory, resets all in-process LanceDB globals, and rebuilds
    from vault/ automatically.
    Logs everything to stderr so systemd journal captures it.
    """
    import sys
    import shutil
    import indexer

    print("[startup] Running index integrity check...", file=sys.stderr)
    try:
        indexer.vector_search("startup probe", top_k=1)
        print("[startup] Index OK.", file=sys.stderr)
        return
    except Exception as e:
        print(f"[startup] Index check failed: {e}", file=sys.stderr)
        print("[startup] Wiping corrupt index and rebuilding from vault...", file=sys.stderr)

    # Wipe the entire lancedb directory — partial wipes leave stale manifests
    if indexer.DB_PATH.exists():
        shutil.rmtree(str(indexer.DB_PATH))
        print(f"[startup] Wiped {indexer.DB_PATH}", file=sys.stderr)

    # Reset all in-process LanceDB globals so index_file gets a clean connection
    indexer._db = None
    indexer._table = None
    indexer._bm25 = None
    indexer._bm25_docs = []

    # Reindex every .md file in the vault
    files = sorted(VAULT_PATH.rglob("*.md"))
    total = 0
    for f in files:
        try:
            n = indexer.index_file(str(f))
            total += n
        except Exception as fe:
            print(f"[startup] Warning: failed to index {f.name}: {fe}", file=sys.stderr)
    print(f"[startup] Rebuilt index: {total} chunks from {len(files)} files.", file=sys.stderr)


def _warmup():
    """Pre-warm the embedder and reranker so first /search request is fast."""
    import sys
    print("[startup] Warming up embedder and reranker...", file=sys.stderr)
    try:
        from reranker import rerank, _ranker  # noqa: F401 — forces model load
        results = vector_search("warmup", top_k=3)
        bm25 = bm25_search("warmup", top_k=3)
        merged = rrf_merge(results, bm25)
        rerank("warmup", merged, top_k=1)
        print("[startup] Warmup complete.", file=sys.stderr)
    except Exception as e:
        print(f"[startup] Warmup error (non-fatal): {e}", file=sys.stderr)


if __name__ == "__main__":
    _startup_integrity_check()
    _warmup()
    mcp.run(transport="http", host="127.0.0.1", port=7891, path="/mcp")
