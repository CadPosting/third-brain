"""
Third Brain — MCP server.
Exposes recall, remember, graph_traverse, and list_map to all connected agents.

Runs as a persistent HTTP server on http://127.0.0.1:7891/mcp
Managed by systemd: systemctl --user start third-brain
"""
import os
import sys
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime, date
import frontmatter as fm

from fastmcp import FastMCP
from indexer import index_file, vector_search, bm25_search, rrf_merge, find_similar_note
from reranker import rerank
from graph import add_episode, search_graph, is_available as graph_available
from classifier import classify, resolve_vault_path, build_frontmatter
# Module import too: the write path reaches for the private helpers
# (_find_moc, _score_against_mocs, _bootstrap_domain) via the module rather
# than name-importing each one.
import classifier

VAULT_PATH = Path.home() / "vault"

# Write-time dedup: before creating a new note, if an existing note in the same
# domain folder is at least this cosine-similar, append to it instead of making
# a near-duplicate sibling. Calibrated start point; lower toward 0.80 if dupes
# still slip through. Disable entirely with TB_DEDUP=0.
DEDUP_SIM = 0.85
DEDUP_MAX_BYTES = 8192  # don't accrete into a note past this size — make a new one

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
    # Stage tracing, off unless TB_TRACE=1. This is what localised the reranker
    # deadlock (every stage before rerank finished in 0.3s); keep it available,
    # quiet by default.
    import sys, os as _os, time as _t
    _trace = _os.environ.get("TB_TRACE", "0") == "1"
    def _stage(msg):
        if _trace:
            print(f"[recall] {msg}", file=sys.stderr, flush=True)
    _t0 = _t.time()
    _stage(f"start query={query!r} top_k={top_k}")
    vec = vector_search(query, top_k=20, topic_filter=topic_filter)
    _stage(f"vector_search done n={len(vec)} t={_t.time()-_t0:.1f}s")
    bm25 = bm25_search(query, top_k=20)
    _stage(f"bm25_search done n={len(bm25)} t={_t.time()-_t0:.1f}s")
    merged = rrf_merge(vec, bm25)
    _stage(f"rrf_merge done n={len(merged)} t={_t.time()-_t0:.1f}s")

    # Add graph results if available
    if graph_available():
        graph_results = await search_graph(query, top_k=5)
        merged = merged + graph_results
    _stage(f"graph stage done t={_t.time()-_t0:.1f}s")

    # rerank() is synchronous ONNX inference. Called directly it blocks FastMCP's
    # event loop, so the server cannot even answer a health check while it runs;
    # to_thread keeps the loop free. The timeout is the safety net that turns a
    # wedged reranker into a degraded-but-answered query instead of a hang past
    # the client's 300s tool timeout — vector+BM25 results are already RRF-merged
    # and useful on their own.
    import asyncio
    try:
        reranked = await asyncio.wait_for(
            asyncio.to_thread(rerank, query, merged, top_k=top_k), timeout=25
        )
        _stage(f"rerank done n={len(reranked)} t={_t.time()-_t0:.1f}s")
    except (asyncio.TimeoutError, RuntimeError, ValueError, OSError) as e:
        # Always logged, not gated behind TB_TRACE: this means results came back
        # unreranked (score 0), which is a real quality degradation worth seeing.
        print(f"[recall] rerank FAILED ({type(e).__name__}) after "
              f"t={_t.time()-_t0:.1f}s — returning RRF-merged results unreranked",
              file=sys.stderr, flush=True)
        reranked = merged[:top_k]

    return [
        {
            "text": r.get("text", ""),
            "source": r.get("source_path", ""),
            "topic": r.get("topic", ""),
            "score": round(float(r.get("rerank_score", 0)), 4),
        }
        for r in reranked
    ]


async def _do_remember(
    content: str,
    title: str = None,
    agent: str = "agent",
    tags: list[str] = None,
    domain: str = None,
) -> dict:
    """Core write path shared by the MCP `remember` tool and the /remember
    HTTP route (used by the session fact-extractor hook)."""
    tags = tags or []

    # The classify() branch runs embedding work (_score_against_mocs embeds the
    # content against every domain MOC; classify() does the same internally and
    # was measured at 20.8s standalone). They are synchronous, so calling them
    # directly here blocks FastMCP's event loop for the whole duration — which is
    # how `remember` came to hang past the client's 300s tool timeout while the
    # request sat logged as received and never completed. Offload to a thread.
    import asyncio

    def _resolve_folder() -> str:
        if domain:
            # Agent knows the domain — use it directly. No embedding work here:
            # a new domain's MOC is bootstrapped AFTER the note is on disk.
            return domain.strip().lower().replace(" ", "-")
        return classify(content)

    # A timed-out wait_for does NOT stop its worker thread; that thread kept
    # embedding and created the sock-studio MOC after we had given up on it.
    # With an explicit domain _resolve_folder is now instant, so only the
    # classify() path can time out.
    try:
        folder = await asyncio.wait_for(asyncio.to_thread(_resolve_folder), timeout=60)
    except asyncio.TimeoutError:
        # Never lose a write to a slow classifier: park it in `inbox`.
        folder = "inbox"
        print("[remember] folder resolution timed out — using 'inbox'",
              file=sys.stderr, flush=True)

    domain_dir = VAULT_PATH / folder.split("/")[0]
    new_domain = classifier._find_moc(domain_dir) is None
    content_hash = hashlib.sha256(content.strip().encode()).hexdigest()[:16]

    # Idempotency: a client that lost the connection and retries must not get
    # a second copy. The hash is stored in each new note's frontmatter, and
    # appended blocks are matched by their raw text.
    if domain_dir.exists():
        for p in domain_dir.rglob("*.md"):
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            if f"content_hash: {content_hash}" in text or (
                    len(content.strip()) >= 40 and content.strip() in text):
                return {"status": "duplicate", "path": str(p), "folder": folder,
                        "chunks_indexed": 0, "deduped_into": None,
                        "new_domain_created": False}

    title = title or _auto_title(content)
    note_path = resolve_vault_path(folder, title)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    # Write-time dedup into a same-domain near-duplicate.
    deduped_into = None
    if os.environ.get("TB_DEDUP", "1") != "0" and not note_path.exists():
        try:
            match = await asyncio.wait_for(
                asyncio.to_thread(find_similar_note, content, str(note_path.parent) + "/"),
                timeout=30,
            )
        except asyncio.TimeoutError:
            match = None
            print("[remember] dedup lookup timed out — writing a new note",
                  file=sys.stderr, flush=True)
        if match:
            cand_path, sim = match
            cand = Path(cand_path)
            is_moc = "MOC" in cand.name or cand.name == "HOME.md"
            small_enough = cand.exists() and cand.stat().st_size <= DEDUP_MAX_BYTES
            # Same subject, different event is not a duplicate: on 2026-11-24 a
            # new UK-page change was merged into an Oct 30 note about the same
            # page. Only merge into a note last touched today.
            same_day = cand.exists() and _last_updated(cand) == date.today().isoformat()
            if sim >= DEDUP_SIM and not is_moc and small_enough and same_day:
                note_path = cand
                deduped_into = cand_path

    # Write the note FIRST, atomically (tmp + os.replace), so a crash at any
    # later step (indexing, MOC bootstrap, graph) cannot lose it.
    if note_path.exists():
        existing = fm.load(str(note_path))
        existing.content += f"\n\n---\n*{agent} — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n{content}"
        existing.metadata["last_updated"] = date.today().isoformat()
        _atomic_write(note_path, fm.dumps(existing))
    else:
        frontmatter_str = build_frontmatter(folder, "", tags, agent)
        frontmatter_str = frontmatter_str.replace(
            "\n---\n\n", f"\ncontent_hash: {content_hash}\n---\n\n", 1)
        moc_stem = classifier._moc_filename(folder.split("/")[0])[:-3]
        footer = f"\n\n---\n*[[{moc_stem}]]*\n"
        _atomic_write(note_path, frontmatter_str + content + footer)

    # Everything below is best-effort: the note is already safe on disk.
    def _post_write() -> int:
        if new_domain:
            related = classifier.related_domains(classifier._score_against_mocs(content))
            classifier._bootstrap_domain(folder.split("/")[0], content, related)
        classifier.link_note_in_moc(note_path)
        return index_file(str(note_path))

    chunks = 0
    try:
        chunks = await asyncio.wait_for(asyncio.to_thread(_post_write), timeout=60)
    except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as e:
        print(f"[remember] post-write step failed ({type(e).__name__}); note is saved, "
              f"watcher will index it", file=sys.stderr, flush=True)

    if graph_available():
        await add_episode(content, source=agent)

    return {
        "status": "saved",
        "path": str(note_path),
        "folder": folder,
        "chunks_indexed": chunks,
        "deduped_into": deduped_into,
        "new_domain_created": new_domain,
    }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _last_updated(path: Path) -> str:
    try:
        return str(fm.load(str(path)).metadata.get("last_updated", ""))
    except (OSError, ValueError):
        return ""


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
    return await _do_remember(content, title=title, agent=agent, tags=tags, domain=domain)


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
        # Optional relevance gate: drop results whose cross-encoder rerank score
        # is below this. Clients (inject.py) use it so weak/irrelevant prompts
        # inject nothing instead of always pulling top_k. Default 0.0 = no gate.
        min_score = float(body.get("min_score", 0.0))
        if not query:
            return JSONResponse([], status_code=200)
        # Run off the event loop: called inline, an 8s rerank under memory
        # pressure froze the whole server, so concurrent MCP calls (remember)
        # could not be answered. Same pattern as recall().
        def _search():
            vec = vector_search(query, top_k=top_k * 4)
            bm25 = bm25_search(query, top_k=top_k * 4)
            merged = rrf_merge(vec, bm25)
            try:
                return rerank(query, merged, top_k=top_k)
            except (RuntimeError, ValueError, OSError):
                return merged[:top_k]
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=25)
        trimmed = [
            {
                "source_path": r.get("source_path", ""),
                "text": r.get("text", ""),
                "score": round(float(r.get("rerank_score", 0.0)), 4),
            }
            for r in results
            if float(r.get("rerank_score", 0.0)) >= min_score
        ]
        return JSONResponse(trimmed)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/remember", methods=["POST"])
async def remember_endpoint(request) -> "Response":
    """
    Plain JSON write endpoint for hooks/scripts (e.g. the session fact-extractor)
    that can't speak MCP. Shares the exact write path — including Phase 3 dedup —
    with the MCP `remember` tool.
    POST {"content": "...", "domain": "...", "title": "...", "agent": "...", "tags": [...]}
    """
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
        content = (body.get("content") or "").strip()
        if not content:
            return JSONResponse({"error": "empty content"}, status_code=400)
        # domain flows into a filesystem path and, via this HTTP route, can come
        # from an LLM reading an untrusted transcript. Reject path-traversal
        # patterns while still allowing legit subdomains like "a/b".
        domain = body.get("domain")
        if domain and (".." in domain or domain.startswith("/") or "\\" in domain):
            return JSONResponse({"error": "invalid domain"}, status_code=400)
        result = await _do_remember(
            content,
            title=body.get("title"),
            agent=body.get("agent", "agent"),
            tags=body.get("tags"),
            domain=domain,
        )
        return JSONResponse(result)
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
        _compact_index()
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


def _compact_index():
    """Merge LanceDB fragments and drop old versions.

    Every index_file() is a delete + add, and nothing ever compacted, so by
    2026-11-24 the table had 2,047 fragments and 1,921 versions for ~1,000
    notes. Every search had to open all of them. The index is a cache of the
    vault, so pruning history loses nothing.
    """
    import sys
    from datetime import timedelta
    import indexer
    try:
        t = indexer._get_table()
        t.optimize(cleanup_older_than=timedelta(hours=1))
        print("[startup] Index compacted.", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — compaction is optional, never block startup
        print(f"[startup] Compaction skipped: {e}", file=sys.stderr)


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
