"""
Third Brain — MCP server.
Exposes recall, remember, graph_traverse, and list_map to all connected agents.
Run: python server.py
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
            "score": round(r.get("rerank_score", 0), 4),
        }
        for r in reranked
    ]


@mcp.tool()
async def remember(
    content: str,
    title: str = None,
    agent: str = "agent",
    tags: list[str] = None,
) -> dict:
    """
    Save a piece of knowledge to Third Brain.
    Auto-classifies into the correct vault folder. No manual filing needed.

    Args:
        content: The knowledge to store. Markdown supported.
        title: Optional note title. Auto-generated from content if omitted.
        agent: Which agent is writing this (e.g. 'claude-code', 'gemini', 'codex').
        tags: Optional list of tags.
    """
    tags = tags or []
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


def _auto_title(content: str) -> str:
    """Generate a short title from the first line of content."""
    first_line = content.strip().splitlines()[0] if content.strip() else "note"
    first_line = first_line.lstrip("#").strip()
    return first_line[:60] if first_line else "note"


if __name__ == "__main__":
    mcp.run()
