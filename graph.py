"""
Knowledge graph layer using Graphiti (by Zep).
Tracks entities, relationships, and temporal context across all agent sessions.
Falls back gracefully if graphiti-core is not installed.
"""
from pathlib import Path
from datetime import datetime

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    _GRAPHITI_AVAILABLE = True
except ImportError:
    _GRAPHITI_AVAILABLE = False

# Graphiti requires Neo4j. Set these in your environment or .env file:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-password
import os
from dotenv import load_dotenv
load_dotenv(Path.home() / ".third-brain" / ".env")

_graph = None

def _get_graph():
    global _graph
    if not _GRAPHITI_AVAILABLE:
        return None
    if _graph is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            return None  # don't crash — just disable graph layer
        _graph = Graphiti(uri, user, password)
    return _graph

async def add_episode(content: str, source: str = "agent", timestamp: datetime = None) -> bool:
    """Add a new fact/event to the knowledge graph."""
    g = _get_graph()
    if g is None:
        return False
    ts = timestamp or datetime.utcnow()
    await g.add_episode(
        name=source,
        episode_body=content,
        source=EpisodeType.text,
        reference_time=ts,
    )
    return True

async def search_graph(query: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge graph for related entities and relationships."""
    g = _get_graph()
    if g is None:
        return []
    results = await g.search(query, num_results=top_k)
    return [
        {
            "text": r.fact,
            "source_path": "graph",
            "topic": "graph",
            "subtopic": "relationship",
            "id": str(r.uuid),
        }
        for r in results
    ]

def is_available() -> bool:
    """True only when the graph layer can actually be USED, not merely imported.

    The import flag alone is not enough. When graphiti-core is installed but
    Neo4j is not running, every caller passed this gate and then blocked inside
    Graphiti's Bolt driver, which has no connect timeout of its own — that is
    what made `recall` hang past the 300s MCP tool timeout while the server
    looked alive. Checked here, once, because every call site is `if
    graph_available(): await search_graph(...)`.
    """
    if not _GRAPHITI_AVAILABLE:
        return False
    if not os.environ.get("NEO4J_PASSWORD", ""):
        return False
    return _bolt_reachable()


# Cached so a dead Neo4j costs one 0.3s probe per process, not one per query.
# None = not yet probed. A restart re-probes, which is the intended way to pick
# up a Neo4j that came up later.
_reachable = None


def _bolt_reachable(timeout: float = 0.3) -> bool:
    global _reachable
    if _reachable is not None:
        return _reachable
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=timeout):
            _reachable = True
    except OSError:
        _reachable = False
    return _reachable
