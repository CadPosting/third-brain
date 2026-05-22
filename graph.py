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
    return _GRAPHITI_AVAILABLE
