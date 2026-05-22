"""
Auto-classifies content into the correct vault folder.
Two-pass: keyword rules first (fast, precise), then vector similarity fallback.
"""
from pathlib import Path

VAULT_PATH = Path.home() / "vault"

# Keyword rules — order matters, first match wins
RULES: list[tuple[list[str], str]] = [
    # HPC — most specific first
    (["heteroc342y", "homoc342y", "nonsense-r534", "c342y", "r534"], "hpc/truba/variants"),
    (["sbatch", "squeue", "scancel", "sacct", "walltime", "job script"], "hpc/slurm"),
    (["namd3", "dcdfreq", "truba", "akya", "rattle", "charmm", "psf", "arf-ui"], "hpc"),
    # ML
    (["ppo", "grpo", "dpo", "rlhf", "reward model", "policy gradient", "proximal policy", "alignment"], "machine-learning/alignment"),
    (["smollm", "llama", "transformer", "attention", "fine-tun", "lora", "peft", "quantiz"], "machine-learning/models"),
    (["polynomial regression", "normal equations", "gradient descent", "loss function", "overfitting", "bias variance"], "machine-learning/math"),
    (["neural network", "epoch", "batch size", "learning rate", "checkpoint", "training loop"], "machine-learning"),
    # Web
    (["flexbox", "grid", "css", "html", "animation", "responsive", "media query", "bem"], "web-development/frontend"),
    (["javascript", "dom", "event listener", "fetch", "async", "es6", "arrow function"], "web-development/frontend"),
    (["fastapi", "flask", "django", "uvicorn", "endpoint", "rest api", "http"], "web-development/backend"),
    (["php", "mysql", "sql", "query", "database", "schema", "migration"], "web-development/backend"),
    # Algorithms
    (["big o", "time complexity", "space complexity", "recurrence", "memoization", "dp table"], "algorithms"),
    (["graph", "bfs", "dfs", "dijkstra", "topological", "adjacency", "cycle detection"], "algorithms"),
    # Projects
    (["p2p", "crypto", "tracker", "portfolio", "website"], "projects"),
    (["ai623", "pa2", "assignment"], "projects"),
]

def classify(content: str, fallback_path: str = "general") -> str:
    """Return vault subfolder path for given content."""
    lower = content.lower()
    for keywords, folder in RULES:
        if any(kw in lower for kw in keywords):
            return folder
    return fallback_path

def resolve_vault_path(folder: str, title: str) -> Path:
    """Build full filesystem path for a note given its folder and title."""
    safe_title = title.lower().replace(" ", "-").replace("/", "-")[:60]
    return VAULT_PATH / folder / f"{safe_title}.md"

def build_frontmatter(topic: str, subtopic: str, tags: list[str], agent: str) -> str:
    from datetime import date
    parts = topic.split("/")
    t = parts[0] if parts else topic
    st = parts[1] if len(parts) > 1 else subtopic
    if tags:
        tag_block = "tags:\n" + "\n".join(f"  - {tag}" for tag in tags)
    else:
        tag_block = "tags: []"
    return f"""---
topic: {t}
subtopic: {st}
{tag_block}
agent: {agent}
last_updated: {date.today().isoformat()}
---

"""
