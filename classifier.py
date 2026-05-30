"""
Dynamic domain classifier for Third Brain.

Two-pass classification:
1. Vector similarity against existing domain MOC files
2. If best score is below NEW_DOMAIN_THRESHOLD — treat as new domain,
   create folder + MOC + cross-domain links automatically

No hardcoded keyword lists. New domains emerge from usage.
"""
from pathlib import Path
from datetime import date
import re

VAULT_PATH = Path.home() / "vault"

# Cosine similarity below this → content is a new domain
NEW_DOMAIN_THRESHOLD = 0.30

# Folders that are not top-level domains (skip when scanning MOCs)
SYSTEM_FOLDERS = {"projects"}


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify(content: str, fallback_path: str = "general") -> str:
    """
    Return the vault subfolder for this content.
    Creates a new domain automatically if nothing is similar enough.
    """
    moc_scores = _score_against_mocs(content)

    if not moc_scores:
        # Vault is empty — use fallback
        return fallback_path

    best_domain, best_score = max(moc_scores.items(), key=lambda x: x[1])

    if best_score >= NEW_DOMAIN_THRESHOLD:
        return best_domain

    # New domain — infer name from content and bootstrap it
    domain_name = _infer_domain_name(content)
    related = [d for d, s in moc_scores.items() if s > 0.15]
    _bootstrap_domain(domain_name, content, related)
    return domain_name


def _score_against_mocs(content: str) -> dict[str, float]:
    """
    Compute cosine similarity between content and each domain's MOC file text.
    Returns {domain_folder: similarity_score}.
    """
    from embedder import embed
    import numpy as np

    if not VAULT_PATH.exists():
        return {}

    content_vec = embed(content[:1000])  # cap for speed
    scores: dict[str, float] = {}

    for domain_dir in VAULT_PATH.iterdir():
        if not domain_dir.is_dir():
            continue
        if domain_dir.name.startswith(".") or domain_dir.name in SYSTEM_FOLDERS:
            continue

        # Find the MOC file — named "<Domain> MOC.md"
        moc = _find_moc(domain_dir)
        if moc is None:
            continue

        moc_text = moc.read_text(errors="ignore")[:1500]
        if not moc_text.strip():
            continue

        moc_vec = embed(moc_text)
        similarity = float(np.dot(content_vec, moc_vec))  # both normalized
        scores[domain_dir.name] = similarity

    return scores


# ---------------------------------------------------------------------------
# MOC filename helpers
# ---------------------------------------------------------------------------

def _moc_filename(domain_name: str) -> str:
    """Return the canonical MOC filename for a domain slug. e.g. 'hpc' → 'HPC MOC.md'"""
    title = domain_name.replace("-", " ").title()
    return f"{title} MOC.md"


def _find_moc(domain_dir: Path) -> Path | None:
    """Find the MOC file in a domain directory. Returns None if not found."""
    # Canonical pattern first
    canonical = domain_dir / _moc_filename(domain_dir.name)
    if canonical.exists():
        return canonical
    # Fall back: any file ending in " MOC.md"
    candidates = list(domain_dir.glob("* MOC.md"))
    if candidates:
        return candidates[0]
    # Last resort: any .md in folder root
    candidates = list(domain_dir.glob("*.md"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# New domain bootstrapping
# ---------------------------------------------------------------------------

def _infer_domain_name(content: str) -> str:
    """
    Extract a clean domain slug from content.
    Looks for topic nouns in the first ~200 chars, falls back to timestamp slug.
    """
    # Common filler words to ignore
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "it", "this", "that", "how", "what", "i", "my",
        "we", "you", "be", "are", "was", "have", "has", "do", "can", "will",
        "want", "need", "make", "use", "get", "new", "about", "also", "so",
        "then", "just", "not", "if", "as", "from", "by", "up", "out", "into",
    }

    text = content[:300].lower()
    # Strip markdown
    text = re.sub(r"[#*`_\[\]()>]", " ", text)
    words = re.findall(r"[a-z][a-z0-9]*", text)
    candidates = [w for w in words if len(w) > 3 and w not in STOPWORDS]

    if not candidates:
        from datetime import datetime
        return f"domain-{datetime.now().strftime('%Y%m%d')}"

    # Pick the most frequent meaningful word as the domain name
    from collections import Counter
    freq = Counter(candidates)
    domain = freq.most_common(1)[0][0]
    return domain.replace(" ", "-")


def _bootstrap_domain(domain_name: str, sample_content: str, related_domains: list[str]):
    """
    Create vault/<domain>/<domain>.md — the MOC for the new domain.
    Adds cross-links to related existing domains.
    """
    domain_dir = VAULT_PATH / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    moc_path = domain_dir / _moc_filename(domain_name)
    if moc_path.exists():
        return  # already created

    # Build related links section
    title = domain_name.replace("-", " ").title()
    if related_domains:
        links = "\n".join(
            f"- [[../{d}/{_moc_filename(d).replace('.md','')}|{d.replace('-', ' ').title()}]]"
            for d in related_domains[:5]
        )
        related_section = f"\n## Related domains\n\n{links}\n"
    else:
        related_section = ""

    moc_content = f"""---
topic: {domain_name}
tags:
  - moc
  - auto-generated
last_updated: {date.today().isoformat()}
---

# {title}

> Auto-generated domain. Expand this note to describe what belongs here,
> how notes in this domain connect, and how to navigate the knowledge.

## What belongs here

Notes about {title.lower()} — concepts, decisions, references, and session learnings.
{related_section}
## Notes

> Notes in this domain appear here as they are written.
"""

    moc_path.write_text(moc_content)

    # Index the new MOC immediately so future searches can find it
    try:
        from indexer import index_file
        index_file(str(moc_path))
    except Exception:
        pass

    # Back-link: add this domain to related domains' MOC files
    _add_backlinks(domain_name, related_domains)


def _add_backlinks(new_domain: str, related_domains: list[str]):
    """
    Append a link to the new domain in each related domain's MOC file.
    Only adds if not already present.
    """
    new_moc_name = _moc_filename(new_domain).replace(".md", "")
    link_line = f"- [[../{new_domain}/{new_moc_name}|{new_domain.replace('-', ' ').title()}]]"

    for domain in related_domains:
        moc = _find_moc(VAULT_PATH / domain)
        if moc is None:
            continue
        current = moc.read_text(errors="ignore")
        if new_domain in current:
            continue  # already linked

        if "## Related domains" in current:
            current = current.replace(
                "## Related domains",
                f"## Related domains\n{link_line}",
                1,
            )
        else:
            current = current.rstrip() + f"\n\n## Related domains\n\n{link_line}\n"

        moc.write_text(current)

        try:
            from indexer import index_file
            index_file(str(moc))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Path helpers (unchanged interface)
# ---------------------------------------------------------------------------

def resolve_vault_path(folder: str, title: str) -> Path:
    """Build full filesystem path for a note given its folder and title."""
    safe_title = title.lower().replace(" ", "-").replace("/", "-")[:60]
    return VAULT_PATH / folder / f"{safe_title}.md"


def build_frontmatter(topic: str, subtopic: str, tags: list[str], agent: str) -> str:
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
