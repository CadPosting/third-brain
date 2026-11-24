"""
Orphan health check + backfill for the vault.

An orphan is a note no other note links to with [[...]]. MOCs, HOME.md and
.trash/.obsidian are ignored.

    python backfill_orphans.py            # report only, changes nothing
    python backfill_orphans.py --apply    # add each orphan to its domain MOC
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from classifier import VAULT_PATH, link_note_in_moc

SKIP_DIRS = {".trash", ".obsidian", ".internal"}
LINK_RE = re.compile(r"\[\[([^\]|#]+)")


def _notes() -> list[Path]:
    return [
        p for p in VAULT_PATH.rglob("*.md")
        if not any(part in SKIP_DIRS for part in p.relative_to(VAULT_PATH).parts)
    ]


def find_orphans() -> list[Path]:
    """Return notes with no incoming [[link]] from any other note."""
    notes = _notes()
    linked: set[str] = set()
    for p in notes:
        for target in LINK_RE.findall(p.read_text(errors="ignore")):
            # [[../hpc/HPC MOC]] and [[HPC MOC]] both resolve to the stem.
            stem = Path(target.strip()).name
            if stem.endswith(".md"):
                stem = stem[:-3]
            if stem != p.stem:
                linked.add(stem)
    return [
        p for p in notes
        if p.stem not in linked and not p.name.endswith(" MOC.md") and p.name != "HOME.md"
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="link orphans into their MOCs")
    args = ap.parse_args()

    orphans = find_orphans()
    by_folder = Counter(str(p.relative_to(VAULT_PATH).parent) for p in orphans)
    print(f"{len(orphans)} orphan notes")
    for folder, n in by_folder.most_common():
        print(f"  {n:4d}  {folder}")

    if args.apply:
        for p in orphans:
            link_note_in_moc(p)
        print(f"linked {len(orphans)} notes into their domain MOCs")


if __name__ == "__main__":
    main()
