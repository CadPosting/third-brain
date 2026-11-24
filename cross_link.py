"""
Multi-topic linking: link each note from the topic MOCs it belongs to, not
just the MOC of the folder it sits in. A tsocks SEO note in
web-development/tsocks/ should also show up under Analytics.

How: each folder that has a MOC gets a topic vector = mean of the vectors of
the notes directly in it (read from the LanceDB index, nothing re-embedded).
Each note is scored against every topic and linked from up to MAX_EXTRA of
the best ones (other than its home folder) under a "## Also relevant" section.
Only MOC files are edited.

    python cross_link.py            # dry run: show what would be linked
    python cross_link.py --apply
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import indexer
from classifier import VAULT_PATH, _add_line_under, _find_moc

MAX_EXTRA = 2
MIN_SIM = 0.82      # cosine vs topic centroid; calibrated on the dry run
HOME_MARGIN = 0.03  # extra topic must be nearly as close as the note's own folder
MIN_NOTES = 3       # a topic needs this many notes for a meaningful centroid
SECTION = "## Also relevant"
DATE_MOC = re.compile(r" \d{4}-\d{2} MOC$")  # monthly Projects index, not a topic


def _note_vectors() -> dict[Path, np.ndarray]:
    """Mean chunk vector per indexed note (unit-normalised)."""
    df = indexer._get_table().to_pandas()[["source_path", "vector"]]
    acc: dict[str, list] = defaultdict(list)
    for sp, vec in zip(df["source_path"], df["vector"]):
        acc[sp].append(vec)
    out = {}
    for sp, vecs in acc.items():
        p = Path(sp)
        if not p.exists() or p.name.endswith(" MOC.md") or p.name == "HOME.md":
            continue
        v = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        out[p] = v / (np.linalg.norm(v) or 1.0)
    return out


def _topics(notes: dict[Path, np.ndarray]) -> dict[Path, tuple[Path, np.ndarray]]:
    """folder -> (MOC path, centroid) for every folder that is a real topic."""
    by_folder: dict[Path, list] = defaultdict(list)
    for p, v in notes.items():
        by_folder[p.parent].append(v)
    topics = {}
    for folder, vecs in by_folder.items():
        if folder == VAULT_PATH / "projects" or len(vecs) < MIN_NOTES:
            continue
        moc = _find_moc(folder)
        if moc is None or DATE_MOC.search(moc.stem):
            continue
        c = np.mean(vecs, axis=0)
        topics[folder] = (moc, c / (np.linalg.norm(c) or 1.0))
    return topics


def plan(only: Path | None = None) -> list[tuple[Path, Path, float]]:
    """Return (note, moc, similarity) links to add (for one note if `only`)."""
    notes = _note_vectors()
    topics = _topics(notes)
    folders = list(topics)
    mat = np.stack([topics[f][1] for f in folders])
    links = []
    for p, v in notes.items():
        if only is not None and p != only:
            continue
        sims = mat @ v
        ranked = sorted(zip(folders, sims), key=lambda x: x[1], reverse=True)
        # BGE scores sit in a narrow band, so an absolute cutoff alone links
        # loosely related topics. Compare with how close the note is to its own
        # folder: link elsewhere only if that topic fits about as well.
        home = dict(zip(folders, sims)).get(p.parent)
        floor = MIN_SIM if home is None else max(MIN_SIM, home - HOME_MARGIN)
        # Skip the note's own folder and its ancestors: it is already there.
        extra = [(f, s) for f, s in ranked
                 if f != p.parent and f not in p.parents and s >= floor][:MAX_EXTRA]
        links += [(p, topics[f][0], float(s)) for f, s in extra]
    return links


def link_note(note: Path) -> int:
    """Cross-link one freshly indexed note. Called from remember()."""
    links = plan(only=note)
    for n, moc, _ in links:
        _add_line_under(moc, SECTION, f"- [[{n.stem}]]")
    return len(links)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    links = plan()
    per_moc = defaultdict(int)
    for _, moc, _ in links:
        per_moc[moc] += 1
    print(f"{len(links)} cross-links into {len(per_moc)} MOCs")
    for moc, n in sorted(per_moc.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {moc.relative_to(VAULT_PATH)}")
    if args.apply:
        for note, moc, _ in links:
            _add_line_under(moc, SECTION, f"- [[{note.stem}]]")
        print("applied")
    else:
        import random
        for note, moc, s in random.sample(links, min(25, len(links))):
            print(f"  {s:.2f}  {note.relative_to(VAULT_PATH)}  ->  {moc.stem}")


if __name__ == "__main__":
    main()
