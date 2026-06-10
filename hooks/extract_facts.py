"""
Hook: SessionEnd (fires once per session, unlike Stop which fires per turn).
Extracts durable facts from the session transcript and writes them to the vault
as proper domain notes — instead of the raw topic-dump that summarize.py makes.

Two modes:
  hook mode   (default): read the SessionEnd JSON payload on stdin, do cheap
              checks, spawn the worker DETACHED, and exit 0 immediately so the
              session never blocks on an LLM call.
  worker mode (--worker <transcript_path> <session_id>): the slow part — calls
              `claude -p` (haiku) to extract facts, then POSTs each to the
              server's /remember endpoint (which dedups, Phase 3).

Recursion safety: the worker spawns `claude -p`, which is itself a Claude Code
run. We prevent that run from triggering more extraction two ways:
  1. env TB_NO_EXTRACT=1 is set on the worker (inherited by claude -p); both this
     hook and summarize.py bail when they see it.
  2. the `claude -p` call uses --setting-sources "" so it loads no hooks at all.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# NOTE: keep module-level imports pure-stdlib. The hook path must return in
# milliseconds; importing summarize/indexer here would load the embedding and
# reranker models (~15s) and block session close. The transcript parser is
# imported lazily inside run_worker() instead.

SERVER = "http://127.0.0.1:7891"
MODEL = "haiku"
MAX_FACTS = 5
MIN_USER_TURNS = 2          # skip trivial sessions
MAX_TRANSCRIPT_CHARS = 24000  # cap what we feed the model (~tail of the session)
CLAUDE_TIMEOUT = 90         # seconds for the extraction call
LOG = Path.home() / ".third-brain" / "extract.log"
MARKER_DIR = Path.home() / ".third-brain" / "extracted"  # one marker per session
VAULT = Path.home() / "vault"

EXTRACT_PROMPT = """You are a memory archivist. From the coding-session transcript below, extract ONLY durable facts worth remembering across future sessions: decisions made, bugs found and how they were fixed, constraints or preferences the user stated, or things learned about the system. Ignore chit-chat, transient status, and anything specific to this one session's mechanics.

Pick `domain` from this list of EXISTING domains whenever one fits — only invent a new kebab-case domain if none apply:
{domains}

Return STRICT JSON only — a list of 0 to {max_facts} objects, nothing else:
[{{"fact": "<one self-contained fact, with the why>", "domain": "<domain from the list, or a new kebab-case topic>", "title": "<short note title>"}}]

If nothing is durable, return []. No prose, no markdown fences — just the JSON array.

TRANSCRIPT:
{transcript}
"""


def read_transcript(transcript_path: str) -> str:
    """Parse a CC .jsonl transcript into USER:/ASSISTANT: lines. Stdlib only —
    a trimmed copy of summarize.read_transcript_jsonl so the worker doesn't have
    to import the model-loading indexer stack."""
    import re
    path = Path(transcript_path).expanduser()
    if not path.exists():
        return ""
    lines = []
    try:
        with open(path) as f:
            for raw in f:
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {})
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    text = str(content)
                text = re.sub(r"<[a-z_-]+>[^<]*</[a-z_-]+>\s*", "", text.strip()).strip()
                if not text or text.startswith("<local-command") or text.startswith("<command-name"):
                    continue
                lines.append(f"{role.upper()}: {text}")
    except Exception:
        return ""
    return "\n\n".join(lines)


def existing_domains() -> list[str]:
    """Top-level vault folders, so the model reuses domains instead of drifting."""
    try:
        return sorted(
            p.name for p in VAULT.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    except Exception:
        return []


def log(msg: str) -> None:
    try:
        with open(LOG, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------- worker mode
def run_worker(transcript_path: str, session_id: str) -> None:
    # Idempotency: extract a given session at most once. SessionEnd normally
    # fires once, but "resume" can re-fire it; and because the LLM picks the
    # domain non-deterministically, re-extracting would land the same fact in a
    # different folder where folder-scoped dedup can't catch it. The marker is
    # the session-level guard (mirrors Phase 1's session-keyed note).
    safe_sid = "".join(c for c in session_id if c.isalnum())[:32] or "nosession"
    marker = MARKER_DIR / safe_sid
    if marker.exists():
        log(f"[worker {session_id}] already extracted, skip")
        return

    transcript = read_transcript(transcript_path)
    if not transcript:
        log(f"[worker {session_id}] empty transcript, skip")
        return
    if sum(1 for l in transcript.splitlines() if l.startswith("USER:")) < MIN_USER_TURNS:
        log(f"[worker {session_id}] < {MIN_USER_TURNS} user turns, skip")
        return

    # Mark BEFORE the LLM call: if extraction half-succeeds we still won't
    # re-run and duplicate. A lost session is cheaper than duplicate notes.
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now().isoformat(timespec="seconds"))

    # Feed the tail — the end of a session holds the conclusions.
    tail = transcript[-MAX_TRANSCRIPT_CHARS:]
    domains = existing_domains()
    domain_list = ", ".join(domains) if domains else "(none yet)"
    prompt = EXTRACT_PROMPT.format(
        max_facts=MAX_FACTS, transcript=tail, domains=domain_list
    )

    env = {**os.environ, "TB_NO_EXTRACT": "1"}  # belt: stop nested extraction
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", MODEL,
             "--output-format", "json", "--setting-sources", ""],  # suspenders: no hooks
            capture_output=True, text=True, timeout=CLAUDE_TIMEOUT, env=env,
        )
    except subprocess.TimeoutExpired:
        log(f"[worker {session_id}] claude -p timed out")
        return
    if proc.returncode != 0:
        log(f"[worker {session_id}] claude -p rc={proc.returncode}: {proc.stderr[:200]}")
        return

    # `claude -p --output-format json` wraps the model text in a {"result": ...}
    # envelope; the model text itself must be our JSON array.
    try:
        envelope = json.loads(proc.stdout)
        raw = envelope.get("result", "").strip()
        # tolerate accidental ```json fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("["):]
        facts = json.loads(raw)
        if not isinstance(facts, list):
            raise ValueError("not a list")
    except Exception as e:
        log(f"[worker {session_id}] unparseable model output ({e}): {proc.stdout[:200]}")
        return

    written = 0
    for f in facts[:MAX_FACTS]:
        if not isinstance(f, dict):
            continue
        content = (f.get("fact") or "").strip()
        domain = (f.get("domain") or "").strip() or None
        title = (f.get("title") or "").strip() or None
        if not content:
            continue
        body = json.dumps({
            "content": content, "domain": domain, "title": title,
            "agent": "claude-code-extractor",
        }).encode()
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{SERVER}/remember", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read())
            written += 1
            log(f"[worker {session_id}] wrote: {res.get('path','?')} "
                f"deduped={bool(res.get('deduped_into'))}")
        except Exception as e:
            log(f"[worker {session_id}] /remember failed: {e}")

    log(f"[worker {session_id}] done: {written}/{len(facts)} facts written")


# ------------------------------------------------------------------ hook mode
def run_hook() -> None:
    # Recursion guard: if we're inside an extractor-spawned claude run, do nothing.
    if os.environ.get("TB_NO_EXTRACT"):
        sys.exit(0)
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        sys.exit(0)

    transcript_path = payload.get("transcript_path", "")
    session_id = payload.get("session_id", "") or "nosession"
    if not transcript_path or not Path(transcript_path).expanduser().exists():
        sys.exit(0)

    # Spawn the worker fully detached and return at once — never block the close.
    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()),
             "--worker", transcript_path, session_id],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        log(f"[hook {session_id}] failed to spawn worker: {e}")
    sys.exit(0)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3])
    else:
        run_hook()


if __name__ == "__main__":
    main()
