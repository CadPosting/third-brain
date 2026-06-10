"""
Hook: UserPromptSubmit
Fires before the agent processes your message.
POSTs to the always-running Third Brain server's /search endpoint
so the embedding model is never loaded cold in this process.
Fast path: ~50ms instead of ~20s.
"""
import sys
import json
import urllib.request
from pathlib import Path

SEARCH_URL = "http://127.0.0.1:7891/search"
TIMEOUT = 12  # seconds — must return fast or CC drops the hook output.
# Steady-state /search is ~4-6s; 12s leaves headroom so a slow query
# doesn't silently drop recall context (the except clause exits 0).

# Relevance gate: only inject memory the cross-encoder is confident about.
# Calibrated 2026-06-10 — irrelevant queries score <=0.0001, real hits 0.95-0.999,
# so 0.4 sits in the empty gap. Below this, inject nothing (the common case).
MIN_SCORE = 0.4
# Cap injected context so even a relevant prompt can't dump the old ~979-token
# median. ~4 chars/token, so 700 tokens ~= 2800 chars.
TOKEN_BUDGET = 700
CHARS_PER_TOKEN = 4


def main():
    try:
        payload = json.loads(sys.stdin.read())
        prompt = payload.get("prompt", "") or payload.get("user_prompt", "")
        if not prompt:
            sys.exit(0)

        body = json.dumps({"query": prompt, "top_k": 3, "min_score": MIN_SCORE}).encode()
        req = urllib.request.Request(
            SEARCH_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            results = json.loads(resp.read())

        # Server already gated on MIN_SCORE; empty means nothing relevant —
        # inject nothing. This is the common case and the whole point.
        if not results or not isinstance(results, list):
            sys.exit(0)

        # Inject in (descending-score) order until the token budget is spent.
        budget_chars = TOKEN_BUDGET * CHARS_PER_TOKEN
        context_lines = ["--- Third Brain context ---"]
        used = 0
        for r in results:
            text = r.get("text", "").strip()
            if not text:
                continue
            source = Path(r.get("source_path", "unknown")).name
            remaining = budget_chars - used
            if remaining <= 0:
                break
            # Truncate the chunk if it overflows, rather than dropping it whole —
            # a partial top hit beats nothing.
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "…"
            line = f"[{source}] {text}"
            context_lines.append(line)
            used += len(line)
        # Only the header was added → nothing usable survived.
        if len(context_lines) == 1:
            sys.exit(0)
        context_lines.append("--- end context ---")

        output = {"additionalContext": "\n".join(context_lines)}
        print(json.dumps(output))

    except Exception:
        # Never crash the agent session
        sys.exit(0)


if __name__ == "__main__":
    main()
