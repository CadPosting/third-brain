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
TIMEOUT = 8  # seconds — must return fast or CC drops the hook output


def main():
    try:
        payload = json.loads(sys.stdin.read())
        prompt = payload.get("prompt", "") or payload.get("user_prompt", "")
        if not prompt:
            sys.exit(0)

        body = json.dumps({"query": prompt, "top_k": 3}).encode()
        req = urllib.request.Request(
            SEARCH_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            results = json.loads(resp.read())

        if not results or not isinstance(results, list):
            sys.exit(0)

        context_lines = ["--- Third Brain context ---"]
        for r in results:
            source = Path(r.get("source_path", "unknown")).name
            text = r.get("text", "").strip()
            if text:
                context_lines.append(f"[{source}] {text}")
        context_lines.append("--- end context ---")

        output = {"additionalContext": "\n".join(context_lines)}
        print(json.dumps(output))

    except Exception:
        # Never crash the agent session
        sys.exit(0)


if __name__ == "__main__":
    main()
