"""
Hook: UserPromptSubmit
Fires before the agent processes your message.
Searches Third Brain for relevant context and injects it silently.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from indexer import vector_search, bm25_search, rrf_merge
from reranker import rerank

def main():
    try:
        payload = json.loads(sys.stdin.read())
        prompt = payload.get("prompt", "") or payload.get("user_prompt", "")
        if not prompt:
            sys.exit(0)

        vec = vector_search(prompt, top_k=20)
        bm25 = bm25_search(prompt, top_k=20)
        merged = rrf_merge(vec, bm25)
        results = rerank(prompt, merged, top_k=3)

        if not results:
            sys.exit(0)

        context_lines = ["--- Third Brain context ---"]
        for r in results:
            source = Path(r.get("source_path", "unknown")).name
            context_lines.append(f"[{source}] {r.get('text', '').strip()}")
        context_lines.append("--- end context ---")

        # Output additionalContext for Claude Code hook protocol
        output = {"additionalContext": "\n".join(context_lines)}
        print(json.dumps(output))

    except Exception:
        # Never crash the agent session
        sys.exit(0)

if __name__ == "__main__":
    main()
