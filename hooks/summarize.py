"""
Hook: Stop
Fires when the Claude Code session ends.
CC sends: { "session_id", "transcript_path", "cwd", "hook_event_name" }
Reads the transcript from transcript_path (a .jsonl file), then writes
a structured session note to vault/projects/ and indexes it.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from indexer import index_file

VAULT_PATH = Path.home() / "vault"
MIN_MESSAGES = 2  # skip trivial sessions


def read_transcript_jsonl(transcript_path: str) -> str:
    """Parse a Claude Code .jsonl transcript into readable USER:/ASSISTANT: lines."""
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
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    text = str(content)

                text = text.strip()
                # Strip IDE/system noise tags
                import re
                text = re.sub(r"<[a-z_-]+>[^<]*</[a-z_-]+>\s*", "", text).strip()
                if not text or text.startswith("<local-command") or text.startswith("<command-name"):
                    continue

                lines.append(f"{role.upper()}: {text}")
    except Exception:
        return ""

    return "\n\n".join(lines)


def summarize_transcript(transcript: str) -> str:
    lines = transcript.splitlines()
    user_lines = [l for l in lines if l.startswith("USER:")]
    assistant_lines = [l for l in lines if l.startswith("ASSISTANT:")]

    if len(user_lines) < MIN_MESSAGES:
        return ""

    topic = user_lines[0].replace("USER:", "").strip()[:120]
    asked = "\n".join(f"- {l.replace('USER:','').strip()}" for l in user_lines[:8])
    outcome = assistant_lines[-1].replace("ASSISTANT:", "").strip()[:500] if assistant_lines else ""

    return f"## Topic\n{topic}\n\n## What was discussed\n{asked}\n\n## Outcome\n{outcome}"


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)

        payload = json.loads(raw)

        # Primary path: CC sends transcript_path pointing to the .jsonl file
        transcript_path = payload.get("transcript_path", "")
        if transcript_path:
            transcript = read_transcript_jsonl(transcript_path)
        else:
            # Fallback: reconstruct path from session_id + cwd
            session_id = payload.get("session_id", "")
            cwd = payload.get("cwd", "")
            if session_id and cwd:
                slug = cwd.replace("/", "-").lstrip("-")
                guessed = Path.home() / ".claude" / "projects" / f"-{slug}" / f"{session_id}.jsonl"
                transcript = read_transcript_jsonl(str(guessed))
            else:
                transcript = ""

        if not transcript:
            sys.exit(0)

        summary = summarize_transcript(transcript)
        if not summary:
            sys.exit(0)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        note_path = VAULT_PATH / "projects" / f"session-{timestamp}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)

        note_path.write_text(f"""---
topic: projects
subtopic: sessions
tags: [session, auto]
agent: claude-code
last_updated: {datetime.now().strftime('%Y-%m-%d')}
---

# Session {timestamp}

{summary}

---
*[[Projects MOC]]*
""")
        index_file(str(note_path))

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
