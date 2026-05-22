"""
Hook: Stop
Fires when the agent session ends.
Writes a session summary note to vault/projects/.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from indexer import index_file

VAULT_PATH = Path.home() / "vault"

def main():
    try:
        payload = json.loads(sys.stdin.read())

        # Claude Code Stop hook provides transcript summary
        summary = (
            payload.get("summary")
            or payload.get("stop_reason")
            or payload.get("message", "")
        )

        if not summary or len(summary.strip()) < 20:
            sys.exit(0)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        agent = payload.get("agent", "agent")
        title = f"session-{timestamp}"

        note_path = VAULT_PATH / "projects" / f"{title}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)

        content = f"""---
topic: projects
subtopic: sessions
tags:
  - session
  - {agent}
agent: {agent}
last_updated: {datetime.now().strftime('%Y-%m-%d')}
---

# Session {timestamp}

{summary}
"""
        note_path.write_text(content)
        index_file(str(note_path))

    except Exception:
        sys.exit(0)

if __name__ == "__main__":
    main()
