"""
Hook: PostToolUse
Fires after every tool call (Write, Edit, Bash, etc.).
Intercepts file writes and auto-saves content to Third Brain.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from indexer import index_file

# Only capture these tool types — ignore reads, searches, etc.
CAPTURE_TOOLS = {"Write", "Edit", "NotebookEdit"}

def main():
    try:
        payload = json.loads(sys.stdin.read())
        tool_name = payload.get("tool_name", "")

        if tool_name not in CAPTURE_TOOLS:
            sys.exit(0)

        tool_input = payload.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path or not file_path.endswith(".md"):
            sys.exit(0)

        # Re-index the file that was just written/edited
        chunks = index_file(file_path)

        # Log quietly — don't pollute agent output
        if chunks > 0:
            result = {"status": "indexed", "file": file_path, "chunks": chunks}
            # Write to stderr so it doesn't interfere with hook output
            print(json.dumps(result), file=sys.stderr)

    except Exception:
        sys.exit(0)

if __name__ == "__main__":
    main()
