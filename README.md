# Third Brain

Local semantic memory server for AI agents. One knowledge base, every agent, zero cloud.

Agents recall relevant notes before every response, write new facts during the session, and summarize what happened at the end — automatically. You never manage the vault yourself.

Works with **Claude Code**, **Antigravity** (Gemini, Claude, GPT), and **Codex**.

---

## How it works

```
You send a message
        ↓
inject.py searches vault → injects top-3 relevant notes silently into the prompt
        ↓
Agent responds with full memory of past sessions
        ↓
Agent writes new facts/decisions to ~/vault/ via the remember MCP tool
        ↓
watcher.py detects the file change → indexes it within 2 seconds
        ↓
Session ends → summarize.py reads the transcript → writes a session note → indexed
        ↓
Next session, any agent, same shared context
```

Every agent reads from and writes to the same vault. A decision made in a Claude Code session is available to Antigravity in the next one.

---

## Stack

| Component | Role |
|---|---|
| `bge-small-en-v1.5` | Local embeddings (~130MB, fully offline) |
| LanceDB | Vector store (single file, zero config) |
| BM25 | Keyword search alongside vector — catches exact terms vectors miss |
| RRF | Fuses vector + keyword rankings before reranking |
| FlashRank `ms-marco-MiniLM-L-12-v2` | Reranks top-20 results (~4MB) |
| watchdog | Auto-indexes any vault file change within 2 seconds |
| FastMCP 3.3.1 | HTTP MCP server all agents connect to |
| Graphiti | Knowledge graph layer (optional — requires Neo4j) |

---

## Architecture

### Services

Two persistent background services. Install once, they restart automatically.

| Service | What it does |
|---|---|
| `third-brain.service` | MCP server on `http://127.0.0.1:7891/mcp` + `/search` endpoint |
| `third-brain-watcher.service` | Watches `~/vault/` and re-indexes any `.md` change within 2s |

### HTTP endpoints

The server exposes two endpoints:

| Endpoint | Protocol | Used by |
|---|---|---|
| `POST /mcp` | MCP over SSE | Claude Code, Antigravity, Codex (MCP clients) |
| `POST /search` | Plain JSON | `inject.py` hook (fast vault search without SSE handshake) |

`/search` request: `{"query": "...", "top_k": 3}`
`/search` response: `[{"source_path": "...", "text": "..."}, ...]`

### Claude Code hooks

Three hooks fire automatically on every Claude Code session — no commands needed.

| Hook | Event | What it does |
|---|---|---|
| `hooks/inject.py` | `UserPromptSubmit` | POSTs to `/search`, injects top-3 chunks as `additionalContext` |
| `hooks/capture.py` | `PostToolUse` (Write/Edit) | Re-indexes any `.md` file the agent just wrote |
| `hooks/summarize.py` | `Stop` | Reads session transcript from `transcript_path`, writes `vault/projects/session-*.md` |

**inject.py:** hooks run as short-lived subprocesses. Loading the embedding model in-process takes ~20s and gets dropped by Claude Code's timeout. The hook POSTs to the already-running server's `/search` endpoint instead (~4-5s warm). The server pre-warms the embedder and reranker on startup.

**summarize.py:** Claude Code's Stop hook sends `{"session_id": "...", "transcript_path": "/path/to/session.jsonl", "cwd": "..."}` — it does NOT send transcript content inline. The hook reads the `.jsonl` file at `transcript_path` directly.

### MCP tools

All agents call these via MCP.

| Tool | Parameters | What it does |
|---|---|---|
| `recall` | `query, top_k, topic_filter` | Hybrid vector+BM25+rerank search. Returns scored results. |
| `remember` | `content, title, domain, agent, tags` | Saves a note. Auto-creates domain folder + MOC if new. |
| `capture` | `file_path` | Re-indexes a `.md` file immediately. |
| `summarize_session` | `summary, agent, tags` | Writes a session summary to `vault/projects/`. |
| `list_map` | `topic` (optional) | Shows vault folder structure and note counts. |
| `graph_traverse` | `entity, depth` | Traverses knowledge graph from an entity (requires Neo4j). |

### Memory protocol (four gates)

Agents follow this on every session. Each gate is a blocking precondition — not a suggestion.

| Gate | When | Action |
|---|---|---|
| GATE 1 | Before every response | `recall` with the user's exact message |
| GATE 2 | During session, on qualifying events | `remember` immediately — never defer |
| GATE 3 | After writing any `.md` file | `capture` the file path |
| GATE 4 | Before ending the session | `summarize_session` |

Qualifying events for GATE 2: decision made, bug found and fixed, preference stated, project started or completed, anything worth knowing next session.

For **Claude Code**: Gates 1, 3, and 4 run automatically via hooks. Gate 2 (`remember`) must be called by the agent when a qualifying event occurs.

For **Antigravity**: all four gates are in the skill file (`config/antigravity-skill.md`) framed as blocking preconditions. The agent must call each tool itself — there are no hooks.

---

## Vault structure

```
~/vault/                              — fully local, never pushed to git
├── HOME.md                           — vault entry point
├── algorithms/
├── hpc/
│   ├── slurm/
│   └── truba/
│       └── variants/
├── machine-learning/
│   ├── alignment/   (RLHF, PPO, GRPO, DPO)
│   ├── math/
│   └── models/
├── projects/                         — session notes auto-appear here
│   └── session-YYYY-MM-DD_HH-MM.md
└── web-development/
    ├── backend/
    └── frontend/
```

New domains are created automatically when an agent calls `remember` with a new `domain` value. The server creates the folder, a MOC file, and cross-links to related domains.

Open `~/vault/` in Obsidian to browse the full knowledge graph.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/CadPosting/third-brain ~/.third-brain
cd ~/.third-brain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Python 3.10+ required. The venv **must** live at `~/.third-brain/.venv` — service files and hooks reference this path directly.

### 2. Create your vault

```bash
cp -r vault ~/vault
```

### 3. Configure environment (optional — knowledge graph layer)

```bash
cp .env.example ~/.third-brain/.env
# Set NEO4J_PASSWORD if you have Neo4j running locally
# Leave blank to skip the graph layer — vector + BM25 + reranker still work
```

### 4. Install and start the background services

```bash
# Copy service files
cp config/third-brain.service ~/.config/systemd/user/
cp config/third-brain-watcher.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable third-brain third-brain-watcher
systemctl --user start third-brain third-brain-watcher

# Verify
systemctl --user status third-brain
systemctl --user status third-brain-watcher
```

The server pre-warms the embedder and reranker on startup — first search takes ~4-5s, subsequent calls are similar (CPU inference per query is not cached).

### 5. Connect Claude Code

**Step 1 — MCP config** (`~/.claude/mcp.json`):

```bash
cp config/claude-mcp.json ~/.claude/mcp.json
```

This tells Claude Code to connect to the running server via HTTP. The config uses `"type": "http"` — do **not** use `"command"`/`"args"` (stdio transport). The server runs HTTP, not stdio. Using the wrong transport causes a silent connection failure where MCP tools never appear in the session.

```json
{
  "mcpServers": {
    "third-brain": {
      "type": "http",
      "url": "http://127.0.0.1:7891/mcp",
      "description": "Third Brain — shared semantic memory for all agents"
    }
  }
}
```

**Step 2 — Hooks** (`~/.claude/settings.json`):

Merge the `hooks` block from `config/claude-settings.json` into your existing `~/.claude/settings.json`. Do **not** replace the whole file — only add the `hooks` key.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.third-brain/.venv/bin/python ~/.third-brain/hooks/inject.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{ "type": "command", "command": "~/.third-brain/.venv/bin/python ~/.third-brain/hooks/capture.py" }]
      },
      {
        "matcher": "Edit",
        "hooks": [{ "type": "command", "command": "~/.third-brain/.venv/bin/python ~/.third-brain/hooks/capture.py" }]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "~/.third-brain/.venv/bin/python ~/.third-brain/hooks/summarize.py"
          }
        ]
      }
    ]
  }
}
```

**Step 3 — CLAUDE.md**:

Add memory instructions to `~/.claude/CLAUDE.md` telling the agent to call `remember` with `domain` and `agent="claude-code"` on qualifying events. See `config/antigravity-skill.md` as a template.

**Restart Claude Code** after making these changes. MCP connections are established at session start.

### 6. Connect Antigravity

**Step 1 — MCP config** (`~/.gemini/antigravity/mcp_config.json`):

Merge the `third-brain` entry into your existing `mcp_config.json` under `mcpServers`:

```json
{
  "third-brain": {
    "serverURL": "http://127.0.0.1:7891/mcp",
    "description": "Third Brain — shared semantic memory."
  }
}
```

> **Important:** Antigravity requires `"serverURL"` (camelCase) for HTTP MCP servers — not `"url"`. The wrong key produces a silent connection failure.

**Step 2 — Skill**:

```bash
mkdir -p ~/.gemini/antigravity/skills/third-brain
cp config/antigravity-skill.md ~/.gemini/antigravity/skills/third-brain/third-brain.md
```

**Step 3 — GEMINI.md**:

```bash
cp config/GEMINI.md ~/GEMINI.md
```

Antigravity reads `~/GEMINI.md` automatically on every session. This file embeds the recall requirement as a top-level instruction so the agent calls `recall` before every response without needing to be asked. Without this file, the skill exists but is treated as on-demand rather than always-on.

### 7. Connect Codex

**MCP**:

```bash
cp config/codex-mcp.json ~/.codex/.mcp.json
```

**Skill**:

```bash
mkdir -p ~/.codex/skills/third-brain
cp config/antigravity-skill.md ~/.codex/skills/third-brain/SKILL.md
sed -i 's/agent="antigravity"/agent="codex"/g' ~/.codex/skills/third-brain/SKILL.md
```

---

## Verifying it works

```bash
# Both services running
systemctl --user is-active third-brain third-brain-watcher

# Server startup log — index loaded cleanly or rebuilt
journalctl --user -u third-brain --no-pager | grep "\[startup\]"
# Expected: "[startup] Index OK." or rebuild ending with "Rebuilt index: N chunks"

# Test the /search endpoint
curl -s http://127.0.0.1:7891/search \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 2}'
# Expected: JSON array (may be empty if vault is new)
```

After connecting an agent, start a session and send a message. The agent should call `recall` before responding. After the session ends, check `~/vault/projects/` — a `session-YYYY-MM-DD_HH-MM.md` file should appear.

---

## Troubleshooting

### MCP tools not appearing in Claude Code

Most common cause: wrong transport in `~/.claude/mcp.json`. Must use `"type": "http"` — not `"command"`/`"args"`. Using `command` tells Claude Code to spawn a subprocess expecting stdio, but the server speaks HTTP. Connection fails silently and no tools appear. Fix the config and restart Claude Code.

### inject.py not injecting context

The hook POSTs to `/search` on the running server. If no context is injected:

```bash
systemctl --user is-active third-brain
curl -s http://127.0.0.1:7891/search -X POST \
  -H "Content-Type: application/json" -d '{"query":"test","top_k":1}'
```

If the server is down, start it: `systemctl --user start third-brain`

### Session notes not appearing in vault/projects/

The Stop hook reads `transcript_path` from the payload — a path to the `.jsonl` session file. Check:

1. Server is running (summarize.py uses the server's venv for indexing)
2. `~/vault/projects/` exists
3. The session had at least 2 user messages (trivial sessions are skipped)

Claude Code (VSCode extension) stores transcripts at `~/Documents/claude/projects/-home-<user>/<session-id>.jsonl`. The `transcript_path` in the hook payload resolves this correctly.

### LanceDB corruption / "Not Found" errors

The server auto-detects and rebuilds on startup. Check the journal:

```bash
journalctl --user -u third-brain --no-pager | grep "\[startup\]"
```

To force a manual rebuild:

```bash
systemctl --user stop third-brain third-brain-watcher
rm -rf ~/.third-brain/lancedb/
systemctl --user start third-brain third-brain-watcher
```

> Always delete the **entire** `lancedb/` directory — never just a subdirectory. Partial deletes leave manifests pointing at missing files.

### Antigravity not recalling automatically

Without `~/GEMINI.md`, the skill is treated as on-demand. Make sure you ran Step 3 of the Antigravity setup:

```bash
cp config/GEMINI.md ~/GEMINI.md
```

---

## Privacy

Everything stays local. `~/vault/` is in `.gitignore` — no notes, session summaries, or Obsidian config ever leave the machine. The embeddings model runs locally with no API calls.

---

## Cost

$0. Fully offline.
