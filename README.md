# Third Brain

Local semantic memory server for AI agents. One knowledge base, every agent, zero cloud.

Works with Claude Code (terminal), Antigravity (Gemini, Claude, GPT, Codex), and Gemini CLI standalone.

## How it works

Every agent session automatically reads relevant notes before responding and writes learnings back after. You never manually save or file anything. Open the vault in Obsidian to see the full knowledge graph.

```
Your question
     ↓
Hooks inject relevant vault notes as context
     ↓
Agent answers with full memory
     ↓
Hooks save learnings back to vault
     ↓
Next session, any agent, same context
```

## Stack

| Component | Role |
|---|---|
| `bge-small-en-v1.5` | Local embeddings (~130MB, fully offline) |
| LanceDB | Vector store (single file, zero config) |
| BM25 | Keyword search (catches exact terms) |
| RRF | Merges vector + keyword rankings |
| FlashRank | Reranks top 20 results (~4MB, <20ms) |
| Graphiti | Knowledge graph (entities + relationships + time) |
| watchdog | Auto-indexes anything you write in Obsidian |
| FastMCP | MCP server (agent interface) |

## Setup

### 1. Install

```bash
git clone https://github.com/CadPosting/third-brain ~/.third-brain
cd ~/.third-brain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create your vault

```bash
cp -r vault ~/vault
```

Open `~/vault` in Obsidian as a new vault.

### 3. Configure environment (optional — for knowledge graph layer)

```bash
cp .env.example ~/.third-brain/.env
# Edit ~/.third-brain/.env and set NEO4J_PASSWORD if you have Neo4j running
# Leave blank to run without the graph layer — vector + BM25 + reranker still work
```

### 4. Start the server and watcher

```bash
python ~/.third-brain/server.py &
python ~/.third-brain/watcher.py &
```

### 4. Connect Claude Code (terminal)

Copy MCP config:
```bash
cp config/claude-mcp.json ~/.claude/mcp.json
```

Merge hooks into `~/.claude/settings.json` from `config/claude-settings.json`.

### 5. Connect Antigravity

Copy MCP config:
```bash
cp config/antigravity-mcp.json ~/.antigravity/mcp_config.json
```

Copy the skill:
```bash
mkdir -p ~/.antigravity/skills/third-brain
cp config/antigravity-skill.md ~/.antigravity/skills/third-brain/SKILL.md
```

All models in Antigravity (Gemini, Claude, GPT, Codex) now share the same memory.

### 6. Connect Gemini CLI (standalone)

Merge `config/gemini-settings.json` into `~/.gemini/settings.json`.

## Vault structure

```
vault/
├── HOME.md
├── web-development/
│   ├── frontend/
│   └── backend/
├── machine-learning/
│   ├── alignment/   (RLHF, PPO, GRPO, DPO)
│   ├── models/
│   └── math/
├── hpc/
│   ├── truba/
│   │   └── variants/
│   └── slurm/
├── algorithms/
└── projects/        (session summaries auto-appear here)
```

Every domain has a `_MOC.md` (Map of Content) — the index note that creates the graph web you see in Obsidian's graph view.

## MCP tools exposed to agents

| Tool | What it does |
|---|---|
| `recall(query, top_k, topic_filter)` | Search memory semantically + by keyword |
| `remember(content, title, agent, tags)` | Save to vault, auto-classify |
| `graph_traverse(entity, depth)` | Find related concepts in knowledge graph |
| `list_map(topic)` | Show vault structure |

## Cost

$0. Runs fully offline. Forever.
