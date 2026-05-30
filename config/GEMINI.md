# GEMINI.md — Global Instructions for Antigravity
# Ramzan Ahmad | SE Undergrad | Istanbul

## Third Brain — Mandatory Memory Protocol

You have access to a shared semantic memory vault via the `third-brain` MCP server at `http://127.0.0.1:7891/mcp`.

**This is not optional. Execute on every single prompt, no exceptions.**

### GATE 1 — Before every response: recall

Call `recall` with the user's message before generating any output.

```
third-brain:recall(query="<exact user message>", top_k=5)
```

Read the results. Use them. If empty, proceed normally — but the call is still mandatory.

### GATE 2 — During session: remember immediately

Call `remember` the moment any of these happen:

| Event | Action |
|---|---|
| Decision made (architecture, design, workflow) | call remember immediately |
| Bug found and fixed | call remember immediately |
| User stated a preference or constraint | call remember immediately |
| Project or task started or completed | call remember immediately |
| Anything worth knowing next session | call remember immediately |

```
third-brain:remember(
    content="<specific fact>",
    title="<short title>",
    domain="<domain>",
    agent="antigravity"
)
```

Always pass `domain`. New domains are auto-created. Always set `agent="antigravity"`.

### GATE 3 — After writing any .md file: capture

```
third-brain:capture(file_path="<absolute path>")
```

### GATE 4 — Before ending session: summarize

```
third-brain:summarize_session(
    summary="## What was done\n...\n\n## Decisions made\n...\n\n## Next steps\n...",
    agent="antigravity",
    tags=["topic"]
)
```

---

## Who I Am
- Software Engineering undergrad (final years), Istanbul/Başakşehir
- GitHub: CadPosting | Email: ramzan@hipswan.com
- Languages: Python, JavaScript/TypeScript, C++

## Active Projects
- Third Brain — local semantic memory (`~/.third-brain/`, GitHub: CadPosting/third-brain)
- LeetSkills — professional scenario practice app (`~/Documents/gitramzan/LeetSkills/`)
- NAMD3 simulations on TRUBA cluster (3 protein variants)
- AI623 PA2 — LLM alignment in PyTorch + HuggingFace (no TRL/Trainer)
- SEN3002 — Web Programming coursework (vanilla HTML/CSS/PHP)
- CEN202 — Algorithms course

## Core Rules
- Minimum code that solves the problem — no speculative features
- Touch only what's needed — match existing style
- Explain non-obvious decisions inline — I'm still learning
- Concise responses, no padding

## Domain Reference
| Domain | Content |
|---|---|
| `machine-learning` | ML concepts, experiments |
| `machine-learning/alignment` | RLHF, PPO, GRPO, DPO |
| `hpc` | Cluster, NAMD3 simulations |
| `hpc/truba` | TRUBA cluster specifics |
| `hpc/slurm` | SLURM scripts, safety rules |
| `algorithms` | CS theory, complexity |
| `web-development` | Web projects |
| `projects` | Decisions, session summaries |
