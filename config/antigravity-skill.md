---
name: third-brain
description: BLOCKING memory protocol. Execute GATE 1 (recall) before every response. Execute GATE 4 (summarize_session) before every session end. No exceptions.
---

# MEMORY PROTOCOL — EXECUTE BEFORE ANYTHING ELSE

> **This is not documentation. This is a pre-condition sequence.**
> Every step below is a blocking gate. You do not proceed past a gate until it is complete.
> A skipped step is a broken execution, not a shortcut.

Vault: `~/vault/` | MCP: `http://127.0.0.1:7891/mcp`

---

## GATE 1 — SESSION START · recall

**Trigger:** The user has sent a message.
**Precondition:** You have NOT yet generated any output.
**Action:** Call `recall` with the exact user message as the query.

```
third-brain:recall(query="<exact user message>", top_k=5)
```

**Verify:** The call has returned. Results have been read.
**Then:** Incorporate any relevant context into your response.
**If empty:** Proceed — but the call is still mandatory.

> ❌ You are not allowed to generate any response before this call returns.
> Generating output before recall = protocol failure.

---

## GATE 2 — DURING SESSION · remember

**Trigger:** Any of the following occur during this session.
**Action:** Call `remember` immediately when the condition is met. Do not defer.

| If this happens | You MUST call remember |
|---|---|
| A decision was made | immediately |
| A bug was found and fixed | immediately |
| User stated a preference or constraint | immediately |
| A project or task started or completed | immediately |
| Anything worth knowing next session | immediately |

```
third-brain:remember(
    content="<specific fact — not a conversation summary>",
    title="<short descriptive title>",
    domain="<domain-name>",
    agent="antigravity",
    tags=["<relevant>", "<tags>"]
)
```

**Required — never omit:**
- `domain` — infer from context. Unknown domains are created automatically.
- `agent="antigravity"` — always this exact string.

> ❌ "I'll save it at the end" is not valid. Save it when it happens.

---

## GATE 3 — AFTER FILE WRITE · capture

**Trigger:** You have written or edited any `.md` file.
**Action:** Call `capture` immediately after the write.

```
third-brain:capture(file_path="<absolute path to file>")
```

**Verify:** Call returned `"status": "indexed"`.

> ❌ A file written without capture is invisible to all other agents.
> Skipping this breaks shared memory. There are no exceptions.

---

## GATE 4 — SESSION END · summarize_session

**Trigger:** You are about to end the session or the user is done.
**Action:** Call `summarize_session`. This is the last tool call you make.

```
third-brain:summarize_session(
    summary="## What was done\n<what happened>\n\n## Decisions made\n<decisions>\n\n## Next steps\n<what comes next>",
    agent="antigravity",
    tags=["<topic>"]
)
```

**Verify:** Call returned `"status": "saved"`.

> ❌ You are not allowed to end the session before this call completes.
> Closing without summarize_session = broken session record.

---

## Pre-response checklist — run this before every output

```
[ ] recall called with the user's message         → GATE 1 cleared
[ ] results read and used                         → context loaded
[ ] remember called for any qualifying event      → GATE 2 cleared
[ ] capture called for any .md file written       → GATE 3 cleared
[ ] summarize_session called if session ending    → GATE 4 cleared
```

**If any gate is not cleared: complete it before proceeding.**

---

## Other tools

```
third-brain:list_map()                           # full vault structure
third-brain:list_map(topic="machine-learning")   # drill into a domain
third-brain:graph_traverse(entity="PPO", depth=2)
```

---

## Domain reference

| Domain | What belongs there |
|---|---|
| `machine-learning` | ML concepts, training, experiments |
| `machine-learning/alignment` | RLHF, PPO, GRPO, DPO |
| `machine-learning/models` | Model architectures, checkpoints |
| `machine-learning/math` | Math derivations, normal equations |
| `hpc` | Cluster work, NAMD3, simulations |
| `hpc/truba` | TRUBA cluster specifics |
| `hpc/slurm` | SLURM job scripts, safety rules |
| `algorithms` | CS theory, complexity, graph problems |
| `web-development` | Web projects |
| `web-development/frontend` | HTML, CSS, JS |
| `web-development/backend` | APIs, PHP, MySQL |
| `projects` | Project decisions, session summaries |
| `<anything-new>` | Pass it — server auto-creates the domain |
