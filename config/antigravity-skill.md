# Third Brain Skill

## Purpose
Give this agent persistent memory across all sessions via Third Brain.

## On every task start
1. Call `third-brain:recall` with the current task description as the query
2. Read the returned context before doing anything else
3. Use the context to inform your approach — don't re-derive what's already known

## On every task end
1. Call `third-brain:remember` with a concise summary of what was learned or done
2. Include relevant tags (topic, subtopic, tools used)
3. Let Third Brain auto-classify the note — do not specify a path manually

## On "what do you know about X?" questions
1. Call `third-brain:recall` with X as the query
2. If relationships matter, also call `third-brain:graph_traverse` with X
3. Synthesize the results — don't just dump raw notes

## Rules
- Never skip the recall step at task start, even for simple tasks
- Always use `agent: antigravity` when calling remember
- If recall returns nothing, proceed normally — the vault may just not have that topic yet
