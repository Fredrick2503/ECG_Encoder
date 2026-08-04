---
name: project-memory
description: Maintain the current state of the project. Use whenever work starts, completes, or changes to ensure project context, progress, and task status remain up to date.
risk: critical
source: project
---

# Project Memory

## Objective

Maintain an accurate and up-to-date view of the project's current state.

This skill is responsible for preserving project continuity across sessions by
tracking progress, completed work, pending work, and important project
decisions.

It does **not** implement features or make architectural decisions.

---

# When to Use

Use this skill whenever:

- starting a new task
- completing a task
- changing implementation status
- updating project progress
- adding or removing tasks
- recording an important project decision

---

# Responsibilities

Maintain:

- Current project status
- Completed tasks
- Tasks in progress
- Pending tasks
- Deferred tasks
- Current milestone
- Important project decisions
- Session summary
- Next recommended task

---

# Workflow

For every update:

1. Read the current project state.
2. Determine what changed.
3. Update the relevant project files.
4. Record important decisions if applicable.
5. Recommend the next logical task.

---

# Files to Maintain

Update these files as needed:

```
.agents/project/

project_state.md
progress.md
backlog.md
milestones.md
decision_log.md
session_summary.md
```

Only modify files affected by the current task.

---

# Rules

- Never assume work is complete.
- Only record completed work with evidence.
- Preserve historical decisions.
- Keep progress consistent across files.
- Prefer updating existing entries over creating duplicates.
- If information is missing, leave a TODO instead of guessing.

---

# Success Criteria

The skill is successful when:

- Project state reflects the latest work.
- Progress is up to date.
- Completed and pending tasks are accurate.
- Important decisions are recorded.
- A clear next task is available.
