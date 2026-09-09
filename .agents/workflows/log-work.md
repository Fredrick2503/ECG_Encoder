---
description: Log all works done to ensure the build progress is preserved.
---

# Workflow: Log Work

This workflow ensures every code modification, feature addition, and experiment is logged immediately, preventing the build from losing track of history and state.

---

## Execution Sequence

### 1. Build Planner
- **Agent**: `@build-planner`
- **Skill**: `build-planner`
- **Actions**:
  1. Inspect files modified during the task using system tools (e.g., git status/diff).
  2. Verify that all modified files comply with the active guardrails in `.agents/project/guardrails.md`.
  3. Identify the target module's research directory (e.g., `.agents/project/research/temporal_encoder/`).
  4. Write/append a detailed entry to `implementation_log.md` with:
     - Timestamp
     - Author (e.g., `@build-planner` or user)
     - Brief objective description
     - Bulleted list of modified/new files and functions
     - Guardrails check status (e.g., Pass)
  5. If an experiment trial was run, write/append details to `research_log.md` with configurations, metrics, and outcomes.

### 2. Research Documentation
- **Agent**: `@thesis-doc` / `@research`
- **Skill**: `thesis-writer` / `research-knowledge-manager`
- **Actions**:
  1. Update research notes or shortcomings if relevant.

### 3. Project Memory
- **Agent**: `@pm` / `@mastermind`
- **Skill**: `project-memory`
- **Actions**:
  1. Update workspace status to reflect the new log state.
