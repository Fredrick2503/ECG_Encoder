---
name: build-planner
description: Plan future work, enforce development guardrails, manage logs of completed work, and keep the build on track.
risk: medium
source: project
---

# Build & Guardrails Planner

## Objective

Ensure the development team never loses track of the build by systematically documenting completed tasks, establishing clear development guardrails, and referencing active build guides for all code modifications.

---

# When to Use

Use this skill whenever:
- Starting a new feature or implementing a task to load guardrails.
- Preparing to write/modify code to ensure alignment with build guides.
- Logging completed tasks at the end of a session or feature implementation.
- Proposing future work, structuring roadmaps, or defining new safety/architectural constraints.

---

# Responsibilities

- **Enforce Guardrails**: Define and check compliance with architectural, styling, and testing limits.
- **Log Completed Work**: Append clear, structured entries to `implementation_log.md` and `research_log.md`.
- **Maintain Build Guides**: Keep module-specific active build guides current and descriptive.
- **Plan Next Steps**: Provide a logical implementation sequence for the backlog.

---

# Files to Maintain

### 1. Guardrails
- **Path**: `.agents/project/guardrails.md`
- Contains: Coding styles, naming conventions, max file complexity, architectural constraints, test coverage thresholds, and evaluation requirements.

### 2. Work Logs
- **Paths**: 
  - `.agents/project/research/<module>/implementation_log.md` (Logs exact structural code changes, fixes, and updates)
  - `.agents/project/research/<module>/research_log.md` (Logs experiments, trials, observations, and findings)
- Entry Format:
  ```markdown
  ### [YYYY-MM-DD] [Feature/Task Name]
  - **Author**: @agent-name or USER
  - **Goal**: Brief description of the task objective.
  - **Changes**: Bulleted list of modified/new files and key functions.
  - **Guardrail Checks**: Confirmed compliance with testing, complexity, and linting rules.
  - **Status**: Completed / Paused / In Progress
  ```

### 3. Build Guides
- **Path**: `.agents/project/build_guides/<module>_guide.md`
- Contains: Precise steps, class interfaces, library imports, and structural constraints for building specific modules (e.g., temporal_encoder).

---

# Workflow

### 1. Pre-Build Check (Guiding the Build)
1. Read the target task from `task.md` or `backlog.md`.
2. Locate and read `.agents/project/guardrails.md`.
3. If available, load `.agents/project/build_guides/<module>_guide.md`.
4. Compare requirements against current codebase state to identify constraints.
5. Create/update a detailed step-by-step checklist in `task.md`.

### 2. Post-Build Log (Logging Work)
1. Perform git diff / status check to audit modified files.
2. Confirm that changes comply with all active guardrails.
3. Write/append structured entries to the appropriate `implementation_log.md` and `research_log.md`.
4. Update `.agents/project/build_guides/<module>_guide.md` if the codebase API or structural design changed.
5. Sync the files and update Project Memory.
