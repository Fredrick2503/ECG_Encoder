---
description: Refer to requirements, load guardrails, and guide the building/development of modules and features.
---

# Workflow: Guide Building

This workflow guides the implementation of features and modules by referencing existing active build guides, loading project guardrails, and validating work before marking tasks complete.

---

## Execution Sequence

### 1. Build Planner
- **Agent**: `@build-planner`
- **Skill**: `build-planner`
- **Actions**:
  1. Retrieve target task requirements from `task.md` or `backlog.md`.
  2. Load and review active guardrails from `.agents/project/guardrails.md`.
  3. Search for existing module build guides under `.agents/project/build_guides/`.
  4. If a build guide does not exist, analyze the codebase layout to draft a build guide detailing class structures, interface expectations, and library dependencies.
  5. Prepare a step-by-step task checklist incorporating both functional requirements and guardrail requirements (e.g., unit test generation).

### 2. Environment Manager
- **Agent**: `@mlops`
- **Skill**: `environment-manager`
- **Actions**:
  1. Verify python environment, dependencies, and imports before building.

### 3. Engineering / Implementation Agent
- **Agent**: `@ml` / `@data`
- **Skill**: `model-engineering` / `data-engineering`
- **Actions**:
  1. Implement code changes following the build guide and task checklist.
  2. Adhere strictly to coding style, complexity limits, and architectural constraints.

### 4. Verification & Guardrails Compliance
- **Agent**: `@build-planner` / `@architect`
- **Skill**: `build-planner` / `architecture-manager`
- **Actions**:
  1. Review changes against guardrails (naming conventions, complexity, type-hints).
  2. Execute unit tests (`pytest`) to verify functionality.
  3. If tests fail or guardrails are violated, return to Step 3.

### 5. Log Progress
- **Agent**: `@build-planner`
- **Skill**: `build-planner`
- **Actions**:
  1. Execute `/log-work` workflow.
