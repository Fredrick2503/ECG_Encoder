---
name: docs-and-thesis-manager
description: Automatically run post-prompt documentation updates, trace failures/shortcomings, review historical conversations/transcripts and project artifacts, and keep all documentation (thesis notes, reports, and logs) unified and complete.
risk: medium
source: project
---

# Docs and Thesis Manager

## Objective
Maintain the complete, continuous alignment of thesis notes, reports, and logs across the workspace. Proactively verify documentation after each prompt/interaction, extract insights from historical conversations, record barrier details and failures, and document shortcomings/mitigations.

---

## When to Use
Use this skill:
- After every prompt, task, or experiment trial to sync relevant docs.
- When an experiment or process fails, to log the failure and capture limitations/shortcomings.
- To audit the codebase, workspace artifacts, or past conversation transcripts for missing documentation.
- To generate or compile master summaries (e.g., `thesis_notes.md`, `shortcomings.md`, `future_work.md`).

---

## Responsibilities
- **Continuous Documentation**: Update `docs/` and project logs on every cycle.
- **Shortcomings & Failure Logging**: Capture and document trial failures, code shortcomings, and algorithmic bottlenecks in a dedicated file (e.g., `shortcomings.md`).
- **Transcript Parsing**: Scan past conversation transcripts in `<appDataDir>\brain\<conversation-id>\.system_generated\logs/transcript.jsonl` to extract requirements, design decisions, and context.
- **Completeness Auditing**: Cross-reference the implementation with documentation to identify gaps and update missing reports.

---

## Workflow

### 1. Identify Missing Docs / Context
Scan the workspace and review logs for gaps. Review the current conversation's transcript if context was lost:
- Check `transcript.jsonl` in `<appDataDir>\brain\<conversation-id>\.system_generated\logs/` for details.

### 2. Log Failures & Shortcomings
When a task fails or runs into a barrier, immediately log it:
1. Identify the root cause (e.g., overfitting, gradient collapse, library incompatibilities).
2. Append to `.agents/project/research/mastermind/shortcomings.md`.
3. Highlight the mitigation strategy.

### 3. Update Thesis & Research Artifacts
Update academic or technical documentation files:
- `docs/<module>/thesis_notes.md`
- `docs/<module>/experiment_log.md`
- `future_work.md`

### 4. Post-Prompt Sync Verification
Run a verification pass to confirm that any new modules, functions, or experiments have corresponding documentation updates.

---

## Success Criteria
- No undocumented codebase changes or experiment runs exist.
- Failures and limitations are clearly logged under `shortcomings.md` with action items.
- All documents, notebooks, and summaries are updated automatically.
