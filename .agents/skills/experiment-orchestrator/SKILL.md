---
name: experiment-orchestrator
description: >
  Core skill of the MasterMind agent. Manages the full autonomous experiment
  lifecycle: reads project state, dispatches specialized agents, monitors
  outcomes, triggers self-correction loops, updates all documentation, and
  applies termination conditions. Use whenever starting or resuming the
  /mastermind-loop workflow.
risk: critical
source: project
---

# Experiment Orchestrator

## Objective

Act as the command brain of the autonomous experiment intelligence system.
Coordinate all specialized agents across the ECG pipeline — from data audit to
thesis update — without requiring user intervention between trials.

---

## When to Use

Use this skill whenever:

- Starting the `/mastermind-loop` workflow
- Resuming an interrupted experiment loop
- Assigning a new training trial to agents
- Checking whether target metrics have been reached
- Deciding whether to trigger a barrier resolution cycle
- Determining the next strategy to attempt

---

## Agent Dispatch Table

| Trigger | Agent to Dispatch |
|---|---|
| Loop start | `@data-intelligence` → audit data pipeline |
| Trial ready | `@model-experiment` → run training trial |
| Trial complete | `@experiment-logger` → record full trial |
| Metrics below target | `@barrier-analyst` → detect and resolve barrier |
| Barrier fix ready | `@adaptive-trainer` → apply fix + mutate strategy |
| Trial logged | `@notebook-sync` → update notebooks |
| Trial logged | `@thesis-doc` → update documentation |
| Target reached | sync workspace + promote model |

---

## Responsibilities

- Read `project_state.md`, `backlog.md`, and `research/` logs before each loop
- Select the next experiment configuration from the strategy queue
- Dispatch agents in the correct order
- Collect structured results from each agent
- Evaluate termination conditions
- Update `mastermind_state.md` after each iteration
- Escalate unresolvable barriers to the user

---

## Loop Structure

```
LOOP:
  1. Read project state + last experiment outcomes
  2. Dispatch @data-intelligence → get data_audit.md
  3. Select next strategy from strategy_queue
  4. Dispatch @model-experiment(strategy) → get trial_result
  5. Dispatch @experiment-logger(trial_result) → get trial_id
  6. Evaluate metrics:
     IF metrics < target:
       Dispatch @barrier-analyst → get barrier_report
       Dispatch @adaptive-trainer(barrier_report) → get fixed_strategy
       Enqueue fixed_strategy → LOOP
     ELSE IF metrics >= target:
       Dispatch @notebook-sync + @thesis-doc
       Promote model → DONE
  7. IF budget exhausted (max_trials reached):
       Dispatch @thesis-doc with summary of all attempts
       Report to user → PAUSE
```

---

## State File

Maintain `.agents/project/mastermind_state.md` with:

- Current loop iteration number
- Strategy queue (tried + pending)
- Best metric achieved and which trial
- Current blockers
- Total trials run
- Total training time
- Target metric and current gap

---

## Termination Conditions

Halt and report when ANY of the following are true:

| Condition | Action |
|---|---|
| Target ROC-AUC reached (default ≥ 0.92) | Promote model, sync, report |
| Max trials exhausted (default 20) | Report best model found, ask user for guidance |
| Unresolvable barrier detected (3 failed fixes) | Escalate to user with full barrier report |
| User explicitly halts the loop | Clean up + save state |

---

## Outputs

After every iteration:

- Updated `mastermind_state.md`
- Updated `experiment_journal.md`
- Updated notebooks
- Updated thesis notes
- Structured console summary per trial

---

## Rules

- Never skip the data audit step — data changes invalidate experiments.
- Never overwrite a better model with a worse one.
- Always log failed trials with full details — failures are research data.
- Never delete the strategy queue — preserve the full search history.
- Escalate to user only when genuinely stuck after multiple fix attempts.

---

## Success Criteria

The skill is successful when:

- Every trial is dispatched, tracked, and documented.
- The strategy queue is exhausted or the target is reached.
- All notebooks and docs are up to date.
- A clear best model is identified.
- The user receives a complete experiment summary.
