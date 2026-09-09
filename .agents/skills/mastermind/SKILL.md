---
name: mastermind
description: Dynamic agent orchestration skill. Decompose complex user goals, monitor task states, and dynamically coordinate specialized subagents (e.g., data engineering, model experiment, evaluation, thesis writers) to complete tasks autonomously.
risk: high
source: project
---

# MasterMind Orchestrator

## Objective
Enable high-level autonomous agent orchestration. The MasterMind agent acts as the central processor that interprets the main goal, plans execution paths, checks conditions, invokes specialized subagents, handles errors, and aggregates results.

---

## When to Use
Use this skill when:
- Orchestrating a multi-step workflow like `/mastermind-loop`.
- You need to delegate tasks to subagents to parallelize work or isolate focus.
- Automatically analyzing experiment/task failures to trigger corrective actions (like hyperparameter mutations or code bug fixes).
- Maintaining global project state across trials.

---

## Orchestration Logic

```mermaid
flowchart TD
    Start[🧠 MasterMind: Analyze Goal] --> Decide[🤔 Decide Specialized Subagent]
    Decide --> |Data Processing| DataAgent[@data-intelligence / data-engineering]
    Decide --> |Training/Experiment| TrainAgent[@model-experiment / model-engineering]
    Decide --> |Evaluation/Metrics| EvalAgent[@evaluator / evaluation-validation]
    Decide --> |Documentation/Thesis| DocAgent[@thesis-doc / docs-and-thesis-manager]
    
    DataAgent --> Callback[📥 Return Result to MasterMind]
    TrainAgent --> Callback
    EvalAgent --> Callback
    DocAgent --> Callback
    
    Callback --> Verify{🏁 Goal Accomplished?}
    Verify --> |No: Self-Correction| Decide
    Verify --> |Yes| End[🏆 Complete & Report to User]
```

---

## Responsibilities
- **Subagent Selection**: Match task demands to specialized agent profiles.
- **Context Passing**: Pass only the necessary files, configs, and constraints to the subagent to prevent context bloat.
- **Monitoring & Heartbeats**: Set timers and monitor async tasks using schedule tools.
- **Dynamic Fallbacks**: If a subagent fails or returns poor metrics, detect the barrier type and mutate the execution strategy.

---

## Success Criteria
- Subagents are invoked with precise instructions and minimum context overlap.
- High-level loops converge to the target objective autonomously.
- Tasks are completed efficiently without redundant steps.
