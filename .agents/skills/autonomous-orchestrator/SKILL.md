---
name: autonomous-orchestrator
description: Coordinate the entire agent team to autonomously loop through implementation tasks, analyze performance drawbacks, search online for solutions, and develop features in parallel git branches without user intervention.
risk: critical
source: project
---

# Autonomous Orchestrator

## Objective
Establish a fully autonomous, self-correcting development and research loop. This skill enables the agent team to manage parallel feature branches, diagnose engineering and model constraints, query the web/literature for cutting-edge solutions, and implement fixes incrementally without interrupting the user.

---

## Git Parallel Branching Rules
When developing features or running experiments in parallel:
1. **Branch Creation**: For any new task or feature, checkout a dedicated feature branch using naming convention `feature/<module_name>-<short_description>` (e.g. `feature/morphology_encoder-initial`).
2. **Isolation**: Never commit code to `main` directly. Keep all development, training runs, and evaluations on the feature branch.
3. **Synchronization**: Periodically merge or rebase `main` into active feature branches to stay up to date.
4. **Integration**: Upon successful verification (all unit tests passing, model evaluations satisfying acceptance criteria):
   - Switch back to `main`.
   - Merge the feature branch.
   - Delete the remote/local feature branch.
   - Sync the workspace using the `/sync-workspace` workflow.

---

## Drawback Detection & Online Literature Search
When an implementation or model run fails to meet performance/metric targets (e.g., overfitting, high validation loss, bad gradient attribution):
1. **Define the Bottleneck**: Identify the exact issue (e.g., "vanishing gradients in deep ECG Transformer layers" or "imbalanced multilabel sampling skew").
2. **Search Literature**: Use the `search_web`, `literature_search_arxiv`, or `pubmed-database` tools to find how similar problems are solved in modern literature.
   - Focus on queries combining "ECG representation learning", "self-supervised learning", and the identified bottleneck.
3. **Formulate Solutions**: Translate academic findings into local code upgrades (e.g., adding pre-layer normalization, tuning masking schedules, trying dynamic loss weighting).
4. **Implement & Benchmark**: Build the changes on the active git branch, run the training pipeline, and compare the updated runs using MLflow.

---

## Continuous Autonomous Loop Workflow
1. **Roadmap Review**: Consult `project_state.md` and `roadmap.md` to identify pending features or target metrics.
2. **Branch & Checkout**: Execute `git checkout -b feature/<task_name>`.
3. **Implement**: Direct the Model Engineer (@ml) or Data Engineer (@data) to write or upgrade the code.
4. **Validate & Review**: 
   - System Architect (@architect) verifies SOLID boundaries.
   - Run existing unit tests (`pytest`).
5. **Evaluate & Tune**:
   - MLOps runs training; checks MLflow logs.
   - Research Scientist (@research) reviews outcomes against baselines.
6. **Self-Correct Loop**: If drawbacks or metric drops are found, trigger the **Drawback Detection & Online Literature Search** loop, apply fixes, and retrain.
7. **Merge & Document**: Merge to `main`, update project memory, and update feature-specific logs and thesis notes.
