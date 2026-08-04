#!/usr/bin/env bash

set -e

echo "Creating Antigravity project structure..."

mkdir -p .agents/{skills,workflows,project}

# Skills
mkdir -p .agents/skills/{project-memory,project-planner,architecture-manager,data-engineering,model-engineering,mlflow-manager,continuous-training,evaluation-validation,representation-reviewer,clinical-review,research-knowledge-manager}

# Create SKILL.md for each skill
for skill in \
project-memory \
project-planner \
architecture-manager \
data-engineering \
model-engineering \
mlflow-manager \
continuous-training \
evaluation-validation \
representation-reviewer \
clinical-review \
research-knowledge-manager
do
    touch ".agents/skills/${skill}/SKILL.md"
done

# Core project files
touch .agents/agents.md

# Workflows
touch .agents/workflows/continue-project.md
touch .agents/workflows/implement-module.md
touch .agents/workflows/train-model.md
touch .agents/workflows/evaluate-model.md
touch .agents/workflows/benchmark-models.md
touch .agents/workflows/review-module.md
touch .agents/workflows/thesis-update.md
touch .agents/workflows/release-model.md

# Project memory files
touch .agents/project/project_state.md
touch .agents/project/backlog.md
touch .agents/project/roadmap.md
touch .agents/project/progress.md
touch .agents/project/milestones.md
touch .agents/project/decision_log.md
touch .agents/project/technical_debt.md

mkdir -p .agents/project/research

touch .agents/project/research/research_log.md
touch .agents/project/research/experiment_journal.md
touch .agents/project/research/implementation_log.md
touch .agents/project/research/literature_map.md
touch .agents/project/research/thesis_notes.md
touch .agents/project/research/lessons_learned.md
touch .agents/project/research/future_work.md
touch .agents/project/research/weekly_summary.md

echo ""
echo "✅ Antigravity workspace initialized."