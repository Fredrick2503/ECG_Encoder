---
name: project-planner
description: Maintain the project roadmap, backlog, milestones, and task priorities. Use whenever planning work, creating milestones, reprioritizing tasks, or determining the next implementation step.
risk: medium
source: project
---

# Project Planner

## Objective

Maintain the project roadmap and implementation plan.

This skill is responsible for organizing work into milestones, prioritizing
tasks, tracking dependencies, and recommending the next logical task.

It does **not** maintain project history or implementation details.

---

# When to Use

Use this skill whenever:

- planning a new feature
- creating or updating milestones
- prioritizing work
- adding new tasks
- changing priorities
- determining the next implementation step
- reorganizing the backlog

---

# Responsibilities

Maintain:

- Project roadmap
- Milestones
- Backlog
- Task priorities
- Task dependencies
- Implementation order
- Next recommended task

---

# Workflow

For every planning request:

1. Review the current project state.
2. Identify completed and pending work.
3. Update the roadmap or backlog.
4. Resolve task dependencies.
5. Recommend the next highest-priority task.

---

# Files to Maintain

Update these files as needed:

.agents/project/

roadmap.md

backlog.md

milestones.md

Only modify files affected by the planning activity.

---

# Rules

- Respect task dependencies.
- Prioritize foundational work before dependent work.
- Keep milestones achievable and clearly defined.
- Do not modify project history.
- Do not mark work complete.
- Coordinate with Project Memory for current project status.

---

# Success Criteria

The skill is successful when:

- The roadmap reflects current priorities.
- Milestones are clearly defined.
- Dependencies are valid.
- The backlog is organized.
- The next recommended task is clear.