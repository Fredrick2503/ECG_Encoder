---
name: representation-reviewer
description: Review representation learning modules to ensure encoder implementations follow the project architecture, expose consistent interfaces, and produce valid representations. Use whenever an encoder or representation module is added, modified, or refactored.
risk: medium
source: project
---

# Representation Reviewer

## Objective

Review representation learning modules to ensure they are consistent,
modular, and aligned with the ECG Foundation Representation System.

This skill validates encoder implementations, interfaces, and integration.

It does **not** evaluate model performance or clinical correctness.

---

# When to Use

Use this skill whenever:

- a new encoder is implemented
- an encoder is modified
- a fusion module changes
- an embedding interface changes
- representation-learning pipelines are refactored
- before merging major encoder changes

---

# Responsibilities

Review:

- Encoder interfaces
- Input/output consistency
- Embedding dimensions
- Module responsibilities
- Integration with downstream pipelines
- Compliance with project architecture

---

# Review Checklist

Verify that:

- Each encoder has a single responsibility.
- Input and output interfaces follow project conventions.
- Embedding outputs are clearly defined.
- Modules remain independent and loosely coupled.
- Shared interfaces are preserved.
- Downstream modules are not broken by the changes.

---

# Workflow

For every review:

1. Identify the affected representation modules.
2. Review interfaces and responsibilities.
3. Check consistency with the project architecture.
4. Record any findings or recommendations.
5. Approve or request changes.

---

# Files to Review

Review modules such as:

- Temporal Encoder
- Morphology Encoder
- Biomarker Encoder
- Fusion Engine
- Shared encoder interfaces

Do not modify implementation unless explicitly requested.

---

# Rules

- Focus on representation-learning modules only.
- Do not review unrelated components.
- Do not change project architecture.
- Report inconsistencies with clear recommendations.
- Preserve encoder independence.

---

# Success Criteria

The skill is successful when:

- Representation modules follow the project architecture.
- Encoder interfaces are consistent.
- Integration points are correct.
- No unnecessary coupling exists.
- Review findings are clearly documented.