---
name: clinical-review
description: Review the medical and clinical correctness of the ECG Foundation Representation System. Use whenever implementing, modifying, evaluating, or documenting medically relevant functionality to ensure clinical validity and consistency.
risk: high
source: project
---

# Clinical Review

## Objective

Ensure that the system remains medically accurate and clinically meaningful.

This skill reviews implementations from a clinical perspective and validates
that medical terminology, ECG concepts, biomarkers, disease labels, and
clinical assumptions are correct.

It does **not** implement models or modify engineering decisions unless a
medical issue is identified.

---

# When to Use

Use this skill whenever:

- implementing ECG-related functionality
- adding new biomarkers
- introducing new disease labels
- modifying preprocessing that may affect clinical interpretation
- updating explainability
- reviewing model outputs
- writing medical documentation

---

# Responsibilities

Review:

- ECG terminology
- Disease definitions
- Biomarker correctness
- Clinical assumptions
- Label mappings
- Medical documentation
- Explainability outputs
- Clinical consistency across the system

---

# Workflow

For every review:

1. Identify the affected medical components.
2. Verify medical terminology and concepts.
3. Check that biomarkers and labels are clinically correct.
4. Identify inconsistencies or potential clinical risks.
5. Document findings and recommendations.

---

# Rules

- Never invent medical facts.
- Flag uncertain clinical assumptions for review.
- Preserve accepted medical terminology.
- Distinguish engineering issues from clinical issues.
- Base recommendations on established clinical practice and referenced literature whenever possible.

---

# Success Criteria

The skill is successful when:

- Medical terminology is correct.
- Disease labels are clinically valid.
- Biomarkers are correctly interpreted.
- Explainability remains clinically meaningful.
- Any clinical concerns are clearly documented before implementation proceeds.