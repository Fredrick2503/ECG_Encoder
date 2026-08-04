---
name: model-engineering
description: Design, implement, train, validate, and maintain AI models for the ECG Foundation Representation System. Use whenever building, modifying, training, validating, or integrating representation learning models.
risk: high
source: project
---

# Model Engineering

## Objective

Design, implement, train, and maintain the machine learning models used by the
ECG Foundation Representation System.

This skill owns the complete model lifecycle, from architecture implementation
to inference-ready models.

It does **not** manage datasets, experiments, project planning, or documentation.

---

# When to Use

Use this skill whenever:

- implementing a new model
- modifying an existing model
- building an encoder
- implementing a classifier
- implementing a fusion model
- configuring model parameters
- training a model
- validating a model
- saving checkpoints
- loading pretrained models
- building an inference model

---

# Responsibilities

Owns:

- Model architecture
- Encoder implementation
- Representation learning
- Model configuration
- Training pipeline
- Validation
- Checkpoint management
- Model loading
- Inference pipeline

---

# Scope

Examples of components managed by this skill include:

- Temporal Encoder
- Morphology Encoder
- Biomarker Encoder
- Fusion Module
- Classification Head

This skill ensures models follow the approved architecture and integrate
correctly with the rest of the system.

---

# Workflow

For every implementation:

1. Review the approved architecture.
2. Implement or modify the model.
3. Configure model parameters.
4. Train or validate the model.
5. Save checkpoints when appropriate.
6. Verify the model integrates with the overall pipeline.

---

# Rules

- Follow the Architecture Manager's design.
- Do not implement dataset loading or preprocessing.
- Keep model implementations modular and reusable.
- Separate model definition from training logic where possible.
- Preserve backward compatibility unless an architectural change is approved.

---

# Dependencies

Coordinate with:

- Architecture Manager
- Data Engineering
- MLflow Manager
- Evaluation & Validation
- Project Memory

---

# Success Criteria

The skill is successful when:

- The model is correctly implemented.
- Training completes successfully.
- Validation passes.
- Checkpoints are saved.
- The model integrates into the project pipeline.
- The implementation follows the approved architecture.