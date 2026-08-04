---
name: data-engineering
description: Manage the complete ECG data pipeline. Use whenever working with datasets, data loading, preprocessing, feature preparation, validation, or data pipeline development.
risk: high
source: project
---

# Data Engineering

## Objective

Build and maintain reliable, reusable, and reproducible ECG data pipelines.

This skill is responsible for acquiring, validating, loading, preprocessing,
and preparing ECG data for downstream model engineering.

It does **not** build or train models.

---

# When to Use

Use this skill whenever:

- adding a new dataset
- downloading datasets
- validating datasets
- loading ECG records
- preprocessing ECG signals
- extracting metadata
- creating dataset splits
- building data pipelines
- debugging data issues

---

# Responsibilities

Owns:

- Dataset acquisition
- Dataset validation
- Dataset loading
- Metadata management
- Label processing
- ECGRecord creation
- Signal preprocessing
- Dataset splitting
- Data augmentation
- Data caching
- Data pipeline optimization

---

# Workflow

For every data-related task:

1. Identify the dataset(s) involved.
2. Validate dataset integrity.
3. Load and standardize the data.
4. Apply preprocessing pipeline.
5. Validate outputs.
6. Produce data ready for Model Engineering.

---

# Inputs

Examples:

- PTB-XL
- MIT-BIH
- CPSC
- Custom ECG datasets

---

# Outputs

Produce standardized data structures such as:

- ECGRecord
- Processed ECG signals
- Labels
- Metadata
- Train / Validation / Test datasets

---

# Rules

- Keep dataset-specific logic isolated.
- Ensure reproducible preprocessing.
- Never modify raw datasets.
- Validate inputs before processing.
- Produce standardized outputs for downstream modules.
- Keep preprocessing independent of model architecture.

---

# Dependencies

Reads:

- Project Memory
- Architecture Manager

Provides outputs to:

- Model Engineering

---

# Success Criteria

The skill is successful when:

- Datasets are validated.
- ECG records load correctly.
- Preprocessing is reproducible.
- Data is standardized.
- Pipelines are reusable.
- Outputs are ready for model training.