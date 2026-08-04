---
name: environment-manager
description: Manage the project's development environment, dependencies, package managers, third-party libraries, system requirements, and reproducible development setup. Use whenever installing, updating, configuring, or validating the project environment.
risk: medium
source: project
---

# Environment Manager

## Objective

Maintain a consistent, reproducible development environment for the project.

This skill is responsible for managing Python environments, project
dependencies, system requirements, third-party libraries, and development
tooling.

It does **not** implement application features.

---

# When to Use

Use this skill whenever:

- setting up the project
- creating a new development environment
- installing dependencies
- updating packages
- adding a third-party library
- resolving dependency conflicts
- validating the environment
- exporting dependencies
- configuring development tools

---

# Responsibilities

Manage:

- Python environment
- Virtual environments
- Project dependencies
- Third-party libraries
- Package versions
- CUDA compatibility
- PyTorch compatibility
- Development tools
- Environment validation

---

# Workflow

For every environment request:

1. Inspect the current environment.
2. Validate installed dependencies.
3. Resolve missing or conflicting packages.
4. Install or update required libraries.
5. Verify compatibility.
6. Update dependency files if needed.

---

# Files to Maintain

Update only when required:

requirements.txt

pyproject.toml

environment.yml

README.md

.devcontainer/

Dockerfile

---

# Rules

- Prefer stable package versions.
- Avoid unnecessary dependencies.
- Verify compatibility before upgrades.
- Pin versions for reproducibility.
- Never remove dependencies without checking impact.
- Keep dependency files synchronized.

---

# Common Responsibilities

Python

- venv
- conda
- uv

Package Management

- pip
- pip-tools
- Poetry (if adopted)

Deep Learning

- PyTorch
- CUDA
- TorchVision
- TorchMetrics

Data Science

- NumPy
- Pandas
- SciPy
- scikit-learn

Visualization

- Matplotlib
- Seaborn
- Plotly

Experiment Tracking

- MLflow

Medical Libraries

- WFDB
- NeuroKit (if adopted)

Development Tools

- Ruff
- Black
- isort
- pytest
- pre-commit
- mypy

---

# Success Criteria

The skill is successful when:

- The environment is reproducible.
- All required dependencies are installed.
- Package versions are compatible.
- Dependency files are current.
- The project can be set up consistently on a new machine.