# Build & Development Guardrails

This document defines the strict guardrails for the ECG Foundation Representation System. All development agents must adhere to these guidelines to ensure consistency, prevent regressions, and avoid losing track of build states.

---

## 1. Architectural Guardrails
- **Modular Boundaries**: Maintain strict separation between dataset pipeline, model encoder architectures, MLOps, and evaluation.
- **Dependency Direction**: Higher-level scripts (e.g., training, pipelines) may import lower-level modules (e.g., encoders, preprocessing), but lower-level modules must never import high-level scripts.
- **Clean Interfaces**: All encoder models must subclass a common base class (if defined) or expose a standardized PyTorch module interface with predictable inputs/outputs.

## 2. Code Quality & Complexity Guardrails
- **No Placeholders**: Never commit TODOs, stubs, or placeholders in production files.
- **Function/Class Length**: Keep functions under 50 lines and classes under 300 lines unless mathematically necessary.
- **Type Hinting**: All new Python code must use type hints for arguments and return types.
- **Comments & Docstrings**: Document public APIs, classes, and complex algorithms using Google-style docstrings. Maintain existing unrelated comments.

## 3. Testing Guardrails
- **Unit Tests**: Every new module or encoder upgrade must be accompanied by or pass corresponding unit tests under `tests/`.
- **Pre-Commit Verification**: Run pytest before completing tasks to verify no regressions are introduced.

## 4. Logging & Documentation Guardrails
- **Every Change Logged**: Every implementation detail, bug fix, and feature addition must be logged in `implementation_log.md` under the respective module research directory.
- **Academic Rigor**: Keep `research_log.md` updated with exact experiment configurations, metrics, and observations.
