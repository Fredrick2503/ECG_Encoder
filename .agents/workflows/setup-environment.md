---
description: Setup or validate the project development environment.
---

When the user invokes `/setup-environment`:

Execution Sequence

1. Environment Manager
   - Detect the current development environment.
   - Create or validate the Python environment.
   - Install required dependencies.
   - Validate CUDA, GPU, and framework compatibility.
   - Configure development tools.

2. Project Memory
   - Record environment setup or updates.

Report the environment status and any required user actions.