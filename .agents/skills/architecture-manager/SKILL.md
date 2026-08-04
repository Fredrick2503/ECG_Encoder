---
name: architecture-manager
description: Design, maintain, and review the software architecture of the ECG Foundation Representation System. Use whenever introducing new modules, changing interfaces, reviewing architecture, or validating design decisions.
risk: high
source: project
---

# Architecture Manager

## Objective

Maintain a clean, modular, and scalable architecture for the project.

This skill defines module boundaries, interfaces, responsibilities, and design
rules to ensure the system remains maintainable as it evolves.

It does **not** implement features.

---

# When to Use

Use this skill whenever:

- designing a new module
- modifying an existing architecture
- introducing a new interface
- reviewing module responsibilities
- validating architectural changes
- reviewing project structure

---

# Responsibilities

Maintain:

- High-Level Design (HLD)
- Low-Level Design (LLD)
- Module responsibilities
- Public interfaces
- Dependency relationships
- Design principles
- Project structure

---

# Workflow

For every architectural request:

1. Understand the requested change.
2. Identify affected modules.
3. Review dependencies.
4. Validate against architecture principles.
5. Update architecture documentation if required.

---

# Architecture Principles

Always follow:

- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
- Loose coupling
- High cohesion
- Layered architecture
- Clear module ownership

---

# Files to Maintain

Update these files as needed:

.agents/project/

architecture.md

hld.md

lld.md

Only update the files affected by the current architectural change.

---

# Rules

- Never mix responsibilities across modules.
- Keep interfaces stable whenever possible.
- Prefer composition over inheritance.
- Avoid circular dependencies.
- Review downstream impact before approving changes.
- Coordinate with Project Memory to understand the current project state.

---

# Success Criteria

The skill is successful when:

- Module responsibilities are clearly defined.
- Dependencies remain clean.
- Architecture documents are current.
- New modules integrate without breaking existing boundaries.
- The architecture remains modular and extensible.