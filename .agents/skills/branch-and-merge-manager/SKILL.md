---
name: branch-and-merge-manager
description: Understand project timelines, manage development and feature branches, create branches for requirements, and merge them cleanly with conflict resolution.
risk: medium
source: project
---

# Branch and Merge Manager

## Objective
Coordinate git branch management across the repository. Ensure feature branches are cleanly separated, respect project timelines/milestones, and resolve merge conflicts autonomously or with systematic guidelines.

---

## When to Use
Use this skill when:
- Starting a new feature, bug fix, or experiment.
- Integrating a completed task back into the main branch (`main` or `master`).
- Experiencing merge conflicts when pulling or merging.
- Aligning work branches with project milestones or release timelines.

---

## Responsibilities
- **Timeline Alignment**: Assess current project phase, milestones, and deadlines before branching.
- **Branch Creation**: Create feature or bugfix branches with clean naming conventions:
  - `feature/<name>` for new capabilities.
  - `bugfix/<name>` for bug fixes.
  - `experiment/<name>` for research/training trials.
- **Branch Cleanliness**: Rebase or merge from the main branch periodically to prevent drifting.
- **Conflict Resolution**: Identify and resolve conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) using structural and syntactical understanding of the code.

---

## Workflow

### 1. Pre-Branch Check
Verify current branch state and checkout/update the main branch:
```powershell
git checkout main
git pull origin main
```

### 2. Branch Creation
Generate a descriptive branch name based on current requirements:
```powershell
git checkout -b feature/ecg-transformer-attention
```

### 3. Keep Synced
Regularly pull changes from the main branch:
```powershell
git fetch origin
git merge origin/main
# OR
git rebase origin/main
```

### 4. Conflict Resolution
If a conflict occurs:
1. Locate conflicting files using `git status`.
2. Inspect the conflict markers.
3. Compare both versions and apply logical code blending (preserving both lines if they are complementary, or choosing the latest/most correct implementation).
4. Remove conflict markers, verify code builds and tests pass, then stage and commit:
   ```powershell
   git add <resolved-files>
   git commit -m "Resolve merge conflicts with main"
   ```

### 5. Merging & Cleanup
Once verification succeeds, merge back:
```powershell
git checkout main
git merge feature/ecg-transformer-attention
git branch -d feature/ecg-transformer-attention
```

---

## Success Criteria
- Branches are logically structured and isolated.
- Zero unresolved merge conflicts remain in files.
- Main branch remains stable and always builds successfully.
