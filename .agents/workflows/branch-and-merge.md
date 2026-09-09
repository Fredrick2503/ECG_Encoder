---
description: Manage Git branches, feature branching, merging, and conflict resolution based on project timelines.
---

# /branch-and-merge

This workflow manages the git branch lifecycle, creating branches for specific features or requirements, and merging them back cleanly with conflict resolution.

---

## Steps

### Step 1 — Check Current Timelines & Objectives
Analyze project state or backlog files to determine the current milestone, targets, and if a branch is required.
If a branch already exists for the task, switch to it. Otherwise, proceed to Step 2.

### Step 2 — Create Feature Branch
Create a descriptive branch:
```powershell
git checkout -b feature/<feature-name>
# or bugfix/<bug-name> or experiment/<experiment-name>
```

### Step 3 — Synchronize with Main Branch
Periodically pull from `main` to avoid large divergence:
```powershell
git checkout feature/<feature-name>
git fetch origin
git merge origin/main
```

### Step 4 — Resolve Merge Conflicts
If conflicts occur:
1. Identify all conflicted files: `git status`.
2. Open each conflicted file and locate conflict blocks.
3. Edit the code to blend the updates cleanly.
4. Verify code correctness (compile/run tests).
5. Stage and commit:
   ```powershell
   git add <resolved-files>
   git commit -m "Resolve conflicts with main"
   ```

### Step 5 — Final Merge and Cleanup
Once work is complete and verified:
```powershell
git checkout main
git merge feature/<feature-name>
git branch -d feature/<feature-name>
```
