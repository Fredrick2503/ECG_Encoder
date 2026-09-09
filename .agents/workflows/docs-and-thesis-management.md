---
description: Ensure all documentation, thesis notes, shortcomings, and reports are systematically updated after every interaction.
---

# /docs-and-thesis-management

This workflow runs to analyze the results of the recent task, scan past conversation files if necessary, and ensure all thesis notes and shortcomings documents are perfectly aligned.

---

## Steps

### Step 1 — Review Task Outcome
Evaluate if the recent task succeeded, failed, or finished with warnings.
- If it failed: Go to Step 2.
- If it succeeded: Go to Step 3.

### Step 2 — Log Failure / Shortcomings
1. Open `.agents/project/research/mastermind/shortcomings.md` (or relevant module shortcoming files).
2. Append a new entry detailing:
   - Date and Timestamp
   - Component / Task Name
   - Failure Log / Error Trace
   - Structural Shortcoming and theoretical limitation
   - Proposed Fix / Actionable Next Steps
3. Go to Step 3.

### Step 3 — Scan Past Conversation Transcripts
If context or specifications are missing, search the transcript log:
```powershell
# Search for user inputs or key definitions in transcript.jsonl
Get-Content -Path "C:\Users\fredr\.gemini\antigravity-ide\brain\<conversation-id>\.system_generated\logs\transcript.jsonl" | Select-String -Pattern "USER_INPUT"
```

### Step 4 — Update Thesis & Reports
1. Update `docs/thesis_notes.md` or the module-specific research logs.
2. Update the experiment notebooks (if any ML trial was completed) using the `notebook-sync` skill.
3. Update `future_work.md` with recommendations based on current findings.

### Step 5 — Verify Alignment
Ensure all modified files have corresponding documentation entries. Stage and commit the documentation updates cleanly.
