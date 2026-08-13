---
description: Run tests for the named feature and report pass/fail mapped to the spec's acceptance criteria.
argument-hint: <feature-id> — e.g. FR-AUTH, FR-TKT
---

Verify the implementation of **$ARGUMENTS** against its spec.

## Steps

1. **Locate the spec file**: glob `specs/*$ARGUMENTS*.md`. If no spec exists, stop and tell the user to run `/spec $ARGUMENTS` first.

2. **Read the spec file** and extract all acceptance criteria (AC-01, AC-02, …). These are the ground truth for pass/fail.

3. **Locate the test file**: look for `tests/test_<feature>.py` where `<feature>` is the lowercase suffix of the feature ID (e.g. `FR-AUTH` → `tests/test_auth.py`). If it does not exist, report it as missing and skip to Step 6.

4. **Run the tests**:
   ```
   pytest tests/test_<feature>.py -v
   ```
   Capture the full output.

5. **Map results to acceptance criteria**: for each AC-NN in the spec, identify which test(s) cover it and whether they passed or failed. If a criterion has no corresponding test, mark it **UNTESTED**.

6. **Report** in this format:

   ```
   ## Verification Report — $ARGUMENTS

   Spec: specs/<file>.md
   Test file: tests/test_<feature>.py

   | AC   | BRD Ref        | Description                  | Test(s)                        | Result      |
   |------|----------------|------------------------------|--------------------------------|-------------|
   | AC-01 | FR-...-01      | <criterion summary>          | test_name_here                 | PASS        |
   | AC-02 | FR-...-02      | <criterion summary>          | test_other_here                | FAIL        |
   | AC-03 | FR-...-03      | <criterion summary>          | —                              | UNTESTED    |

   ### Failures
   <!-- For each FAIL row, paste the relevant pytest output and explain what the mismatch is -->

   ### Untested criteria
   <!-- For each UNTESTED row, describe what test case would cover it -->

   ### Summary
   N passed / N failed / N untested out of N acceptance criteria
   ```

## Constraints

- Do **not** modify test files or implementation code during this command.
- Do **not** mark a criterion PASS unless a test explicitly exercises the stated outcome.
- If pytest cannot run (import errors, missing DB, etc.), report the error and list all criteria as UNTESTED rather than guessing.
