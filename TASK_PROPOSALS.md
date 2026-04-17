# Codebase Task Proposals

This document lists four concrete, scoped tasks discovered during a quick codebase audit.

## 1) Typo fix task

**Title:** Fix README installation typo (`.mltb` → `.mltbx`) and wording errors.

**Why:** The installation section references `StiffMa1.6.mltb` in the visible link text, while the actual package extension is `.mltbx`. The same sentence also has a wording typo (`double clic`).

**Acceptance criteria:**
- Update visible extension to `.mltbx`.
- Fix `double clic` to `double click`.
- Keep the hyperlink target unchanged unless release naming changed.

## 2) Bug fix task

**Title:** Validate `CreateMesh2` element counts as strictly positive.

**Why:** `CreateMesh2` only rejects negative counts (`< 0`) but allows zeros. Later, the code divides by `nelx`, `nely`, `nelz`, which can lead to division-by-zero (`selX = sX/nelx`, etc.).

**Acceptance criteria:**
- Reject zero values in input validation.
- Improve error messages to mention all three inputs correctly (`nelx`, `nely`, `nelz`).
- Add regression tests for zero and negative inputs.

## 3) Documentation/comment discrepancy task

**Title:** Update `AssemblyStiffMa` function header comment to match current API.

**Why:** The function signature is `AssemblyStiffMa(iK, jK, Ke, sets)` but the help text still advertises `ASSEMBLYSTIFFMA(iK,jK,Ke,dTE,dTN)`.

**Acceptance criteria:**
- Align function help text with the real `sets`-based interface.
- Correct minor wording/spelling issues in the same block (`colomn`, `dtermines`, etc.).
- Ensure examples in help text use current call style.

## 4) Test improvement task

**Title:** Convert `tests/Verification/verify_scalar.m` from visual/manual script into assertion-based test.

**Why:** The current verification script mostly generates figures and prints norm differences; it does not enforce pass/fail thresholds.

**Acceptance criteria:**
- Implement a function-based test (or `matlab.unittest` class) with explicit tolerances for:
  - ANSYS vs MATLAB CPU result discrepancy.
  - MATLAB CPU vs GPU discrepancy.
- Gate figure generation behind an optional debug flag so CI can run headless.
- Fail test automatically when norms exceed thresholds.
