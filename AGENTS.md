# Agent instructions (KC761 rewrite)

Binding rules for every change in this repository. They restate the frozen
decisions in `docs/plan.md`; if anything here conflicts with the plan, the plan
wins and this file must be fixed in the same change.

## Hard rules

1. **English only.** All repository artifacts - docs, docstrings, comments, CLI
   help, error messages - are written in English.
2. **No packaging.** No `pyproject.toml`, `setup.py`, `setup.cfg` or
   installable distribution. Tool configuration lives in `ruff.toml` and
   `pytest.ini`. Supported entry points: `python kc761.py ...` and
   `python -m kc761 ...`.
3. **Python >= 3.12.** Use `from __future__ import annotations` and modern
   typing syntax.
4. **Frozen decisions are frozen.** `docs/plan.md` Section 1 is normative.
   Undecided points are listed in Appendix A; raise them as questions, never
   resolve them by assumption or silent code change.
5. **No baseline/golden correctness.** Never define correctness by comparing
   against legacy outputs, golden files or regression references. Correctness
   comes from single-source formulas, sympy-generated code and runtime
   certificates.
6. **Tests are auxiliary.** pytest/hypothesis checks codify invariants but do
   not define correctness. CI runs ruff + tests.
7. **Workstream discipline.** W0 establishes contracts and skeleton only.
   Stubs must keep raising `NotImplementedError` until their owning workstream
   implements them. Do not implement logic assigned to a later workstream.
8. **Strict mode.** `--strict` or `KC761_STRICT=1` enables all certificate
   suites (KKT, conservation, PSD, column sums, efficiency bounds,
   monotonicity, positivity, finiteness). Basic schema/shape/finiteness
   validation is always on. Certificates fail fast and report the formula ID.
9. **Formula IDs.** Every formula gets an ID in `docs/derivations.md`; code
   references it. IDs are append-only; derivations may gain detail but must not
   contradict frozen decisions.
10. **Products.** Read/write only through `kc761/schema/io.py`: atomic write,
    reopen validation, full provenance, refuse-overwrite unless `--force`.
11. **Single source.** No duplicated formula, constant, axis convention or
    object-name literal across modules. Cross-language copies do not exist
    anymore (pure uproot, no C++ IO).
12. **No silent failures.** No bare `except`, no silent clamping or NaN
    substitution. Numerically necessary clamps must be explicit, warned about
    and recorded in product metadata.

## File ownership

| Path | Owner | Notes |
|------|-------|-------|
| `kc761/core/model.py`, `binning.py`, `kernel.py`, `response.py`, `projection.py` | W1 | formula IDs F-MODEL/F-BIN/F-KERN/F-RESP/F-PROJ |
| `kc761/core/solver.py`, `covariance.py`, `uncertainty.py` | W1 | F-SOLVE/F-COV/F-UNC |
| `kc761/schema/axes.py`, `products.py`, `io.py` | W2 | product contract; changes go through the contract-change process |
| `kc761/schema/_uproot.py` | W2 | verified spike helpers; keep the head comment in sync with `docs/formats.md` |
| `kc761/calib/` | W3 | F-CAL |
| `kc761/unfold/` | W4 | uses F-SOLVE/F-UNC |
| `kc761/sim/` | W5 | F-SIM |
| `kc761/cli/`, `kc761.py`, `kc761/__main__.py` | W6 | CLI surface only; no numerics |
| `kc761/errors.py`, `kc761/runtime.py` | W0 (stable) | change requires a plan update |
| `kc761/plotting/` | W3/W4 | shared style, no duplicated helpers |
| `tests/`, `tests/fixtures/synthetic.py` | shared | fixture changes must stay deterministic |
| `docs/plan.md` | user-approved | edit only to record a new decision |
| `docs/architecture.md`, `docs/formats.md`, `docs/derivations.md` | owning workstream | update in the same change as the code |
| legacy `app/`, `kc761calib/`, `kc761sim/`, `kc761unfold/`, `kc761util/`, C++ sources | W7 deletion | do not modify before W7 |

## Contract-change process

1. Raise the point as a question (open points are tracked in `docs/plan.md`
   Appendix A).
2. Record the decision in `docs/plan.md` Section 1 (or Appendix A resolution).
3. Update the affected formula registry, ownership table and product schema.
4. Implement it in the owning workstream; stubs in earlier workstreams must not
   be bypassed.

## Verification commands

```bash
ruff check .
pytest -q -m "not g4 and not root"
python kc761.py --help
python -m kc761 --help
```
