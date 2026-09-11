# Agent instructions (KC761 toolkit)

Binding rules for every change in this repository. They restate the frozen
decisions in `docs/plan.md`; if anything here conflicts with the plan, the plan
wins and this file must be fixed in the same change.

## Hard rules

1. **American English only.** All repository artifacts - docs, docstrings,
   comments, CLI help, error messages, identifiers - are written in American
   English (`center`, `color`, `behavior`, `normalize`/`normalization`,
   `labeled`, `modeled`, `defense`, `neighbor`, `realized`, `toward`), per
   D-182. Literals that belong to another project's interface stay verbatim;
   the only ones here are the Geant4 UI commands `/gps/pos/centre` and
   `/vis/geometry/set/colour`, each marked as external where it appears.
2. **No packaging.** No `pyproject.toml`, `setup.py`, `setup.cfg` or
   installable distribution. Tool configuration lives in `ruff.toml` and
   `pytest.ini`. Supported entry points: `python kc761tool.py ...` and
   `python -m kc761tool ...` (D-7, renamed by D-183).
3. **Python >= 3.12.** Use `from __future__ import annotations` and modern
   typing syntax.
4. **Frozen decisions are frozen.** `docs/plan.md` Section 1 is normative.
   Existing contract points are recorded in Appendix A; a new one must be
   raised as a question, never resolved by assumption or silent code change.
5. **No baseline/golden correctness.** Never define correctness by comparing
   against stored reference outputs, golden files or regression references.
   Correctness
   comes from single-source formulas, sympy-generated code and runtime
   certificates.
6. **Tests are auxiliary.** pytest/hypothesis checks codify invariants but do
   not define correctness. CI runs ruff, the single-source gate and tests.
7. **No stubs.** `kc761tool/` must not contain
   `NotImplementedError`, placeholder bodies or unfinished paths. The
   single-source gate enforces the stub check for `kc761tool/core`; anywhere else,
   an unfinished path must fail loudly rather than degrade silently.
8. **Strict mode.** `--strict` or `KC761TOOL_STRICT=1` enables all certificate
   suites (KKT, conservation, PSD, column sums, efficiency bounds,
   monotonicity, positivity, finiteness). Basic schema/shape/finiteness
   validation is always on. Certificates fail fast and report the formula ID.
9. **Formula IDs.** Every formula gets an ID in `docs/derivations.md`; code
   references it. IDs are append-only; derivations may gain detail but must not
   contradict frozen decisions.
10. **Products.** Read/write only through `kc761tool/schema/io.py`: atomic write,
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
| `kc761tool/core/model.py`, `binning.py`, `kernel.py`, `response.py`, `projection.py` | core | formula IDs F-MODEL/F-BIN/F-KERN/F-RESP/F-PROJ |
| `kc761tool/core/solver.py`, `covariance.py`, `uncertainty.py` | core | F-SOLVE/F-COV/F-UNC |
| `kc761tool/schema/axes.py`, `products.py`, `io.py` | schema | product contract; changes go through the contract-change process |
| `kc761tool/schema/_uproot.py` | schema | verified spike helpers; keep the head comment in sync with `docs/formats.md` |
| `kc761tool/calib/` | calib | F-CAL |
| `kc761tool/unfold/` | unfold | uses F-SOLVE/F-UNC |
| `kc761tool/sim/` | sim | F-SIM |
| `kc761tool/cli/`, `kc761tool.py`, `kc761tool/__main__.py` | cli | CLI surface only; no numerics |
| `kc761tool/errors.py`, `kc761tool/runtime.py` | infrastructure | change requires a plan update |
| `tests/`, `tests/fixtures/synthetic.py` | shared | fixture changes must stay deterministic |
| `tools/` | shared | kernel generation and the single-source gate; keep `docs/derivations.md` in sync |
| `examples/` | shared | shipped TOML examples; keep in sync with `README.md` and `docs/formats.md` |
| `docs/plan.md` | authoritative | edit only to record a new decision |
| `docs/architecture.md`, `docs/formats.md`, `docs/derivations.md` | owning layer | update in the same change as the code |
| `work/` | user | untracked data and products; tooling must never delete or overwrite without `--force` |

## Contract-change process

1. Raise the point as a question (existing contract points are recorded in
   `docs/plan.md` Appendix A).
2. Record the decision in `docs/plan.md` Section 1 (or Appendix A resolution).
3. Update the affected formula registry, ownership table and product schema.
4. Implement it in the owning layer; no other contract may bypass or
   duplicate the agreed contract.

## Verification commands

```bash
ruff check .
pytest -q -m "not g4 and not root"
python tools/check_single_source.py
python kc761tool.py --help
python -m kc761tool --help
for c in calib compose sim unfold; do python kc761tool.py "$c" -c "examples/$c.toml" --dry-run; done
```

CI runs the same commands on Python 3.12 and 3.13, installing the runtime
requirements except `geant4-pybind` (no Geant4, no ROOT executable; D-180).
