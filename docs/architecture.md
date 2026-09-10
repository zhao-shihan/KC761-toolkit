# Architecture

Status: W0-W6 implemented (W7 pending). The authoritative specification is
[docs/plan.md](plan.md); this document explains the layering rules that the
code must keep.

## Layering

```
cli  ->  calib / unfold / sim / plotting
           |        |        |
           v        v        v
        schema  (products, axes, uproot IO)
           |
           v
        core  (pure numerics)
```

Rules:

1. `kc761/core` imports only the standard library plus `numpy`, `scipy` and
   (inside generated kernel modules) `numba`/`sympy`. It never imports
   `uproot`, `matplotlib`, Geant4, or any other `kc761` subpackage.
2. `kc761/schema` owns all file IO (`uproot`) and the product contracts. It
   may import `core` for elementary types but never `calib`, `unfold`, `sim`,
   `plotting` or `cli`.
3. `kc761/calib`, `kc761/unfold` and `kc761/sim` contain orchestration and
   physics/fit logic. They import `core` and `schema`.
4. `kc761/plotting` and `kc761/cli` are leaves. `plotting` must not duplicate
   style helpers across callers (single shared implementation).
5. `kc761/sim` is the only place allowed to import Geant4, and only lazily
   inside functions, so that the contract layer imports in a no-G4
   environment.

## Module map

| Module | Responsibility | Formula IDs |
|--------|----------------|-------------|
| `core/model.py` | energy calibration (dual basis), resolution, positivity/monotonicity certificates | F-MODEL-1..5 |
| `core/binning.py` | channel/energy grids, working window/pad, support limit, fixed source-mode MC axis | F-BIN-1..4 |
| `core/kernel.py` | exact Gaussian bin integrals, smoothstep taper, kernel derivatives, sparse assembly | F-KERN-1..4 |
| `core/response.py` | response matrix C, composed matrix R, window slicing, response Jacobian | F-RESP-1..4 |
| `core/projection.py` | rebinning/folding projections and variance propagation | F-PROJ-1..2 |
| `core/solver.py` | non-negative QP, regularization spec, KKT certificate | F-SOLVE-1..3 |
| `core/covariance.py` | Fisher information, `s**2` scaling, optional profile diagnostic | F-COV-1..3 |
| `core/uncertainty.py` | strict stat/syst propagation, simulation MC term, band combination | F-UNC-1..3 |
| `core/_checks.py` | shared array/shape/finiteness guards | - |
| `core/_linalg.py` | shared symmetric-positive-definite factorization policy | - |
| `core/_gen/` | committed sympy-generated kernels with manifest and import-time freshness check | F-MODEL/F-KERN (D-77) |
| `schema/axes.py` | `Axis` (edges + unit), axis constructors and `reported_parameter_axis` | F-BIN-1 |
| `schema/products.py` | product containers, `product_kind` dispatch, object names, meta field/type tables, provenance | F-IO-1 |
| `schema/io.py` | atomic write, reopen validation, overwrite policy, provenance assembly, product certificates | F-IO-1 |
| `schema/_uproot.py` | verified low-level uproot helpers (spike): variance buffers, axis titles/units, bin labels, meta RNTuple | F-IO-1 |
| `calib/model.py` | global calibration objective over datasets: fixed-axis `C_fit`, F-CAL-1 weights, start values/bounds, analytic prediction Jacobian and exact gradient | F-CAL-1/F-CAL-3/F-CAL-4 |
| `calib/scaling.py` | per-dataset quadratic-Bezier scale (free middle control abscissa `s0`) and analytic derivatives | F-CAL-2 |
| `calib/covariance.py` | full-parameter Fisher inverse, scale marginalization, reported-basis transform, `s**2` scaling | F-CAL-5 |
| `calib/fit.py` | single bounded trust-region Gauss-Newton stage, certificates, product write, figure; public `run_fit` | F-CAL-1..5 |
| `calib/product.py` | export `C` on the channel-derived axis `E(i +- 1/2)` (D-101), reported parameters, clamp record | F-RESP-1/F-IO-1 |
| `calib/report.py` | text report of chi2/dof, parameters, scales and clamp statistics | - |
| `calib/plot.py` | calibration figure on the shared plotting style | - |
| `plotting/` | shared style, palette and atomic figure saving (leaf) | D-70 |
| `unfold/inputs.py` | product loading, axis bitwise checks, upstream provenance checks | D-114 |
| `unfold/compose.py` | `run_compose`: full-axis composition and compose artifact | F-RESP-2/F-RESP-3 |
| `unfold/selection.py` | energy window to channel/primary selection and data-side fit weights | F-UNF-1/F-UNF-2 |
| `unfold/solve.py` | exact-zero pruning, non-negative solve, strict uncertainty bands, diagnostics | F-UNF-3/F-UNF-4 |
| `unfold/unfold.py` | `run_unfold`: full and `calib_only` orchestration, product assembly | F-UNF-5/F-UNF-6 |
| `unfold/report.py`, `unfold/plot.py` | text report and figure on the shared plotting style | D-70 |
| `sim/config.py` | run defaults: base seed, event-block size, raw histogram names, memory budget | F-SIM-7/D-123/D-124 |
| `sim/geometry.py` | frozen detector geometry dataclass with per-field provenance | D-34 |
| `sim/materials.py` | material compositions/densities and the lazy Geant4 builder | D-32/D-34 |
| `sim/sources.py` | seven-key source registry, geometry dataclasses, matrix modes, primary axis and column schedule | F-SIM-1/D-122 |
| `sim/detector.py` | Geant4 detector/source construction and the pure plane/sphere source builders | D-31/D-34 |
| `sim/physics.py` | physics list (Penelope, 0.1 mm) and decay/GPS configuration | D-32 |
| `sim/generator.py` | matrix primary sampling and the deterministic seed derivation (pure functions) | F-SIM-4/F-SIM-7 |
| `sim/actions.py` | run/event/stepping actions for both scoring paths, pulse merging | F-SIM-5 |
| `sim/certificates.py` | physical-layer certificates: accounting, variance, efficiency, boundary | F-SIM-1/F-SIM-2/F-SIM-3/F-SIM-6 |
| `sim/runner.py` | memory-budgeted workers, uproot merge, product output, interactive entry | F-SIM-7/D-120/D-121/D-124/D-125 |
| `cli/` | one command with six subcommand modules, output/provenance plumbing and the `sim` batch driver | D-66/D-67/D-68/D-134 |
| `cli/config.py` | strict TOML parsing (stdlib `tomllib` only; no Geant4, no numerics) for `sim`/`calib`/`compose`/`unfold` | D-129..D-141 |

## CLI orchestration (W6)

* `kc761/cli/_common.py` owns the cross-command plumbing: the `out/<command>/`
  default-name convention (D-19), full-argv provenance pairs (D-18), the
  config/run-option mutual-exclusion check (D-130) and the shared option
  groups. Handlers import their workstream entry point lazily, so `--help`
  does not import Geant4 or the numerics stack.
* The library entry points (`run_fit`, `run_compose`, `run_unfold`,
  `run_source`, `run_matrix`) take an optional `extra_inputs` sequence that is
  hashed into the product provenance. Config mode passes the configuration
  file there, satisfying D-133 without duplicating provenance assembly.
* `kc761/cli/sim.py` config mode expands each `[[sim.runs]]` entry into
  `sys.executable -m kc761 sim ...` and runs the children serially from the
  repository root. The parent validates the config, manages resume
  (`verify_product`), aggregates failures (return 1 if any run failed) and
  never imports Geant4; the child receives the config path through the hidden
  `--provenance-input` option.
* `kc761/cli/config.py` is a leaf: standard library only, frozen dataclasses,
  unknown keys and missing sections are errors, relative paths resolve against
  the config file directory.
* `run_fit`/`run_compose`/`run_unfold` are called in-process by `calib`,
  `compose` and `unfold`; the CLI only builds products (`DatasetSpec`,
  `SpectrumProduct`), resolves names, and prints the library report.

## Contract surfaces

* Public product containers and object names are frozen in
  `schema/products.py`; `schema/io.py` is the only allowed read/write path.
  The containers are `CalibProduct`, `SimProduct`, `ComposeProduct`,
  `UnfoldProduct`, `SpectrumProduct` and `McSpectrumProduct`; `read_product`
  dispatches on the on-disk `product_kind` (unfold splits into
  full/`calib_only` by `mode`; `mc_spectrum` is the source-mode simulated
  spectrum, D-120).
* The `meta` RNTuple field list is specified in `docs/formats.md` section 4
  and implemented by `products.meta_field_types`; adding or renaming a field
  is a contract change (see `AGENTS.md`).
* `schema/io.py` exposes `build_provenance`, `sha256_file`,
  `fingerprint_for`/`input_sha256` (input lookup for later workstreams),
  `write_product`, `read_product` and `verify_product`. The readers/writers
  take keyword-only `strict: bool = False`: always-on schema/axes/units/shape/
  finiteness validation runs in both modes, while the product certificates
  (F-RESP-1/2, F-COV-2, F-SIM-1/2, F-UNC-3, F-IO-1) run only under strict.
* `core` dataclasses (`InternalCalibration`, `ReportedCalibration`,
  `ResponseMatrix`, `ComposedResponse`, `KktCertificate`, `UnfoldSolution`,
  `CovarianceEstimate`, `UncertaintyBands`, `SparseTriples`,
  `KernelGradients`, `RegularizationSpec`, `ProjectionPlan`,
  `BandComponent`) are the frozen interfaces between numerics and
  orchestration. The W1 repairs in `docs/plan.md` section 1.7 added
  keyword-only parameters (`channel_max`, `primary_edges_kev`, `strict`,
  propagation context) and the `ComposedResponse.channel_low/high` fields;
  existing positional signatures are unchanged.
* The fixed source-mode Monte-Carlo axis (`0..4096 keV / 4096 bins`) is single
  sourced in `core.binning.source_mode_deposition_edges_kev()` (F-BIN-4):
  `kc761.calib` imports it for the parameter-independent fit-time `C_fit`, and
  `kc761.sim` uses it for the source-mode `mc_spectrum`. The matrix-mode primary
  and deposition axes are both the calibration product's `C.y` (D-121 revised),
  so matrix `G` is square and does not use this fixed axis.
* `kc761/core/_gen/` is generated by `tools/generate_kernels.py` and guarded
  by `_manifest.json`; the mechanical single-source gate is
  `tools/check_single_source.py`.

## Open points

See `docs/plan.md` Appendix A. Items 4 (plot CLI surface) and 8 (csv2root
strict grammar) are resolved by W6 (D-142; `docs/formats.md` section 7). Item 5
(optimizer controls) is addressed by the W6 `calib --max-iter/--tolerance`
flags, which map onto `FitSettings` and keep the D-107 defaults when omitted.
Item 3 (`calib --sim` naming) remains open: the W6 surface keeps `--sim`,
which reads an `mc_spectrum` product. The non-strict resolution clamp (D-73)
and the removal of the SNIP peak mask (D-74) are decided and implemented in
`kc761/core/`.
