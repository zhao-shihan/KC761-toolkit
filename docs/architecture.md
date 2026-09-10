# Architecture

Status: W0/W1/W2 implemented. The authoritative specification is
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
| `core/model.py` | energy calibration (dual basis), resolution, positivity certificate | F-MODEL-1..5 |
| `core/model.py` | energy calibration, resolution model, positivity/monotonicity certificates | F-MODEL-1..5 |
| `core/binning.py` | channel/energy grids, working window/pad, support limit | F-BIN-1..3 |
| `core/kernel.py` | exact Gaussian bin integrals, smoothstep taper, kernel derivatives, sparse assembly | F-KERN-1..4 |
| `core/response.py` | response matrix C, composed matrix R, window slicing, response Jacobian | F-RESP-1..4 |
| `core/projection.py` | rebinning/folding projections and variance propagation | F-PROJ-1..2 |
| `core/solver.py` | non-negative QP, regularization spec, KKT certificate | F-SOLVE-1..3 |
| `core/covariance.py` | Fisher information, `s**2` scaling, optional profile diagnostic | F-COV-1..3 |
| `core/uncertainty.py` | strict stat/syst propagation, simulation MC term, band combination | F-UNC-1..3 |
| `core/_gen/` | committed sympy-generated kernels with manifest and import-time freshness check | F-MODEL/F-KERN (D-77) |
| `schema/axes.py` | `Axis` (edges + unit), axis constructors and `reported_parameter_axis` | F-BIN-1 |
| `schema/products.py` | product containers, `product_kind` dispatch, object names, meta field/type tables, provenance | F-IO-1 |
| `schema/io.py` | atomic write, reopen validation, overwrite policy, provenance assembly, product certificates | F-IO-1 |
| `schema/_uproot.py` | verified low-level uproot helpers (spike): variance buffers, axis titles/units, bin labels, meta RNTuple | F-IO-1 |
| `calib/model.py` | global calibration objective over datasets: fixed-axis `C_fit`, F-CAL-1 weights, start values/bounds, analytic prediction Jacobian and exact gradient | F-CAL-1/F-CAL-3/F-CAL-4 |
| `calib/scaling.py` | per-dataset quadratic-Bezier scale with the fixed middle control channel and analytic derivatives | F-CAL-2 |
| `calib/covariance.py` | full-parameter Fisher inverse, scale marginalization, reported-basis transform, `s**2` scaling | F-CAL-5 |
| `calib/fit.py` | single bounded trust-region Gauss-Newton stage, certificates, product write, figure; public `run_fit` | F-CAL-1..5 |
| `calib/product.py` | export `C` on the channel-derived axis `E(i +- 1/2)` (D-101), reported parameters, clamp record | F-RESP-1/F-IO-1 |
| `calib/report.py` | text report of chi2/dof, parameters, scales and clamp statistics | - |
| `calib/plot.py` | calibration figure on the shared plotting style | - |
| `plotting/` | shared style, palette and atomic figure saving (leaf) | D-70 |
| `unfold/` | window/pad orchestration, uncertainty bands, report, plots | F-SOLVE/F-UNC |
| `sim/` | geometry, materials, sources, detector, physics, actions, runner | F-SIM-1..5 |
| `cli/` | argument parsing, logging, strict mode, exit codes | - |

## Contract surfaces

* Public product containers and object names are frozen in
  `schema/products.py`; `schema/io.py` is the only allowed read/write path.
  The containers are `CalibProduct`, `SimProduct`, `ComposeProduct`,
  `UnfoldProduct` and `SpectrumProduct`; `read_product` dispatches on the
  on-disk `product_kind` (unfold splits into full/`calib_only` by `mode`).
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
* `kc761/core/_gen/` is generated by `tools/generate_kernels.py` and guarded
  by `_manifest.json`; the mechanical single-source gate is
  `tools/check_single_source.py`.

## Open points

See `docs/plan.md` Appendix A (items 3, 4, 5, 6 and 8 remain open) and the
W1 interface repairs recorded in `docs/plan.md` section 1.7. The non-strict
resolution clamp (D-73) and the removal of the SNIP peak mask (D-74) are
decided and implemented in `kc761/core/`.
