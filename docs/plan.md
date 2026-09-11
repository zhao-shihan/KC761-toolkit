# KC761 Toolkit Rewrite Plan

Status: **frozen** (all decision branches answered; see the decision register).
Authority: this document is the single specification for the rewrite. When code
and this document disagree, the code is wrong unless an open contract point
(Appendix A) says otherwise.

## 0. How to read this plan

* Section 1 is the decision register: every frozen decision, grouped by area.
* Sections 2-6 specify the target architecture, data contracts, mathematics,
  correctness machinery and engineering rules.
* Section 7 is the deletion list (executed in W7).
* Section 8 defines the workstreams W0-W7 and their exit criteria.
* Appendix A lists contract points that are *not* decided yet; they must be
  raised as questions, never resolved unilaterally.
* Appendix B records the W0 deliverable.

Correctness is established by construction and by runtime certificates, not by
comparison against baselines, golden files or regression references. No test
suite defines "correct"; tests are auxiliary (Section 5).

## 1. Decision register

### 1.1 Governance and scope

| ID | Decision |
|----|----------|
| D-1 | Whole-repository rewrite, including the C++ macros. Old packages are replaced by `kc761/` and deleted in W7. |
| D-2 | No backward compatibility: old formats are not read, old command lines are not preserved. |
| D-3 | Correctness from code construction (single source of truth, symbolic generation) plus runtime certificates; never from baseline/regression comparisons. |
| D-4 | Keep the `kc761` name for packages and the command. |
| D-5 | Repository artifacts (docs, docstrings, CLI text, comments) are written in **English**. |
| D-6 | Python >= 3.12. |
| D-7 | **No packaging**: no `pyproject.toml`, no `setup.py`, no installable distribution. The supported entry points are the repository-root launcher `kc761.py` and `python -m kc761`. Tool configuration uses `ruff.toml` and `pytest.ini` only. |
| D-8 | Dependencies are not constrained in advance; introduce what the work needs and record it here. |

### 1.2 Persistence and data contracts

| ID | Decision |
|----|----------|
| D-10 | Pure `uproot` direct read/write. The binary export protocol, `rootcxxfrontend`, and the C++ writer macros are removed. No dependency on a `root` or `hadd` executable. |
| D-11 | Multi-worker simulation merging is done in Python with `uproot`. |
| D-12 | Product metadata (settings, parameters, provenance) is carried by a single `meta` **RNTuple**. Revised 2026-09-10: uproot's default object for dict assignment is an RNTuple, and the user decided to adopt it; W0 pins it explicitly with `mkrntuple` so the contract does not depend on uproot defaults. The earlier wording ("meta TTree") is superseded. |
| D-13 | Covariance matrices (e.g. `param_cov`) are stored as `TH2D` with axis labels equal to the parameter names. |
| D-14 | Units are part of key/field names: `_kev`, `_mm`, `_counts`, ...; dimensionless keys carry no suffix. |
| D-15 | sigma bands are stored as `content = sigma`, `fSumw2 = sigma**2`, so both `values()` and `errors()` return the 1-sigma band. |
| D-16 | Writes are atomic (`temp` + close + reopen validation + rename). |
| D-17 | Existing output files are refused by default; `--force` overwrites. |
| D-18 | Every product records full provenance: git revision and dirty flag, Python/dependency/Geant4 versions, sha256 of every input file, and the complete CLI arguments. |
| D-19 | Default outputs live in `work/<subcommand>/` (root revised by D-163; originally `out/`); names encode the inputs and run parameters (e.g. `<calib>-<mode>-n<N>-s<SEED>.root`). |
| D-20 | Channel-axis products: matrix convention is `x = output side`, `y = input side` (C: channel/deposition; G: deposition/primary; R: channel/primary). |

### 1.3 Simulation and physics

| ID | Decision |
|----|----------|
| D-30 | The radioactive-source Geant4 mode is kept, but the per-event **ntuple is deleted** (no consumer exists). |
| D-31 | Matrix modes: plane-front and circumscribed-sphere sources are kept and parameterized; current geometry/angular defaults are preserved. |
| D-32 | Physics list and cut stay `G4EmPenelopePhysics` with the 0.1 mm default cut; physics behaviour does not change. |
| D-33 | Source-mode spectrum and pile-up behaviour stay as-is: 4096 bins over 0-4096 keV, 10 us pulse merging. |
| D-34 | Detector geometry values are unchanged; they are collected into one frozen dataclass with per-field provenance comments ("assumed" where unknown). |
| D-35 | Matrix-mode sampling: a fixed number of events per active primary-energy column, uniform within the column; each column total `N_j` is known exactly. |
| D-36 | The simulation product stores counts plus `N_j` as an independent `TH1D`; zero-deposition information is recovered from `N_j`, and detection efficiency is derived, not stored. |
| D-37 | Per-bin variance is the exact binomial/multinomial value (`N_j p (1-p)`); `fSumw2` carries it. Multi-bin correlations are reconstructed downstream from counts and `N_j`. |
| D-38 | Simulation workers are limited by a memory budget, not just by CPU count. |

### 1.4 Mathematics and statistics

| ID | Decision |
|----|----------|
| D-40 | Energy calibration keeps the dual basis: fit in `(c0, k1, k2, k3)`, report/store `(c0, c1, c2, c3)`; the basis transform has a single implementation. |
| D-41 | Resolution keeps the quadratic Bernstein form in `sigma**2`; the expression is used **without abs**. Negative variance is a strict-mode certificate failure. |
| D-42 | Response kernels use exact Gaussian bin integrals (error function) with a soft support taper; the extended binning geometry is frozen once and never depends on fit parameters. |
| D-43 | `R = C . p_tilde . diag(eta)` is composed over the **full** primary axis, then the working window (with pad) is sliced. |
| D-44 | The working window is `[chlo - pad, chhi + pad]` for solving and uncertainty propagation; only `[chlo, chhi]` is reported. |
| D-45 | Regularization is Tikhonov `chi2 + alpha * ||D mu||**2`; `--alpha` is mandatory (no default, no automatic selection). |
| D-46 | `D` is normalized so that `alpha` is dimensionless and comparable across problems. |
| D-47 | The fit uses one quasi-Newton stage (L-BFGS-B / trust-constr class) with analytic gradients generated from sympy. |
| D-48 | Fit weights keep the current formula `var = max(stat, 1) + (syst_frac * data)**2 + MC` with `syst_frac = 0.10`; the formula and the floor are documented. **Default revised to 0.05 by D-169** (2026-09-11). |
| D-49 | The parameter covariance is the analytic Fisher `F**-1`, scaled by `s**2 = chi2/dof` by default (PDG convention) and recorded in the product. Profile covariance is an optional diagnostic only. |
| D-50 | Uncertainty bands use the strict decomposition `total**2 = stat**2 + syst**2`; `stat` is pure data statistics, `syst` contains data-side `syst_frac`, calibration covariance and simulation MC terms. |
| D-51 | The non-negative QP solver is self-implemented (banded Cholesky active set) and emits a KKT certificate. |
| D-52 | `MAX_CHANNELS` is the validated support limit (initially 4096); inputs beyond it fail fast with a memory estimate instead of being accepted. |

### 1.5 Correctness, CLI, quality

| ID | Decision |
|----|----------|
| D-60 | Formula single source: each formula/constant has exactly one implementation; derivatives/kernels are generated from sympy, never hand-copied. |
| D-61 | Runtime certificates run in **strict mode only** (`--strict` or `KC761_STRICT=1`); all suites (KKT, conservation, PSD, column sums, efficiency bounds, monotonicity, finiteness). |
| D-62 | Basic schema/shape/finiteness validation is always on and cannot be disabled. |
| D-63 | Certificates fail fast, with the offending formula ID and inputs recorded. |
| D-64 | Each formula carries an ID registered in `docs/derivations.md` with its derivation and approximations; code docstrings reference the ID. |
| D-65 | Tests (pytest + hypothesis) are auxiliary only and never define correctness. CI runs ruff + tests. |
| D-66 | CLI: a single `kc761` command with six subcommands `calib`, `unfold`, `sim`, `compose`, `csv2root`, `subbkg`; window options use long names with short aliases; `app/*.py` is deleted. |
| D-67 | Logging uses stdlib `logging` with the format `[kc761.<command>] level: message`; errors use the `Kc761Error` hierarchy and a uniform `[kc761.<command>] error:` prefix. |
| D-68 | Exit codes: 0 success, 1 runtime failure, 2 usage error. |
| D-69 | The static quality gate is **ruff only** (no separate type checker); CI runs ruff and the auxiliary tests. |
| D-70 | Plots keep the pre-rewrite visual style. The "shared plotting code" clause is superseded by D-153 (per-package, legacy-faithful figure modules). |
| D-71 | No hard wall-clock performance target. The implementation must still apply reasonable optimizations (sparse-first, no dense `n**2` intermediates, no redundant recomputation). |
| D-72 | Legacy ROOT products are not read; the input CSV files will be provided by the user and re-imported through the new `csv2root` strict parser. |

### 1.6 W1 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-73 | Non-strict resolution negativity: when `sigma**2 < 0` on the export grid and strict mode is off, clamp `sigma` to the documented floor `SIGMA_FLOOR_KEV`, emit a warning and record it (W4 writes the record into `meta`). Strict mode still raises the F-MODEL-5 certificate error. Closes Appendix A item 1. |
| D-74 | The SNIP peak mask (`snip_iter`, `mask_z0`, `mask_floor`) is removed entirely; `D` is only the normalized finite-difference operator (F-SOLVE-1). Closes Appendix A item 2. **Superseded by D-154** (2026-09-10): a redesigned SNIP peak mask is part of the default operator (F-SOLVE-4..6). |
| D-75 | W1 owns every `core` derivative. Additional formula IDs: F-KERN-4 (`d p / d c`, `d p / d sigma` and the taper derivatives) and F-RESP-4 (chain assembly of `dC/dq`), appended to the registry; further IDs may be appended as derivations require. |
| D-76 | Support taper: `w = 3 s**2 - 2 s**3` with `s = clip((n*sigma - abs(x)) / sigma, 0, 1)`, equivalently `w = 1 - 3 r**2 + 2 r**3` with `r = clip((abs(x) - (n-1)*sigma) / sigma, 0, 1)`; exact column renormalization follows (F-KERN-2). |
| D-77 | `sympy` is a runtime dependency. `tools/generate_kernels.py` generates `kc761/core/_gen/`; generated artifacts are committed and their headers record the sympy version, the formula IDs and the exact command. Imports perform a staleness check and regenerate when necessary. CI installs sympy. |
| D-78 | New product `format_version` starts at 1 and is independent of the legacy calibration version 3. Closes Appendix A item 7. |

### 1.7 W1 interface repairs (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-79 | F-BIN-2 is revised: there is **no** parameter-dependent bin selection. C is evaluated over the full deposition axis and the requested channel rows; the smoothstep taper is exactly zero beyond `n_sigma`, so the objective is continuous in the parameters by construction. `FrozenBinning` and `freeze_extended_binning` are removed (the legacy selection was a conservative superset that degenerated to the full axis and caused chi2 jumps). |
| D-80 | D-46 implementation correction: the difference operator is scaled on the right, `D_tilde = D . diag(sqrt(diag(R^T W R)))`, so the penalty differences the signal-to-noise normalized solution; zero-curvature columns are dropped, and `alpha` is dimensionless (the earlier left-multiplied form carried counts). |
| D-81 | Binning repairs: `working_window` takes `calibration`, `resol_params`, `channel_max` (keyword-only) and converts the local resolution width to channels; `build_response_matrix` takes the required keyword-only `channel_max` of the internal basis. |
| D-82 | Response repairs: `compose_response` takes the required keyword-only `primary_edges_kev`; `ComposedResponse` gains defaulted `channel_low`/`channel_high` fields, `column_sums`/`efficiency` keep full-axis semantics after slicing, and F-RESP-2/F-RESP-3 certificates check the full-axis and sliced identities separately. |
| D-83 | Derivative interfaces: F-KERN-4 exposes the **unnormalized** local kernel derivatives (edge, sigma, centre, source); the renormalization quotient rule (including the global per-column sum) is assembled once in F-RESP-4 and reused by F-UNC-2. `ProjectionPlan` gains the `weights` matrix. |
| D-84 | Solver/covariance/uncertainty mechanics: F-SOLVE-3 metrics are normalized by `max(1, norm_inf(b))` and `max(1, norm_inf(mu))`; `scaled_covariance` fails on `dof < 1` and on a non-positive-definite Fisher matrix; F-UNC-1/2 gain keyword-only context inputs (`stat_variance`, `spectrum`, `sigma_fit`, `fisher` = the full half-Hessian from F-SOLVE-1). |
| D-85 | Derivative generation: the taper derivative uses the clipped smoothstep chain rule with a vanishing prefactor outside the transition band; `verify_energy_monotonicity` checks the exact quadratic vertex of `dE/dch` instead of a sampling grid. |
| D-86 | F-UNC-1/F-UNC-2 use the **reduced** free-set system at a boundary solution (`H_FF^-1` with active variables fixed at zero), not the full inverse Hessian; using the full inverse would overstate the uncertainty of every free direction whenever a constraint is active. Found in the W1 self-review and verified against finite differences on 200 random problems. |

### 1.8 W2 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-87 | `meta` gains an explicit `product_kind` field (`calib` / `sim` / `compose` / `unfold` / `spectrum`); the two unfold variants share `product_kind = unfold` and are separated by `mode` (`unfold` / `calib_only`). `read_product` dispatches on `product_kind` + `mode`. |
| D-88 | New `SpectrumProduct` container; `SimProduct` gains `mode_name`, `geometry_name`, `geometry_param_mm`, `angular_distribution` and uses `mode: int` with `mode_name: str` (aligned with `docs/formats.md`). |
| D-89 | The dedicated `calib_sha256`/`sim_sha256` meta fields are removed; every input fingerprint lives in `provenance.inputs` (path + sha256). W2 exposes `fingerprint_for` / `input_sha256` helpers so later workstreams can verify an input path against a recorded digest. |
| D-90 | `write_product`, `read_product` and `verify_product` take a keyword-only `strict: bool = False`. |
| D-91 | All product-level strict certificates are implemented in W2 and run under `strict`; physical-layer certificates remain owned by W3/W4/W5. |
| D-92 | Any object outside the envelope of the product kind is a `SchemaError` (no warning, no ignore); duplicate on-disk object versions are rejected too. |
| D-93 | Writing in a non-git environment records `git_revision = "unknown"` and `git_dirty = 0`, emits one warning and does not abort. |
| D-94 | `README.md` is not touched in W2 (owned by W6/W7). |

### 1.8.1 W2 metadata finalization (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-95 | Calibration parameters are stored as JSON vectors `params_reported_json` (4, order `c0..c3`) and `resol_params_json` (3, order `b0..b2`) plus the scalar `channel_max`; the single source of parameter names is `core.model.PARAM_NAMES_REPORTED`, and `param_cov` carries them as TAxis bin labels. |
| D-96 | `compose` meta carries only the common fields; its inputs are recorded in `provenance.inputs`. |
| D-97 | `unfold` in `calib_only` mode carries only the common fields plus `mode`; the full mode additionally carries the 11 unfold settings. |
| D-98 | The always-on required `fSumw2` objects are `primary_to_deposition` (sim), `sigma_statistical` / `sigma_systematic` / `sigma_total` (full unfold) and `kc761_spectrum` (spectrum); all other variance buffers are optional. |
| D-99 | Compose follows the frozen F-RESP-2 derivation: `primary_efficiency` is the detection efficiency `colsum(G)/N_j` (F-SIM-3) and the certificate checks `R` column sums against the **reachable** detected mass/N via the core helper, not against `eta`. R column sums equal `eta` only when every deposition column has a reachable channel column; an exactly-zero C column is therefore allowed. |

### 1.9 W3 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-100 | The calibration forward model is `data_d ~= diag(scale_d(ch)) . (C_fit(q) @ mc_d)` with the shared internal core `q = (c0, k1, k2, k3, b0, b1, b2)` (F-CAL-1/F-CAL-4); each dataset keeps its own quadratic-Bezier `scale_d(ch)` (F-CAL-2). |
| D-101 | The fit-time `C_fit` uses the **fixed uniform deposition axis** `0..4096 keV / 4096 bins` (the source-mode MC axis, D-33), so its geometry is parameter independent (D-79). After the fit the **export** `C` is rebuilt from `q_hat` on the channel-derived non-uniform axis `E(i +- 1/2)` (the legacy axis construction), keeping the W5 matrix `G` and the W4 `compose` aligned. Both matrices use the same single core kernel. |
| D-102 | The `+MC` term of F-CAL-1 is the diagonal MC variance of the folded prediction: `(scale_d * sqrt((C_fit**2) @ var_mc_d))**2`, i.e. the diagonal of the C/identity projection of the per-deposition-bin MC variance, carrying the `scale**2` factor. |
| D-103 | Per-dataset scaling is the legacy quadratic-Bezier model `s0 + s1..s3` (F-CAL-2) with the legacy parameter bounds, and `s0` (the middle control **abscissa**) stays a free fit parameter: the four parameters count in the degrees of freedom (`n_free = 7 + 4 D`). Fixing `s0` at the window midpoint would linearize the abscissa parametrization and change the model class to degree-2 Bernstein polynomials, which is not the legacy model. The family is a genuine parabola for `s0` away from the midpoint; at a constant scale, or at `s0` exactly equal to the midpoint, the `s0` direction is an exact gauge, so the scale is seeded non-constant (F-CAL-3) and the covariance marginalizes the scale stably (D-105). |
| D-104 | The fit adds no extra `sigma**2` constraint beyond the core model; it relies on the non-strict F-MODEL-5 clamp with a warning (D-73). The clamp effect is summarized in the calibration `meta` (count and clamped energy range) and strict mode still raises F-MODEL-5. The chi-square is only C0 at the clamp boundary, so the quasi-Newton step may see a kink; this limitation is documented in the F-CAL derivations. |
| D-105 | The calibration covariance is the `(c, b)` 7x7 core block of the full-parameter Fisher inverse with the per-dataset scale **marginalized**, scaled by one global `s**2 = chi2/dof` (F-COV-2) and transformed from the internal to the reported basis with the F-MODEL-2 Jacobian (`b` unchanged). Because the legacy Bezier scale has a gauge direction when the scale is (nearly) polynomial, the block is computed with the equivalent, numerically stable Schur complement `(F_cc - F_cs F_ss^+ F_sc)^-1` under Jacobi preconditioning; for an invertible scale block this equals the full-inverse core block. A non-positive-definite core Schur complement is a hard failure: the fit parameters themselves are not identifiable (no pseudo-inverse fallback for the core). |
| D-106 | The calib `meta` gains `chi2`, `dof`, `covariance_scale`, `fit_status`, `scales_json` (per-dataset label plus the four `s` parameters), `scale_bound_flags_json` (per-dataset 0/1 flags for scale parameters sitting on a fit bound; `s0` at its bound is the allowed polynomial-stratum limit of D-103) and the resolution-clamp record `resol_clamp_count`, `resol_clamp_energy_low_kev`, `resol_clamp_energy_high_kev`; `CalibProduct` gains the matching defaulted fields. `docs/formats.md` section 4 is updated in the same change. |
| D-107 | Optimizer controls are fixed, documented defaults inside `FitSettings` (`maxiter`, `ftol`, `xtol`, `gtol`); the fit runs one bounded trust-region (reflective) Gauss-Newton stage with the analytic residual Jacobian (`scipy.optimize.least_squares`, `method="trf"`, `x_scale="jac"`). A plain L-BFGS-B line search stalls on the nearly flat `sigma -> 0` corner of the resolution model, so the named "L-BFGS-B/trust-constr" family is realized by its bounded trust-region member. Exposing the controls on the CLI is W6 (Appendix A item 5). |
| D-108 | A non-converged optimizer records `fit_status` and, outside strict mode, still writes the product with the fit marked; strict mode aborts with a classified error. A degenerate fit (`dof < 1`, singular Fisher, non-finite objective) is a hard `ValidationError`/`SolverError` in every mode. |

### 1.10 W4 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-110 | Exact-zero primary-column pruning (F-UNF-3): after composing over the full primary axis and slicing the padded channel rows, primary columns whose sliced response is **exactly** zero are removed (no tolerance) and the non-negative solve runs on the remaining columns only. Pruned columns are fixed at zero in the reported solution. |
| D-111 | Energy-to-channel window mapping (F-UNF-1): `E(ch)` is evaluated at the channel **centres**; `chlo`/`chhi` are the first/last channel centre inside `[elo, ehi]`, clipped to the acquisition, and at least one centre must fall inside. `elo`/`ehi` must lie inside the primary axis range. The reported `mu` covers the primary bins whose centres lie in `[elo, ehi]`. Violations are `ValidationError`. |
| D-112 | Unfold fit weights are data-side only (F-UNF-2): `sigma_fit**2 = max(stat, 1) + (syst_frac * data)**2`. The simulation-MC term enters only the systematic band (F-UNC-2), never the weights, so the unfold does not iterate. |
| D-113 | `calib_only` relabels the channel axis to the channel-derived deposition axis `E(i +- 1/2)` stored in `C.y`, keeps counts and `fSumw2` bin-by-bin unchanged, sets `mode = calib_only`, carries only common meta plus `mode`, and needs no `alpha`. |
| D-114 | Axis and provenance validation: `C.y` (deposition) must equal `G.x` bitwise, `C.x` must equal the data channel axis, and `N_j` must use `G.y`; a mismatch is `ValidationError`. Input digests recorded by an upstream product (`fingerprint_for`/`input_sha256`) are checked against the actual file whenever present (`ProvenanceError` on mismatch). |
| D-115 | Compose orchestration lives in `kc761/unfold/compose.py` (`run_compose`); `kc761 compose` stays a thin W6 wrapper. |
| D-116 | Solver failures (iteration exhaustion, unbounded objective, non-positive-definite normal matrix) raise `SolverError` in every mode; no product and no `.part` file is left behind. |
| D-117 | `refolded` is `R . mu` evaluated on the reported channel window `[chlo, chhi]` (F-UNF-5). |
| D-118 | Unfold diagnostics (F-UNF-4): `chi2` is the weighted residual sum of squares over the fit rows; `dof = n_fit_rows - n_active` with `n_active` the number of strictly positive solution entries; `covariance_scale = 1.0` (the analytic propagation is not rescaled). |
| D-119 | F-UNC-2 core repair (found in W4, user-approved): `simulation_mc_variance` used a rank-one half-gradient derivative `d_js e_s^T` that dropped the `(R^T W C_j) mu_s` vector term and only broadcast for square `n_primary == n_deposition`. The exact multinomial propagation (full-vector derivative, non-centred `sum_j p X^2 - Xbar^2` form so the unrecorded zero-deposition category is included, reduced free-set inverse) replaces it; the F-UNC-2 derivation is corrected and an FD regression is added. |

### 1.11 W5 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-120 | New additive product kind `mc_spectrum` (source-mode simulated spectrum), separate from the measured `spectrum` kind. It carries one `kc761_mc_spectrum` TH1D (energy axis in keV, counts, required `fSumw2`) plus common meta and the fields `source_key`, `mode_name`, `geometry_name`, `geometry_param_mm`, `n_events`, `seed`, `workers`. W6 connects `calib --mc` to it; the calib library only consumes a `Histogram1D`. |
| D-121 | (revised in R1) The matrix-mode **primary** and **deposition** axes are both the calibration product's `C.y` (channel-derived, legacy square layout), bitwise (D-114); `G` has shape `(n_deposition, n_primary)` with equal axes. The fixed source-mode Monte-Carlo axis `0..4096 keV / 4096 bins` (D-33) is used only by the source-mode `mc_spectrum` and the parameter-independent fit-time `C_fit` (D-101); `kc761/calib` imports it and its former local `FIXED_DEPOSITION_*` names are aliases. |
| D-122 | `kc761/sim` owns the final source registry: the seven legacy keys `k40`, `lu176`, `am241`, `th232`, `th232-unshielded`, `ra226`, `ra226-unshielded`, carried verbatim with per-entry provenance notes; source geometry, container, shield and material values are unchanged (D-34). Appendix A item 6 is resolved by this decision. |
| D-123 | (R1 revision) Simulation randomness is deterministic for a fixed base seed **and a fixed worker count/order**: the same invocation reproduces bit-for-bit. Matrix mode derives an independent RNG stream per active primary column from `(base seed, column index)`; source mode derives one stream per fixed event block from `(base seed, block index)`. Both derivations are the single formula F-SIM-7. Bit-for-bit equivalence across *different* worker counts is not part of the contract. The default base seed is the legacy value `908136382`. |
| D-124 | The worker count is estimated from available memory and the per-worker footprint (histograms plus a documented Geant4 baseline), and the estimate is logged with its reason; an explicit `threads` argument overrides it. `MAX_CHANNELS` (D-52) still bounds the deposition axis. |
| D-125 | Each worker writes only the minimal raw ROOT histograms (spectrum, or G plus zero-deposition counts). Merging is done in Python with `uproot` (D-11): axis arrays are checked bitwise and contents summed; the final `mc_spectrum`/`SimProduct` is written through `schema.write_product` (atomic, refuse-overwrite, provenance). Temporary worker files are removed on success and on failure; no `.part` file is left behind. |
| D-126 | The source-mode interactive/visualization path is kept. The Geant4 macros move into `kc761/sim` and the interactive macro enables `/tracking/storeTrajectory 1`; the repository-root `G4History.macro` is deleted in W7. The per-event ntuple remains deleted (D-30). |
| D-127 | Physical-layer certificates owned by W5: F-SIM-1 event accounting (`sum_deposition + zero_j = N_j`, `sum_j N_j = n_events`), F-SIM-2 exact stored variance (matrix: `N_j p (1-p)`; source spectrum: `c (1 - c/P)` with `P` the recorded pulse total), F-SIM-3 `eta in [0, 1]` with `eta_j = column_sum_j / N_j`, and the new F-SIM-6 physical boundary (any G entry whose deposition-bin lower edge exceeds its primary-column upper edge is exactly zero). F-SIM-6 runs in strict mode and fails with its formula ID. |
| D-128 | The source-mode spectrum variance uses the conditional-binomial convention of D-127 with the recorded pulse total `P = sum(counts)` as the fixed total; the derivation documents this as a plug-in approximation to the multinomial over pulses. |

### 1.12 W6 config-file mode (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-129 | Configuration files are TOML read with the standard library `tomllib` (no new dependency). One file may contain the top-level tables `[sim]`, `[calib]`, `[compose]` and `[unfold]`; each subcommand reads only its own table, and a missing table is a usage error. `csv2root` and `subbkg` take no config file. |
| D-130 | Invocation is `-c/--config FILE` on `sim`, `calib`, `compose` and `unfold`. Config mode is mutually exclusive with the run-selection arguments; only the global options `--strict`, `--log-level`, `--dry-run` and `--force` are accepted alongside it. Without `--config` the existing flag surface is unchanged. |
| D-131 | Every config file declares `config_version = 1`; a missing or different value is a usage error. Any key outside the frozen schema is an error (fail loud; no silent ignore). |
| D-132 | Relative paths in a config file resolve against the directory containing that file (`~` is expanded); absolute paths are used as-is. **Superseded by D-164** (2026-09-11): they resolve against the current working directory. |
| D-133 | Products written in config mode record the config file path and sha256 in `provenance.inputs` (D-18) and the fully resolved settings in their existing settings/meta fields. |
| D-134 | `[sim]` replaces the legacy `runsim` batch runner: it carries batch options (`resume`, `force`, `dry_run`) and a list of `[[sim.runs]]` single-run specs. A run specifies a source key XOR one matrix mode (`plane-front-gamma`/`sphere-gamma`) plus `calib`; `events` is required (interactive runs are not part of config mode); `threads`, `seed`, `verbose` and `output` are optional. Runs execute sequentially (run-level parallelism is not part of the contract). |
| D-135 | Each sim run executes in a fresh child process by re-invoking `sys.executable -m kc761 sim ...` in single-run mode; the parent process never imports Geant4. This is required because a `G4RunManager` can be initialized only once per process. |
| D-136 | sim batch failure policy: a failed run is recorded and the batch continues; the process returns 1 if any run failed and 0 only when all succeeded. |
| D-137 | sim resume: `resume = true` (default) skips a run whose target exists and passes schema validation, and fails loudly when an existing target is invalid (never delete or overwrite another product); `resume = false` applies the D-17 refuse-overwrite rule. |
| D-138 | `output` is optional everywhere in config mode; when absent the D-19 default naming applies, e.g. source mode `<source>-n<N>-s<SEED>.root`, matrix `<calib-stem>-<mode>-n<N>-s<SEED>.root`, and `<command>[-<label>]...` under `work/<subcommand>/` for calib/compose/unfold. |
| D-139 | `calib` config mode runs one fit from `[[calib.datasets]]` (data, sim, label plus optional channel window and `syst_frac`); `compose`/`unfold` config mode run one operation from their tables. Mode-dependent required fields are enforced per mode: `calib_only` requires neither `alpha` nor the energy window, and the single-run CLI applies the same rule (R1 leftover). |
| D-140 | Business options `force`, `no_plot`, `dry_run` and `resume` may appear in config files; `strict` and `log_level` stay CLI-only and apply to the whole invocation. English-commented `examples/*.toml` are shipped for the four config-capable subcommands. |
| D-141 | Config parsing and validation live in `kc761/cli/config.py` as strict, frozen dataclasses; the module imports no Geant4 and no numerics. The sim batch driver only validates, expands argv, manages resume and subprocesses, and aggregates the exit code. |
| D-142 | The plot CLI surface (Appendix A item 4) is resolved as a single `--no-plot` switch (plots on by default) for `calib` and `unfold`, with `no_plot` as the config key; `compose` and `sim` produce no plots and redirecting plots to another directory is not supported. |

### 1.13 R2 review and consolidation (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-143 | Canonical matrix-mode tokens are the hyphen strings `plane-front-gamma` and `sphere-gamma`, single-sourced in `kc761.sim.sources` (`MODE_NAME_PLANE`, `MODE_NAME_SPHERE`, `MATRIX_MODE_NAMES`). They are used for the CLI flags, the config `mode` value, the default file-name token and the sim `mode_name`; underscore and short aliases are rejected (the runner accepts only the canonical name or the integer mode). |
| D-144 | The calibration simulated-spectrum option is `calib --mc` (no `--sim` alias) and the config key in `[[calib.datasets]]` is `mc`; docs, examples and help text use it. `compose`/`unfold` keep `--sim`/`sim` for the matrix simulation product. |
| D-145 | Appendix A item 5 is closed: `calib --max-iter N` and `calib --tolerance T` are part of the frozen CLI, mapping to `FitSettings(maxiter, ftol=xtol=gtol=T)`; both default to the library `FitSettings`. |
| D-146 | Revises D-111: `select_window` raises `ValidationError` when the requested energy window lies entirely outside the channel energy range; it never degenerates to a single top channel. Windows that merely extend beyond the acquisition are still clipped as before. |
| D-147 | Cross-package error taxonomy: argument/config misuse raises `UsageError` (CLI) or `ValidationError` (library inputs), product/schema violations raise `SchemaError`, and solver/covariance failures raise `SolverError`; the CLI maps `UsageError` to exit 2 and every other `Kc761Error` to exit 1. |
| D-148 | `param_cov` axis bin labels are an always-on structural part of the calib product contract and are validated on every read (previously strict-only). |
| D-149 | R2 updates `README.md` to the current implementation state. Any large-scale real end-to-end run (real CSV data, full source-mode simulation campaigns) requires prior user confirmation of parameters/budget; products are written under `work/` and are never deleted by tooling. |
| D-150 | `core.solver.solve_nonnegative` gains an optional keyword-only `normal=(H, b, penalty_scale)` injection; the unfolding layer builds `normal_equations` once and passes it, so the solver no longer rebuilds the identical half-Hessian that F-UNC already needs. Behaviour is unchanged. |
| D-151 | `uncertainty.simulation_mc_variance` keeps the exact free-set inverse `H_FF**-1` materialization (D-119). This is documented as a scale limitation: it is acceptable for the supported 2048-channel window (~32 MiB) but a band-only solve path is deferred to a later perf pass; no correctness shortcut is taken. |
| D-152 | Audit-only fields are retained deliberately and documented as such: `SpectrumProduct.source_file` (originating path), the unfold setting values recorded in the product meta, and the generator manifest `formula_ids`. They are provenance/audit records, not inputs to any numeric path; removing them would lose traceability. |
| D-153 | Supersedes the "shared plotting code" clause of D-70: each figure module (`kc761/calib/plot.py`, `kc761/unfold/plot.py`) is a self-contained, legacy-faithful port of the pre-rewrite figure (same geometry, palette, log axes, legends, bands, output-format inference). The shared `kc761/plotting/` package is removed; the `DatasetDetail` gained plotting-only raw-MC/scale fields (`raw_mc_counts`, `raw_mc_uncertainties`, `scale_params`) populated in diagnostics, never in the fit. |

### 1.14 SNIP-based regularization (2026-09-10, user-approved)

SNIP is adopted for regularization to suppress spurious peaks. The mask is not
a spurious-peak detector: it marks genuine peaks so the global `alpha` can be
raised enough to damp noise-induced oscillatory structure without eroding real
peaks. Full derivation and limits: `docs/derivations.md` F-SOLVE-4..6.

| ID | Decision |
|----|----------|
| D-154 | Supersedes D-74. The SNIP peak mask `D' = diag(rho**0.5) D` (row-stencil weights) is part of the Tikhonov operator and is **enabled by default**; `snip_enabled = false` (or the CLI switch) disables it and restores the plain F-SOLVE-1 operator. |
| D-155 | Mechanism A only: the mask is derived from the measured spectrum `y` and frozen before the solve. No solution-adaptive (IRLS) reweighting is part of this contract; it remains a possible future extension. |
| D-156 | Peak detection: `y+ = max(y, 0)`; residual from the F-SOLVE-4 baseline; resolution-matched Gaussian significance with threshold `k = 5` standard deviations; bins within `protect_sigma = 2` resolution widths of a candidate are protected; mask floor `0.1`. All values are settings with these defaults. |
| D-157 | The SNIP iteration count is resolution-derived: `m = clip(round(FWHM_bins/2), 1, m_max)` at the reported-window midpoint, with an explicit override recorded in the product. |
| D-158 | The masked operator is `D_tilde' = diag(rho**0.5) D . diag(sqrt(diag(A)))` with `rho_r = prod_{j=0}^{order} w_{r+j}` (F-SOLVE-6); `D_tilde'^T D_tilde'` stays symmetric PSD and banded, and the F-SOLVE-3 KKT certificate is unchanged. A new strict certificate F-SOLVE-6 verifies the mask, the bandwidth and the hashes. `alpha` remains mandatory and unchanged in meaning. |
| D-159 | The data-derived mask makes the reported covariance conditional on the realised mask (plug-in); the mask-selection uncertainty is not propagated. The approximation is documented and coverage is validated on synthetic pulls with the mask on and off. |
| D-160 | `UnfoldSettings`/product meta record `snip_enabled` (default true), `snip_threshold_sigma`, `snip_protect_sigma`, `snip_floor`, `snip_iterations`, `snip_max_iterations`, the clipped-negative count and index range, and the baseline/mask sha256, so a masked solve is exactly reproducible. |
| D-161 | Implementation is a W4-scope extension (F-SOLVE-4..6 in `core/solver.py`, settings/meta and certificates). Acceptance is quantitative on synthetic spectra: spurious-peak suppression, true-peak area bias, pull coverage with mask on/off, threshold/iteration robustness and bitwise reproducibility; no golden or reference outputs. |
| D-162 | Provisional defaults after the R2 sensitivity study: `snip_enabled = true`, `snip_threshold_sigma = 5`, `snip_protect_sigma = 2`, `snip_floor = 0.1`, resolution-derived `m` capped at `m_max = 8`. The truth-based synthetic closure study (D-161) remains the acceptance gate and may revise them; the realised values are recorded per product (D-160), so a re-tuned default does not invalidate existing products. |

### 1.15 Workspace layout and config path resolution (2026-09-11, user-approved)

| ID | Decision |
|----|----------|
| D-163 | The default product root is `work/` under the repository root (was `out/`). Project data, scratch and products live under `work/` (`work/data`, `work/tmp`, `work/temp`, `work/<subcommand>`); `.gitignore` ignores `work/` and no longer ignores `out/`, `data/`, `tmp/` or `temp/`. |
| D-164 | Supersedes D-132: relative paths in a TOML configuration file resolve against the current working directory, not the config file's directory; `~` expansion and absolute paths are unchanged. |
| D-165 | `csv2root` and `subbkg` default their output next to the input file: `csv2root` writes `<csv-dir>/<stem>.root`, `subbkg` writes `<signal-dir>/<signal-stem>-subbkg.root`. |
| D-166 | `compose` defaults its output next to its `--sim` input (the matrix energy response): `<sim-dir>/compose-<calib-stem>-<sim-stem>.root`. `calib`, `sim`, `unfold` keep the `work/<subcommand>/` default (D-19/D-163); the "every other command" clause of D-165 is revised accordingly. |
| D-167 | Terminology standard (complements D-144): `sim` denotes the Geant4 simulation workstream and the matrix-mode `sim` product (`primary_to_deposition`); `mc` denotes Monte-Carlo statistics and the source-mode `mc_spectrum` product consumed by `calib --mc`. The finite-MC variance of a template/response is named `mc_variance` everywhere (`propagate_systematic(..., mc_variance=...)`, `BandComponent(name="mc_variance")`); the helper keeps the name `simulation_mc_variance` to denote its simulation origin. The `sim` product, `SimProduct` and the `[sim]` config table are intentionally not renamed. |
| D-168 | `calib` fit observability and fail-fast output (2026-09-11, user-approved): `run_fit(..., progress=..., progress_every_s=1.0)` emits a `FitProgress` callback (a pre-fit summary event with `nfev == 0`, time-cadenced events and one final event) carrying the global chi2/dof, elapsed time and ms/eval. The CLI prints the summary and progress lines by default (`--no-progress` disables them; `--progress-every SECONDS` sets the interval, `0` = every evaluation). Output targets are validated before the fit through `schema.io.validate_output_path` (overwrite policy plus parent creation and writability), raising `UsageError` (exit 2); `write_product` keeps its `SchemaError` as the last line of defence. |
| D-169 | The default data-side fractional systematic uncertainty is `syst_frac = 0.05` (5%) instead of 0.10; it is single-sourced as `core.uncertainty.DEFAULT_SYST_FRAC` and used by `DatasetSpec`, `UnfoldSettings`, `run_unfold`, the `--syst-frac` CLI default and the config loaders. Explicit per-dataset values are unchanged. |
| D-170 | ROOT titles are display-only and human-readable (2026-09-11, user-approved): the axis canonical name travels in `fName` and its unit is derived from that name via `schema.axes.AXIS_UNITS`; `fTitle` is `Label (unit)` with no parentheses for unitless axes (`channel`/`counts`/`dimensionless`), and is never parsed. Histogram object titles are set from `schema.products.HUMAN_TITLES`. The naming table is fixed in `schema.axes.HUMAN_AXIS_LABELS` and `schema.products.HUMAN_TITLES`. |
| D-171 | All product-writing commands validate their output targets before doing work (2026-09-11, user-approved, extends D-168): `unfold`, `compose`, `sim` (single-run), `calib`, `csv2root` and `subbkg` check the product target and, where a figure is produced by default, the figure target, through `schema.io.validate_output_path` (overwrite policy plus parent creation/writability), raising `UsageError` (exit 2). The `sim` config batch keeps its resume logic and does not pre-check skipped runs. |

## 2. Target architecture

```
kc761.py                      # repository-root launcher (adds repo root to sys.path, calls kc761.cli)
kc761/
  __init__.py                 # version
  __main__.py                 # python -m kc761
  errors.py                   # Kc761Error hierarchy
  runtime.py                  # logging, strict-mode resolution
  core/                       # pure numerics: stdlib + numpy/scipy/numba only, no IO
    model.py                  # E(ch) dual basis, Bernstein sigma, formula registry
    binning.py                # channel/energy grids, working window/pad (D-79)
    kernel.py                 # erf bin integrals, soft taper, sympy generation convention
    response.py               # C construction, compose R = C . p_tilde . diag(eta)
    projection.py             # rebinning/folding projections
    solver.py                 # non-negative QP + KKT certificate
    covariance.py             # Fisher information, s**2 scaling, optional profile diagnostic
    uncertainty.py            # strict stat/syst propagation
  schema/                     # product contracts and IO (uproot)
    axes.py                   # Axis (edges + unit)
    products.py               # Calib/Sim/Compose/Unfold products, object names, Provenance
    io.py                     # atomic write, reopen validation, provenance, overwrite policy
    _uproot.py                # verified low-level uproot helpers (W0 spike)
  calib/                      # fit orchestration, Bezier scaling, report, plots
  unfold/                     # window/pad orchestration, report, plots
  sim/                        # geometry, materials, sources, detector, physics, actions, runner
  cli/                        # argparse tree, subcommand modules and config parsing (D-141)
tests/                        # auxiliary tests and deterministic fixtures
docs/                         # plan, architecture, formats, derivations
examples/                     # shipped TOML config examples (D-140)
```

Dependency rules:

* `core` imports only the standard library, numpy, scipy (and numba/sympy in the
  generated kernel modules). It must not import `schema`, `calib`, `unfold`,
  `sim`, `cli`, uproot, matplotlib or Geant4.
* `schema` imports `core` only for type contracts where unavoidable; it owns
  uproot and file IO.
* `calib`, `unfold` and `sim` import `core` and `schema`.
* `cli` imports everything; each figure module owns its own matplotlib style (D-153).
* Nothing imports Geant4 except `kc761/sim` (lazily, inside functions), so the
  contract layer stays importable in a no-G4/no-ROOT environment.

## 3. Data contracts

### 3.1 Envelope

Every product file contains:

* its data objects (see 3.2), written with explicit axes and units;
* a single `meta` RNTuple with format version, parameters/settings and provenance;
* no additional objects.

Writes go to `<target>.part`, are closed, reopened and validated object by
object, and only then renamed onto `<target>`. An existing target is refused
unless `--force` is given.

### 3.2 Product objects and units

| Product | Objects | Notes |
|---------|---------|-------|
| calib | `deposition_to_channel` (TH2D), `param_cov` (TH2D, 7x7, axis labels c0..c3/b0..b2), `meta` | C columns sum to 1; per-element C sigma is **not** stored (recomputed from `param_cov` + analytic Jacobian on demand). |
| sim | `primary_to_deposition` (TH2D, counts, `fSumw2` = exact binomial variance), `primary_column_totals` (TH1D of `N_j` over the primary axis), `meta` | Detection efficiency is derived: `eta_j = column_sum_j / N_j`. No ntuple, no spectrum. |
| compose | `response_matrix` (TH2D, R), `deposition_to_channel` and `primary_to_deposition` copies, `primary_column_totals`, `primary_efficiency` (derived convenience), `meta` | Inspection artifact for `R = C . p_tilde . diag(eta)`. |
| unfold | `kc761_spectrum_unfolded`, `sigma_statistical`, `sigma_systematic`, `sigma_total` (bands, content = sigma, fSumw2 = sigma**2), `kc761_spectrum_refolded`, `meta` | Calibration-only mode uses `kc761_spectrum_calibrated` and marks the mode in `meta`. |
| spectrum | `kc761_spectrum` (TH1D, counts, fSumw2), `meta` | Produced by `csv2root` and `subbkg`; carries `daq_time_s`. |
| mc_spectrum | `kc761_mc_spectrum` (TH1D, energy axis keV, counts, required fSumw2), `meta` | Source-mode simulated pulse spectrum (D-120); variance `c (1 - c/P)` (D-128). The CLI connects `calib --mc` to it (D-144). |

Units live in key names (`energy_kev`, `daq_time_s`) and axis titles; matrix
axes follow D-20.

### 3.3 `meta` RNTuple fields (finalized by W2)

The authoritative field list is `docs/formats.md` section 4; it is implemented
in `kc761/schema/products.py` (`meta_field_types`).

* Every product: `format_version` (int), `product_kind` (str), `producer`
  (str), `created_utc` (str), `git_revision` (str), `git_dirty` (int),
  `python_version` (str), `dependency_versions` (JSON), `command`, full CLI
  arguments (JSON), input paths and sha256 (JSON).
* calib adds: `channel_max`, `params_reported_json`, `resol_params_json`, and
  the W3 diagnostics `chi2`, `dof`, `covariance_scale`, `fit_status`,
  `scales_json`, `scale_bound_flags_json`, `resol_clamp_count`,
  `resol_clamp_energy_low_kev`, `resol_clamp_energy_high_kev` (D-106).
* sim adds: `mode` (int), `mode_name`, `geometry_name`, `geometry_param_mm`,
  `angular_distribution`, `seed`, `n_events`, `workers`.
* compose adds nothing beyond the common fields.
* unfold adds: `mode` (str), `alpha`, `difference_order`, `energy_low_kev`,
  `energy_high_kev`, `channel_low`, `channel_high`, `pad_nsigma`, `syst_frac`,
  `chi2`, `dof`, `covariance_scale`; calib-only carries only `mode`.
* spectrum adds: `daq_time_s`, `source_file`.
* mc_spectrum adds: `source_key`, `mode_name`, `geometry_name`,
  `geometry_param_mm`, `n_events`, `seed`, `workers` (D-120).

Input sha256 digests live only in the common `inputs_json` provenance field
(D-89); the former `calib_sha256`/`sim_sha256` fields are removed. Changes to
this list require the contract-change process (AGENTS.md).

## 4. Mathematics and statistics

Formula IDs (`F-...`) are registered in `docs/derivations.md`; each ID will
carry the derivation, its approximations, and the certificate that guards it.

| ID | Content |
|----|---------|
| F-MODEL-1 | `E(ch) = c0 + k1 ch + (4k2 - 3k1 - k3)/(2 ch_max) ch^2 + 2(k1 - 2k2 + k3)/(3 ch_max^2) ch^3`; internal basis = value at 0 and slopes at 0, ch_max/2, ch_max |
| F-MODEL-2 | internal `(c0,k1,k2,k3)` to reported `(c0..c3)` map, its constant Jacobian and inverse (single source) |
| F-MODEL-3 | exact monotonicity certificate: `dE/dch` is quadratic, its minimum lies at an endpoint or the vertex |
| F-MODEL-4 | `sigma^2(t) = (1-t)^2 b0^2 + 2(1-t)t b1^2 + t^2 b2^2`, `t = max(E,0)/E_REF`, `E_REF = 2000 keV`; exact b-derivatives of variance and sigma |
| F-MODEL-5 | resolution positivity certificate on the export grid; strict raises, non-strict clamps to `SIGMA_FLOOR_KEV` (D-73) |
| F-BIN-1 | channel axis `-0.5 .. n-0.5`; variable energy axis in keV with strictly increasing edges |
| F-BIN-2 | parameter-independent evaluation geometry: full deposition axis x requested channel rows, no bin selection (D-79) |
| F-BIN-3 | working window `[chlo - pad, chhi + pad]`, pad in local resolution widths converted to channels; reported window `[chlo, chhi]` |
| F-BIN-4 | fixed source-mode Monte-Carlo axis `0..4096 keV / 4096 bins` (source-mode spectrum and fit-time `C_fit` only; the matrix primary axis is `C.y`, D-121 revised) |
| F-KERN-1 | `P(i|j) = Phi((e_{i+1}-c_j)/sigma_j) - Phi((e_i-c_j)/sigma_j)` exact bin integral |
| F-KERN-2 | smoothstep support taper (D-76) and exact column renormalization; empty columns are exactly zero |
| F-KERN-3 | sparse triple assembly with exact-zero pruning beyond `n_sigma sigma` |
| F-KERN-4 | unnormalized local kernel derivatives `dn/d(e_lo)`, `dn/d(e_hi)`, `dn/d sigma`, `dn/dc` (generated) |
| F-RESP-1 | `C[i,j]` assembly (sparse); every column sums to 1 or is exactly zero (D-79) |
| F-RESP-2 | `R = C . p_tilde . diag(eta) = C . G . diag(1/N)` (equivalent; the second form is implemented) |
| F-RESP-3 | full-axis composition then window slice; `channel_low/high` self-description; split certificates (D-82) |
| F-RESP-4 | chain assembly of `dC/dq` and `dR/dq`, including the global per-column renormalization sum (D-83) |
| F-PROJ-1 | exact bin-overlap projection weights (target must cover source; fail loud) |
| F-PROJ-2 | values `W v` and independent variances `W**2 Var`; identity for equal grids |
| F-CAL-1 | fit weights `var = max(stat,1) + (syst_frac*data)**2 + MC` |
| F-CAL-2 | per-dataset quadratic Bezier scale (free middle control abscissa `s0`, D-103) |
| F-CAL-3 | fit parameter start values and bounds (D-103) |
| F-CAL-4 | calibration prediction Jacobian and exact chi-square gradient (chain through F-RESP-4) |
| F-CAL-5 | calibration covariance: Fisher inverse, Schur scale marginalization, reported-basis transform, `s**2` (D-105) |
| F-SIM-1 | fixed per-column sampling and uniform energy draw |
| F-SIM-2 | binomial/multinomial variance and `fSumw2` |
| F-SIM-3 | detection efficiency `eta = column_sum/N = 1 - zero/N` |
| F-SIM-4 | plane/sphere source sampling (Lambertian) |
| F-SIM-5 | 10 us pulse merging |
| F-SIM-6 | physical boundary certificate: deposition-bin lower edge above a primary-column upper edge is exactly zero (strict) |
| F-SIM-7 | deterministic seed derivation for matrix columns and source event blocks; same seed + same worker partition reproduces bit-for-bit (D-123) |
| F-SOLVE-1 | Tikhonov `chi2 + alpha * ||D_tilde mu||^2`, `D_tilde = D . diag(sqrt(diag(R^T W R)))` (D-80); with the default SNIP mask `D' = diag(rho**0.5) D` (F-SOLVE-6/D-154) |
| F-SOLVE-2 | self-implemented active-set non-negative QP (dense/banded/sparse Cholesky) |
| F-SOLVE-3 | KKT certificate in units of the data-gradient scale (D-84) |
| F-SOLVE-4 | SNIP LLS baseline (transform, resolution-derived iteration count, `max(y,0)` handling) (D-155/D-157) |
| F-SOLVE-5 | resolution-matched peak significance and the fixed peak mask `W` (D-156) |
| F-SOLVE-6 | masked operator `D_tilde' = diag(rho**0.5) D . diag(sqrt(diag(A)))`, certificate and acceptance metrics (D-158/D-161) |
| F-COV-1 | Fisher information from the analytic Jacobian |
| F-COV-2 | `s**2 = chi2/dof` scaling (PDG convention); PD required, no pseudo-inverse; PSD certificate |
| F-COV-3 | optional profile-covariance diagnostic: `dchi2 = 1` widths with inner re-optimization, correlations from the numerical Hessian inverse |
| F-UNC-1 | statistical propagation with the same Hessian that solved the problem |
| F-UNC-2 | systematic propagation (calibration, simulation MC multinomial, data-side term) |
| F-UNC-3 | strict band decomposition `total**2 = stat**2 + syst**2` |
| F-UNF-1 | energy window to channel rows and reported primary bins, with F-BIN-3 padding |
| F-UNF-2 | data-side unfold fit weights `sigma_fit**2 = max(stat,1) + (syst_frac*data)**2` (D-112) |
| F-UNF-3 | exact-zero primary column pruning before the solve (D-110) |
| F-UNF-4 | unfold diagnostics: `chi2`, `dof = fit rows - active solution bins`, `covariance_scale = 1` |
| F-UNF-5 | full unfolding orchestration (compose, solve, bands, product) |
| F-UNF-6 | `calib_only` relabeling onto the channel-derived energy axis (D-113) |
| F-IO-1 | atomic write / reopen validation protocol |

Numerical decisions already frozen elsewhere in this document are normative;
derivations may add detail but must not contradict them.

## 5. Correctness machinery

1. **Single source.** Every formula, constant, axis convention and object name
   has one implementation. Mechanical checks (W1) fail the build when a
   duplicate formula body appears outside its owning module.
2. **Symbolic generation.** Kernels and derivatives are generated from sympy
   into `kc761/core/_gen/` with a header recording sympy version, formula ID
   and the generation command. Hand-written derivatives are forbidden.
3. **Runtime certificates (strict mode).** All certificate suites run under
   `--strict` / `KC761_STRICT=1` and abort on violation: KKT and
   complementarity, count conservation (`sum(counts) + zero = N_j`,
   `sum(N_j) = n_events`), column sums, efficiency bounds `[0,1]`, energy
   monotonicity, covariance PSD, resolution positivity, finiteness.
4. **Always-on validation.** Schema/format version, axes/units/shape checks and
   finiteness checks run in every mode and raise `SchemaError` /
   `ValidationError`.
5. **Fail loud.** No silent clamping, NaN substitution or fallback defaults.
   Any remaining numerically necessary clamp must be explicit, logged as a
   warning, and recorded in `meta`.

## 6. Engineering and quality gates

* Code: Python >= 3.12, `from __future__ import annotations`, type hints on
  public functions, line length 100.
* Lint: `ruff check .` must be clean (rule set in `ruff.toml`).
* Tests: `pytest` under `tests/`; hypothesis for property tests of invariants.
  No golden files, no baseline comparisons.
* CI: GitHub Actions runs ruff + tests on Python 3.12/3.13 without Geant4 or the
  ROOT executable.
* Logging and errors per D-67/D-68.
* Documentation: `README.md`, `docs/architecture.md`, `docs/formats.md`,
  `docs/derivations.md`, and this plan.

## 7. Deletion list (W7)

* `app/`, `kc761calib/`, `kc761sim/`, `kc761unfold/`, `kc761util/`.
* C++ sources and headers: `calib2root.cxx`, `sim2root.cxx`,
  `unfold2root.cxx`, `csv2root.cxx`, `subbkg.cxx`, `rootmacros.h`.
* `binexport.py`, `rootcxxfrontend.py`, `hadd.py` (replaced by uproot merge).
* `G4History.macro`, repository-root `sim_vis_output.root`.
* `work/` stale products and moved scratch/prototypes.
* The pre-rewrite design document (superseded by `docs/plan.md`).

## 8. Workstreams

| Phase | Deliverable | Exit criteria (self-certifying, no baselines) |
|-------|-------------|-----------------------------------------------|
| W0 | Contract layer: launcher, package skeleton, frozen signatures, error/logging/strict infrastructure, schema contracts, verified uproot helpers, synthetic fixtures, tooling, docs skeleton | `ruff` clean; `pytest --collect-only` passes; all stubs import with no Geant4/ROOT; uproot round-trip test passes; `--help` works for both entry points |
| W1 | `core` numerics with derivations and sympy-generated kernels | every formula has derivation + ID; strict certificates pass on synthetic inputs; no duplicated formula bodies |
| W2 | `schema` write/read/validate/provenance implementation | every product writes atomically, reopens, validates, and round-trips through its own reader |
| W3 | `calib`: datasets + Bezier scaling + single-stage fit + Fisher covariance + report | continuous chi2 surface; resolution positivity certificate; covariance single definition |
| W4 | `unfold`: compose, solver + KKT, strict uncertainty decomposition, bands | KKT/column-sum/conservation certificates; `total**2 = stat**2 + syst**2` identity |
| W5 | `sim`: geometry dataclass, parameterized source modes, no ntuple, memory-budgeted workers, uproot merge | per-column accounting and physics-boundary certificates; sampling derivation documented; fixed seed reproducible |
| W6 | `cli`: six subcommands wired to the libraries, strict csv2root parser, subbkg semantics, compose artifact, config-file mode (sim batch/calib/compose/unfold) and `examples/*.toml` | uniform logging/errors/exit codes; `--force`/resume semantics; config mode runs a multi-run sim batch and single-run calib/compose/unfold; runs on the provided CSV sample |
| W7 | docs final English pass, CI complete, legacy deletion | deletion list empty; repository contains only the new implementation, tests and docs |

## 9. Risks and required inputs

1. **Unknown geometry provenance.** Crystal dimensions, R4600 composition,
   shield geometry and sample densities keep their current values by decision
   D-34; the geometry dataclass must mark each value as measured/assumed.
2. **CSV sample provided.** The raw files are in `work/data/exp/2609a/`
   (`Channel,Count` header with a `#<...>` acquisition time); W6 finalizes the
   strict `csv2root` grammar, ranges and error policy (Appendix A item 8).
3. **Strict mode is opt-in.** Normal runs do not self-certify; CI runs the
   certificate suites in strict mode on synthetic fixtures.
4. **No packaging.** Import paths are guaranteed only by the launcher and the
   `pytest.ini` `pythonpath` setting; this is documented as the only supported
   usage.
5. **`kc761.py` versus the `kc761/` package.** The launcher executes as
   `__main__` and never imports itself; `python -m kc761` uses
   `kc761/__main__.py`. Tests import the package, not the launcher.

## 10. Acceptance (non-baseline)

A module is complete when, simultaneously:

1. its formulas are registered in `docs/derivations.md` and have no second
   implementation;
2. generated kernels/derivatives come from sympy output;
3. all strict-mode certificates pass;
4. invalid inputs produce classified errors with actionable messages;
5. `ruff` is clean and CI is green;
6. the replaced legacy implementation has been deleted (from W7 onward).

## Appendix A. Open contract points (undecided; ask, do not assume)

1. **Non-strict resolution negativity.** Resolved by D-73 (2026-09-10):
   non-strict clamps `sigma` to `SIGMA_FLOOR_KEV` with a warning and a record;
   strict still raises F-MODEL-5. Implementation mechanics (floor value, API)
   are W1 contract points.
2. **SNIP peak masking.** Re-opened and re-resolved by D-154..D-161
   (2026-09-10): a redesigned, default-on SNIP peak mask is part of the
   Tikhonov operator (F-SOLVE-4..6); D-74 is superseded. Full derivation and
   limits in `docs/derivations.md`.
3. **`calib --sim` naming.** Resolved by D-144 (2026-09-10): the option is
   `calib --mc` with no alias, and the `[[calib.datasets]]` key is `mc`;
   `compose`/`unfold` keep `--sim`/`sim` for the matrix simulation product.
4. **Plot CLI surface.** Resolved by D-142 (2026-09-10): a single `--no-plot`
   switch (plots on by default) for `calib` and `unfold`, `no_plot` as the
   config key; `compose`/`sim` produce no plots and redirecting plots is not
   supported.
5. **Optimizer controls.** D-47 fixes a single quasi-Newton stage; the CLI
   flags for iteration limits/tolerances are not decided (legacy had
   `--stage1-maxiter` / `--stage2-maxiter`).
6. **Source key registry.** Resolved by D-122 (2026-09-10): the seven legacy
   keys are frozen in the W5 registry with per-entry provenance; additions or
   renames require a new contract decision.
7. **Format version numbering.** Resolved by D-78 (2026-09-10): the new
   products start at `format_version = 1`, independent of the legacy
   calibration version 3.
8. **csv2root strict parser details.** The sample CSV is now provided
   (`work/data/exp/2609a/`); W6 finalizes the column semantics, header grammar and
   accepted ranges and records them in `docs/formats.md` under the
   contract-change process.

## Appendix B. W0 deliverables

* `kc761.py`, `kc761/__main__.py`, package skeleton with `core/`, `schema/`,
  `calib/`, `unfold/`, `sim/`, `cli/`.
* Frozen signatures with docstrings, formula IDs and `NotImplementedError`
  bodies for all `core` and `schema` contracts listed in the W0 brief.
* `errors.py` hierarchy and `runtime.py` (logging + strict-mode resolution).
* Six-subcommand CLI skeleton with uniform long/short option names.
* `schema/_uproot.py`: verified uproot helpers (TH1D/TH2D with variance, meta
  RNTuple) with the spike conclusions recorded in `docs/formats.md`.
* `tests/`: deterministic synthetic fixtures and contract/spike tests.
* `ruff.toml`, `pytest.ini`, CI workflow.
* `README.md`, `AGENTS.md`, `docs/{architecture,formats,derivations}.md`.
