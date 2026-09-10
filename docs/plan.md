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
| D-19 | Default outputs live in `out/<subcommand>/`; names encode the inputs and run parameters (e.g. `<calib>-<mode>-n<N>-s<SEED>.root`). |
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
| D-48 | Fit weights keep the current formula `var = max(stat, 1) + (syst_frac * data)**2 + MC` with `syst_frac = 0.10`; the formula and the floor are documented. |
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
| D-70 | Plots keep the current visual style; the plotting code is shared, not duplicated. |
| D-71 | No hard wall-clock performance target. The implementation must still apply reasonable optimizations (sparse-first, no dense `n**2` intermediates, no redundant recomputation). |
| D-72 | Legacy ROOT products are not read; the input CSV files will be provided by the user and re-imported through the new `csv2root` strict parser. |

### 1.6 W1 contract decisions (2026-09-10, user-approved)

| ID | Decision |
|----|----------|
| D-73 | Non-strict resolution negativity: when `sigma**2 < 0` on the export grid and strict mode is off, clamp `sigma` to the documented floor `SIGMA_FLOOR_KEV`, emit a warning and record it (W4 writes the record into `meta`). Strict mode still raises the F-MODEL-5 certificate error. Closes Appendix A item 1. |
| D-74 | The SNIP peak mask (`snip_iter`, `mask_z0`, `mask_floor`) is removed entirely; `D` is only the normalized finite-difference operator (F-SOLVE-1). Closes Appendix A item 2. |
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
  plotting/                   # shared style and panel helpers
  cli/                        # argparse tree and subcommand modules
tests/                        # auxiliary tests and deterministic fixtures
docs/                         # plan, architecture, formats, derivations
```

Dependency rules:

* `core` imports only the standard library, numpy, scipy (and numba/sympy in the
  generated kernel modules). It must not import `schema`, `calib`, `unfold`,
  `sim`, `plotting`, `cli`, uproot, matplotlib or Geant4.
* `schema` imports `core` only for type contracts where unavoidable; it owns
  uproot and file IO.
* `calib`, `unfold` and `sim` import `core` and `schema`.
* `cli` imports everything; `plotting` is dependency-light and shared.
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

Units live in key names (`energy_kev`, `daq_time_s`) and axis titles; matrix
axes follow D-20.

### 3.3 `meta` RNTuple fields (initial contract)

* Every product: `format_version` (int), `producer` (str), `created_utc`
  (str), `git_revision` (str), `git_dirty` (int), `python_version` (str),
  dependency versions, input paths and sha256, full CLI arguments.
* sim adds: `mode`, `mode_name`, `geometry_name`, `geometry_param`
  (with unit suffix), `angular_distribution`, `seed`, `n_events`, `workers`,
  source file sha256.
* unfold adds: `alpha`, `difference_order`, `energy_low_kev`,
  `energy_high_kev`, `channel_low`, `channel_high`, `pad_nsigma`, `syst_frac`,
  `chi2`, `dof`, `covariance_scale`, `calib_sha256`, `sim_sha256`.

W2 finalizes the exact branch list; changes require the contract-change
process (AGENTS.md).

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
| F-SOLVE-1 | Tikhonov `chi2 + alpha * ||D_tilde mu||^2`, `D_tilde = D . diag(sqrt(diag(R^T W R)))` (D-80); SNIP mask removed (D-74) |
| F-SOLVE-2 | self-implemented active-set non-negative QP (dense/banded/sparse Cholesky) |
| F-SOLVE-3 | KKT certificate in units of the data-gradient scale (D-84) |
| F-COV-1 | Fisher information from the analytic Jacobian |
| F-COV-2 | `s**2 = chi2/dof` scaling (PDG convention); PD required, no pseudo-inverse; PSD certificate |
| F-COV-3 | optional profile-covariance diagnostic: `dchi2 = 1` widths with inner re-optimization, correlations from the numerical Hessian inverse |
| F-UNC-1 | statistical propagation with the same Hessian that solved the problem |
| F-UNC-2 | systematic propagation (calibration, simulation MC multinomial, data-side term) |
| F-UNC-3 | strict band decomposition `total**2 = stat**2 + syst**2` |
| F-CAL-1 | fit weights `var = max(stat,1) + (syst_frac*data)**2 + MC` |
| F-CAL-2 | per-dataset quadratic Bezier scale |
| F-SIM-1 | fixed per-column sampling and uniform energy draw |
| F-SIM-2 | binomial/multinomial variance and `fSumw2` |
| F-SIM-3 | detection efficiency `eta = 1 - zero/N` |
| F-SIM-4 | plane/sphere source sampling (Lambertian) |
| F-SIM-5 | 10 us pulse merging |
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
* `out/` stale products and `tmp/` prototypes.
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
| W6 | `cli`: six subcommands, strict csv2root parser, subbkg semantics, compose artifact | uniform logging/errors/exit codes; `--force` semantics; runs on the provided CSV sample |
| W7 | docs final English pass, CI complete, legacy deletion | deletion list empty; repository contains only the new implementation, tests and docs |

## 9. Risks and required inputs

1. **Unknown geometry provenance.** Crystal dimensions, R4600 composition,
   shield geometry and sample densities keep their current values by decision
   D-34; the geometry dataclass must mark each value as measured/assumed.
2. **CSV format pending.** The strict `csv2root` parser cannot be finalized
   until the user provides the raw CSV files in `data/exp/`; W0 only freezes its
   CLI surface and error policy.
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
2. **SNIP peak masking.** Resolved by D-74 (2026-09-10): the mask is removed
   entirely; `D` is only the normalized finite-difference operator.
3. **`calib --sim` naming.** The data-fitting spectrum option is currently
   `--sim`, which collides conceptually with `unfold --sim` (the matrix-mode
   simulation file). Renaming (e.g. `--mc`) was proposed but not decided.
4. **Plot CLI surface.** D-70 fixes the visual style, but the flags for
   enabling/disabling or redirecting plots (`--no-plot`, output paths) are not
   decided; W0 exposes `--no-plot` provisionally and marks it here.
5. **Optimizer controls.** D-47 fixes a single quasi-Newton stage; the CLI
   flags for iteration limits/tolerances are not decided (legacy had
   `--stage1-maxiter` / `--stage2-maxiter`).
6. **Source key registry.** `kc761/sim/__init__.py` mirrors the seven legacy
   source keys so the CLI is stable; W5 owns the final physics registry and
   may add/rename entries via the contract-change process.
7. **Format version numbering.** Resolved by D-78 (2026-09-10): the new
   products start at `format_version = 1`, independent of the legacy
   calibration version 3.
8. **csv2root strict parser details.** Column semantics, header grammar and
   accepted ranges depend on the sample CSV the user will provide.

## Appendix B. W0 deliverables

* `kc761.py`, `kc761/__main__.py`, package skeleton with `core/`, `schema/`,
  `calib/`, `unfold/`, `sim/`, `plotting/`, `cli/`.
* Frozen signatures with docstrings, formula IDs and `NotImplementedError`
  bodies for all `core` and `schema` contracts listed in the W0 brief.
* `errors.py` hierarchy and `runtime.py` (logging + strict-mode resolution).
* Six-subcommand CLI skeleton with uniform long/short option names.
* `schema/_uproot.py`: verified uproot helpers (TH1D/TH2D with variance, meta
  RNTuple) with the spike conclusions recorded in `docs/formats.md`.
* `tests/`: deterministic synthetic fixtures and contract/spike tests.
* `ruff.toml`, `pytest.ini`, CI workflow.
* `README.md`, `AGENTS.md`, `docs/{architecture,formats,derivations}.md`.
