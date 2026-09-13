# Derivations and formula registry

Status: final. Every registered formula is implemented and guarded by the
runtime certificates in section 3; F-IO-1 is in section 4. IDs are append-only;
derivations may gain detail but must never contradict `docs/plan.md`.

## 1. Formula IDs

| ID | Statement | Owner module | Status |
|----|-----------|--------------|--------|
| F-MODEL-1 | `E(ch) = c0 + k1 ch + (4 k2 - 3 k1 - k3)/(2 ch_max) ch^2 + 2 (k1 - 2 k2 + k3)/(3 ch_max^2) ch^3` | `core/model.py` | implemented |
| F-MODEL-2 | internal `(c0,k1,k2,k3)` -> reported `(c0..c3)` map, its constant Jacobian and inverse | `core/model.py` | implemented |
| F-MODEL-3 | strict monotonicity of `E(ch)` on `[0, ch_max]` for the fitted bounds | `core/model.py` | implemented |
| F-MODEL-4 | `sigma^2(t) = (1-t)^2 b0^2 + 2 (1-t) t b1^2 + t^2 b2^2`, `t = max(E, 0)/E_REF`, `E_REF = 2000 keV`; derivative with respect to `(b0, b1, b2)` | `core/model.py` | implemented |
| F-MODEL-5 | resolution positivity certificate; strict raises, non-strict clamps to `SIGMA_FLOOR_KEV` (D-73) | `core/model.py` | implemented |
| F-BIN-1 | channel axis `-0.5 .. n-0.5`; variable energy axis in keV | `core/binning.py` | implemented |
| F-BIN-2 | parameter-independent evaluation geometry: full deposition axis x requested channel rows, no bin selection (D-79) | `core/binning.py` | implemented |
| F-BIN-3 | working window and pad from the local resolution width | retired by D-187 | retired |
| F-BIN-4 | fixed source-mode Monte-Carlo axis `0..4096 keV / 4096 bins` (source-mode spectrum and fit-time `C_fit` only; the matrix primary axis is `C.y`, D-121 revised) | `core/binning.py` | implemented |
| F-KERN-1 | exact Gaussian bin integral `Phi((e_{i+1}-c_j)/s_j) - Phi((e_i-c_j)/s_j)` | `core/kernel.py` | implemented |
| F-KERN-2 | smoothstep support taper (D-76) and exact column renormalization; empty columns are zero | `core/kernel.py` | implemented |
| F-KERN-3 | sparse triple assembly of C from the kernel | `core/kernel.py` | implemented |
| F-KERN-4 | unnormalized local kernel derivatives `dn/d(e_lo)`, `dn/d(e_hi)`, `dn/d sigma`, `dn/dc` (sympy-generated; quotient rule in F-RESP-4, D-83) | `core/kernel.py` | implemented |
| F-RESP-1 | `C[i,j]` assembly (sparse); every column sums to 1 or is exactly zero (D-79) | `core/response.py` | implemented |
| F-RESP-2 | `R = C . p_tilde . diag(eta) = C . G . diag(1/N)` (equivalent forms) | `core/response.py` | implemented |
| F-RESP-3 | full-axis composition followed by window slicing; split full-axis/sliced certificates (D-82) | `core/response.py` | implemented |
| F-RESP-4 | chain assembly of `dC/dq` and `dR/dq` from the model and kernel derivatives (sympy-generated) | `core/response.py` | implemented |
| F-PROJ-1 | rebinning/folding projection matrix | `core/projection.py` | implemented |
| F-PROJ-2 | variance propagation through the projection | `core/projection.py` | implemented |
| F-CAL-1 | fit weights `var = max(stat, 1) + (syst_frac * data)**2 + MC` and the folded-prediction MC term (D-102) | `calib/model.py` | implemented |
| F-CAL-2 | per-dataset quadratic Bezier scale model (free middle control abscissa `s0`, D-103), value and derivatives | `calib/scaling.py` | implemented |
| F-CAL-3 | fit parameter start values and bounds (specification reference, D-103) | `calib/model.py` | implemented |
| F-CAL-4 | calibration prediction Jacobian and exact chi-square gradient (chain through F-RESP-4) | `calib/model.py` | implemented |
| F-CAL-5 | calibration covariance: Fisher inverse, scale marginalization, reported-basis transform, `s**2` scaling (D-105) | `calib/covariance.py` | implemented |
| F-SIM-1 | fixed per-column sampling, uniform energy draw within the column, exact event accounting `sum(counts) + zero = N_j` | `sim/sources.py`, `sim/certificates.py` | implemented |
| F-SIM-2 | exact fixed-total variance in `fSumw2`: `N_j p (1-p)` (matrix) and `c (1 - c/P)` (source spectrum) | `sim/certificates.py` | implemented |
| F-SIM-3 | detection efficiency `eta_j = column_sum_j / N_j` (equivalently `1 - zero_j/N_j`) | `sim/certificates.py` | implemented |
| F-SIM-4 | plane and circumscribed-sphere source sampling (Lambertian) | `sim/generator.py` | implemented |
| F-SIM-5 | 10 us pulse merging for the source mode | `sim/actions.py` | implemented |
| F-SIM-6 | physical boundary certificate: deposition-bin lower edge above a primary-column upper edge is exactly zero | `sim/certificates.py` | implemented |
| F-SIM-7 | deterministic seed derivation: `column_seed(seed, column)` and `block_seed(seed, block)`; same seed + worker partition is bit-for-bit (D-123 revised) | `sim/generator.py` | implemented |
| F-SOLVE-1 | Tikhonov objective `chi2 + alpha * ||D_tilde mu||^2`, `D_tilde = D . diag(sqrt(diag(R^T W R)))`, dimensionless `alpha` (D-80); with the SNIP peak mask `D' = diag(rho**0.5) D` (F-SOLVE-6/D-154, supersedes D-74) | `core/solver.py` | implemented |
| F-SOLVE-2 | self-implemented banded Cholesky active-set non-negative QP | `core/solver.py` | implemented |
| F-SOLVE-3 | KKT certificate in units of the data-gradient scale (D-84) | `core/solver.py` | implemented |
| F-SOLVE-4 | SNIP LLS baseline on the measured spectrum: transform, resolution-derived iteration count at the caller's reference bin, `max(y, 0)` handling, edge-preserving ends (D-155/D-157/D-188; the iteration cap and the mask floor default to 32 and 0.01, D-191) | `core/solver.py` | implemented |
| F-SOLVE-5 | resolution-matched peak significance and the fixed diagonal peak mask `W` with a `protect_bins` width (D-156/D-188/D-191) | `core/solver.py` | implemented |
| F-SOLVE-6 | masked penalty operator `D_tilde' = diag(rho**0.5) D . diag(sqrt(diag(A)))` with `rho_r = min_j w_{r+j}`; the mask-equality certificate is implemented, the operator's symmetry/half-bandwidth hold by construction and the meta hashes are provenance only (D-158/D-161/D-188; `alpha` defaults to 1.0, D-191); the D-161 acceptance metrics are still pending | `core/solver.py` | implemented (acceptance metrics pending) |
| F-COV-1 | Fisher information from the analytic Jacobian | `core/covariance.py` | implemented |
| F-COV-2 | `s**2 = chi2/dof` scaling (PDG convention); PD required, no pseudo-inverse fallback (D-84) | `core/covariance.py` | implemented |
| F-COV-3 | optional profile-covariance diagnostic | `core/covariance.py` | implemented |
| F-UNC-1 | statistical covariance propagation through the reduced free-set system `H_FF**-1` (D-86) | `core/uncertainty.py` | implemented |
| F-UNC-2 | systematic propagation (calibration covariance, simulation MC, data-side term) | `core/uncertainty.py` | implemented |
| F-UNC-3 | strict band decomposition `total**2 = stat**2 + syst**2` | `core/uncertainty.py` | implemented |
| F-UNF-1 | energy window -> channel/primary selection from `E(ch)` at channel centers; the selection is the solve space (D-111/D-187) | `unfold/selection.py` | implemented |
| F-UNF-2 | unfold fit weights `sigma_fit**2 = max(stat, 1) + (syst_frac*data)**2`, data-side only (D-112) | `unfold/selection.py` | implemented |
| F-UNF-3 | exact-zero primary-column pruning and reduced non-negative solve (D-110) | `unfold/solve.py` | implemented |
| F-UNF-4 | unfold diagnostics: weighted chi2, `dof = n_fit_rows - n_active`, `covariance_scale = 1` (D-118) | `unfold/solve.py` | implemented |
| F-UNF-5 | refolded `R . mu` restricted to the reported channel window `[chlo, chhi]` (D-117) | `unfold/unfold.py` | implemented |
| F-UNF-6 | calib-only channel-to-energy relabeling on `C.y = E(i +- 1/2)` (D-113) | `unfold/unfold.py` | implemented |
| F-IO-1 | atomic write, reopen validation and provenance protocol | `schema/io.py` | implemented |
| F-SPEC-1 | spectrum sum: `values = v_a + v_b`, `variances = max(e_a, 1)**2 + max(e_b, 1)**2`, `daq_time_s = t_a + t_b` (D-185/D-186) | `spectra/combine.py` | implemented |
| F-SPEC-2 | DAQ-time scaled subtraction: `r = t_a / t_b`, `values = v_a - r v_b`, `variances = max(e_a, 1)**2 + r**2 max(e_b, 1)**2`, `daq_time_s = t_a` (D-185/D-186) | `spectra/combine.py` | implemented |

## 2. Sympy generation convention

1. Generated modules live in `kc761tool/core/_gen/` and begin with
   `# GENERATED by tools/generate_kernels.py -- do not edit`.
2. The generator header records the sympy version, the formula IDs produced and
   the exact command line.
3. Value and derivative always come from the same symbolic expression; a
   hand-written derivative of a registered formula is a contract violation.
4. The mechanical single-source check fails when a registered expression
   appears outside its `_gen` module or its owning implementation.

## 3. Runtime certificates (strict mode)

| Certificate | Formula ID | Check | Entry point |
|-------------|------------|-------|-------------|
| Resolution positivity | F-MODEL-5 | `sigma**2 >= -1e-9` on the export grid; strict raises, non-strict clamps (D-73) | `model.verify_resolution_positivity`, `model.resolution_sigma_kev(..., strict=)` |
| Energy monotonicity | F-MODEL-3 | `dE/dch > 0` exactly at the interval endpoints or the quadratic vertex | `model.verify_energy_monotonicity` |
| Kernel column sums | F-KERN-2 | every non-empty column sums to 1 within `1e-10`; empty columns are 0 | `kernel.verify_column_sums` |
| Response columns | F-RESP-1 | every C column sums to 1 or is exactly 0 | `response.verify_response_columns` |
| Composition identity | F-RESP-2 | R column sums equal the detected mass on reachable columns / `N_j`; `eta in [0, 1]` | `response.verify_composed_columns` |
| Slice integrity | F-RESP-3 | sliced row sums never exceed the full column sums; removed mass >= 0 | `response.verify_window_slice` |
| KKT | F-SOLVE-3 | `mu >= 0`, scaled reduced gradient and complementarity `<= KKT_TOL` | `solver.solve_nonnegative(strict=True)`, `solver.verify_kkt` |
| Covariance PSD | F-COV-1/F-COV-2 | symmetric, smallest eigenvalue `>= -tol` | `covariance.verify_covariance_psd` |
| Band decomposition | F-UNC-3 | `total**2 = stat**2 + syst**2` within `1e-9` relative | `uncertainty.verify_band_decomposition` |
| Event accounting | F-SIM-1 | per column `sum(counts) + zero = N_j`; `sum(N_j) = n_events` | `sim.certificates.verify_event_accounting` |
| Simulation variance | F-SIM-2 | stored `fSumw2` equals `N_j p (1-p)` (matrix) or `c (1-c/P)` (source spectrum) | `sim.certificates.verify_binomial_variance`, `sim.certificates.verify_source_spectrum_variance`, `schema.io` strict |
| Efficiency bounds | F-SIM-3 | derived `eta_j = column_sum_j / N_j` lies in `[0, 1]` | `sim.certificates.verify_efficiency` |
| Physical boundary | F-SIM-6 | G entries whose deposition-bin lower edge exceeds the primary-column upper edge are exactly 0 (strict only) | `sim.certificates.verify_physical_boundary` |
| Finiteness | all | no NaN/Inf; shape and unit checks run in every mode | `kc761tool.core._checks` |

## 4. Derivations

### F-MODEL-1 - internal energy basis

**Assumptions.** `E(ch)` is cubic on the acquisition range `[0, ch_max]`;
`ch_max` is a fixed acquisition constant. The fit coordinates are the value
at the origin and the slopes at three nodes:
`c0 = E(0)`, `k1 = E'(0)`, `k2 = E'(ch_max/2)`, `k3 = E'(ch_max)`.

**Derivation.** Writing `E = a0 + a1 ch + a2 ch^2 + a3 ch^3` gives `a0 = c0`,
`a1 = k1`; the two slope conditions are
`a2 h + (3/4) a3 h^2 = k2 - k1` and `2 a2 h + 3 a3 h^2 = k3 - k1` with
`h = ch_max`. Solving the 2x2 system yields
`a2 = (4 k2 - 3 k1 - k3) / (2 h)` and `a3 = 2 (k1 - 2 k2 + k3) / (3 h^2)`.

**Discretization error.** None: the expression is exact for the cubic model;
only floating-point round-off (a few `eps`) enters.

**Limits.** `ch = 0` returns `c0` exactly. Values outside `[0, ch_max]` are
defined by the same polynomial but are outside the domain protected by
F-MODEL-3; callers only evaluate inside the acquisition range.

**Production.** Generated `model_expr.energy_kev`; expanded polynomial, no
loops, scalar parameters broadcast against array channels.

### F-MODEL-2 - internal/reported basis map

**Derivation.** The plain cubic coefficients are `(c0, c1, c2, c3) =
(c0, k1, a2, a3)`. The inverse is obtained from the slope conditions:
`k1 = c1`, `k2 = c1 + h c2 + (3/4) h^2 c3`, `k3 = c1 + 2 h c2 + 3 h^2 c3`.
The map is affine in the parameters, so the Jacobian
`d(reported)/d(internal)` is constant in the parameters (rows
`[1,0,0,0]`, `[0,1,0,0]`,
`[0,-3/(2h), 2/h, -1/(2h)]`,
`[0, 2/(3h^2), -4/(3h^2), 2/(3h^2)]`).

**Numerical strategy.** One generated matrix and one generated forward and
inverse map; round-tripping is part of the auxiliary tests.

### F-MODEL-3 - monotonicity certificate

**Derivation.** `E'` is a quadratic polynomial, so its minimum on `[0, h]`
lies at an endpoint or at the vertex. The certificate evaluates `E'` at
`0`, `h/2`, `h`, reconstructs the quadratic coefficients exactly by finite
differences, and checks the vertex when it is interior. This is exact for
the model - a sampling grid could in principle hide a narrow dip.

**Limits.** A zero slope fails strict positivity: a plateau is not
invertible, so the channel-energy map would not be one-to-one.

### F-MODEL-4 - resolution model

**Assumptions.** The energy resolution is a quadratic Bernstein form in
`t = max(E, 0) / E_REF` (`E_REF = 2000 keV`) with control values
`(b0^2, b1^2, b2^2) >= 0`. `b_k` may have either sign; only its square enters.

**Derivation.** `sigma^2(t) = (1-t)^2 b0^2 + 2 (1-t) t b1^2 + t^2 b2^2`.
For `t in [0, 1]` this is a convex combination of the non-negative controls,
so `sigma^2 >= 0`; the endpoints give `sigma^2(0) = b0^2` and
`sigma^2(E_REF) = b2^2`. The parameter derivatives are
`d(sigma^2)/d b_k = 2 b_k B_k(t)` with `B = ((1-t)^2, 2(1-t)t, t^2)` and
`d sigma / d b_k = b_k B_k / sigma`.

**Limits.** `E <= 0` clamps `t` to zero (`max(E, 0)`), i.e. the low-energy
resolution is flat at `b0`. For `E > E_REF` the form is extrapolated; the
cross term `2(1-t)t b1^2` becomes negative for `t > 1`, so `sigma^2` can
turn negative at high energy. That is exactly what F-MODEL-5 guards. For
`sigma = 0` the derivative is undefined; the degenerate branch is handled
explicitly by the clamp policy.

**Production.** Sympy-generated variance, sigma and their b-derivatives. The
clamped branch is evaluated by boolean masking so no NaN is produced inside
`sqrt`.

### F-MODEL-5 - resolution positivity certificate and clamp

Strict mode raises `CertificateError("F-MODEL-5")` when
`sigma^2 < -SIGMA_POSITIVITY_TOL_KEV2` (`1e-9`) on the export grid. Outside
strict mode a non-positive variance is clamped to
`SIGMA_FLOOR_KEV = 1e-3 keV`, a `RuntimeWarning` is emitted and
`resolution_clamped_mask` lets the caller record the affected energies in `meta`
(D-73). Values in `[-tol, 0]` are treated as zero-width in strict mode and
as clamped outside it.

### F-BIN-1 - channel and energy grids

The channel axis is uniform with edges `-0.5 .. n - 0.5`; bin centers are
`0 .. n-1` and every channel has unit width. Energy axes are variable,
strictly increasing, finite, in keV; centers are midpoints and widths are
edge differences. Validation is always on.

### F-BIN-2 - parameter-independent evaluation geometry (D-79)

**Design.** The response is evaluated over the **full** deposition axis and
the requested channel rows. There is no support-based bin selection and no
frozen bin subset.

**Why.** A parameter-independent selection would have to be a superset of
the contributing bins for every parameter in the fit box; for a wide box the
superset degenerates to the full axis while still adding index bookkeeping
and a discontinuity risk. The smoothstep taper is exactly zero for
`|x| >= n_sigma sigma` and all its first derivatives vanish there, so terms
outside the support can be pruned **exactly** for the current parameters
(F-KERN-3) without any freeze and without changing the value of the sum.
The objective is therefore continuous in the parameters by construction.

**Limits.** Deposition bins whose support never intersects the detector
range produce exactly zero columns (F-RESP-1); this is a physical statement
(no channel can see those depositions), not an approximation.

### F-BIN-3 - working window and pad (retired by D-187)

**Former definition.** The solver modeled `[chlo - pad, chhi + pad]` and
reported `[chlo, chhi]` (D-44), where
`pad = ceil(pad_nsigma * sigma(E_edge) / width_edge)` is the number of channels
covering the local resolution at each window edge. `core.binning.working_window`
and the `pad_nsigma` setting are removed.

**Why it was retired.** The padded rows are the only data that constrain the
primary columns just outside the reported window, so dropping them costs edge
accuracy (see F-UNF-1). Keeping them is only sound while they lie inside the
region where the composed response describes the measurement. On the Ra-226
validation dataset that condition fails below ~30 keV: the padded rows (9-30
keV) are reproduced by the model two orders of magnitude below the data, so
they dominate the weighted chi2 and the fit gives up the genuine 242/295/352 keV
lines. A region the analysis window excludes is therefore no longer fitted; a
padding-like structure is available by requesting a wider window and reporting
the interior.

### F-BIN-4 - fixed source-mode Monte-Carlo axis

**Definition.** `SOURCE_MODE_DEPOSITION_MAX_KEV = 4096.0` and
`SOURCE_MODE_DEPOSITION_BINS = 4096`, i.e. uniform edges `e_k = k` keV for `k = 0..4096`.
This is the frozen source-mode pulse-spectrum axis (D-33/D-120) and the
parameter-independent fit-time `C_fit` axis (D-101). It has a single implementation
(`core.binning.source_mode_deposition_edges_kev`): `calib.model` imports it and
re-exports the former `FIXED_DEPOSITION_*` aliases, and `sim` uses it for the
source-mode `mc_spectrum`. A column is active when its upper edge is `> 0`.

**Why uniform.** The fit-time response must not depend on the fitted parameters
(D-79), so its deposition axis is fixed. The matrix-mode primary and deposition axes
are the calibration product's channel-derived `C.y` (D-101/D-121 revised), so the
matrix `G` is square and does not use this fixed axis.

### F-KERN-1 - exact Gaussian bin integral

**Derivation.** For a source at `c_j` with width `sigma_j`, the probability
that the smeared energy falls in channel bin `i` is the Gaussian integral
over the bin edges `[e_i, e_{i+1}]`:
`P(i|j) = Phi((e_{i+1} - c_j)/sigma_j) - Phi((e_i - c_j)/sigma_j)`, with
`Phi` the standard normal CDF evaluated through `erf`. No midpoint
approximation is made.

**Discretization error.** None beyond floating point: the expression is
analytic. For `|z|` large the difference of CDFs underflows to zero, which
is the correct limit.

**Limits.** `sigma -> 0` degenerates to an indicator function; `sigma > 0`
is enforced by validation and by F-MODEL-5 in strict mode.

### F-KERN-2 - smoothstep taper and renormalization

**Derivation.** With `x` the bin-center offset from the source,
`s = clip((n_sigma sigma - |x|)/sigma, 0, 1)` and
`w = 3 s^2 - 2 s^3` (D-76). `w = 1` for `|x| <= (n_sigma - 1) sigma`,
decreases smoothly to 0 at `|x| = n_sigma sigma` and is exactly zero beyond.
The first derivative is `(6s - 6s^2) ds/dx`, which vanishes at both clip
boundaries, so `w` is C1 and the transition adds no kink to the objective.
The tapered, renormalized kernel is `p_ij = n_ij / D_j` with
`n_ij = P(i|j) w_ij` and `D_j = sum_i n_ij`; the column then sums to exactly
one. A column with `D_j = 0` is exactly zero (D-79).

**Limits.** The taper argument is the **bin-center** offset (D-83); the
support boundary is handled by the vanishing prefactor. `sigma` is bounded
below by `SIGMA_FLOOR_KEV` outside strict mode (D-73).

### F-KERN-3 - sparse assembly and exact-zero pruning

For column `j` only bin centers strictly inside
`(c_j - n_sigma sigma_j, c_j + n_sigma sigma_j)` are evaluated
(`searchsorted`); everywhere else the taper is exactly zero, so the pruned
sum is identical to the full sum for every parameter value. The pattern has
`O(sum_j reach_j)` entries with `reach_j ~ 2 n_sigma sigma_j / width`; no
dense `n_channels x n_deposition` intermediate is created. Zero-valued
entries are dropped; the assembled matrix is CSR.

### F-KERN-4 - local kernel derivatives

**Derivation.** With `n(e_lo, e_hi, c, sigma, center) = P w(center - c,
sigma)`, the generated module provides the partial derivatives
`dn/de_lo`, `dn/de_hi`, `dn/dc`, `dn/dsigma` and `dn/dcenter`. All factors
come from `sympy.diff`; the clipped smoothstep uses the chain rule with the
clipping prefactor, which vanishes outside the transition band, so
evaluating the derivative at the clipped coordinate is the correct C1
extension.

**Center folding.** The bin center is `(e_lo + e_hi)/2`, so a change of
either edge moves the taper argument by half the displacement; the kernel
adds `0.5 dn/dcenter` to `dn/de_lo` and `dn/de_hi` before returning.

**Renormalization.** The quotient rule couples every entry of a column
through `D_j`; it is assembled once in F-RESP-4 from these unnormalized
pieces and the per-column sums, never approximated locally.

### F-RESP-1 - deposition-to-channel matrix C

`C[i,j]` is the probability that a gamma depositing energy in deposition bin
`j` lands in channel bin `i`. Channel edges are `E(i - 1/2)` from F-MODEL-1;
deposition bin centers are the midpoints of the product axis; widths come
from F-MODEL-4. The matrix is sparse (F-KERN-3) and every column sums to 1
or is exactly 0; `deposition_edges_kev` is stored with the matrix.
`build_response_matrix` validates that the calibrated channel edges are
strictly increasing (the always-on consequence of F-MODEL-3).

**Limits.** A source whose entire support lies outside the channel range
gives an empty column; the F-RESP-1 certificate expects exactly 0 for it.

### F-RESP-2 - composition

**Derivation.** With `G` the deposition x primary count matrix, `S_j` its
column sums and `N_j` the per-column totals,
`p_tilde = G / S_j` and `eta_j = S_j / N_j`, so
`p_tilde . diag(eta) = G . diag(1/N)` and therefore
`R = C . p_tilde . diag(eta) = C . G . diag(1/N)`. The implementation uses
the second, equivalent form: it has no intermediate 0/0 and one fewer
normalization step. `R` column sums equal the detected mass that lands on
reachable C columns divided by `N_j`; `eta` is validated to lie in `[0, 1]`
(F-SIM-3) and detected counts above `N_j` are rejected.

**Limits.** `N_j = 0` yields a zero column; deposition mass on empty C
columns is dropped by construction (it is unobservable), which the
F-RESP-2 certificate accounts for explicitly.

### F-RESP-3 - full-axis composition then slicing

Composition runs over the full primary axis (D-43); slicing happens
afterward (D-44). `slice_response` selects only matrix rows and keeps
`column_sums` and `efficiency` as **full-axis** facts, while the new
`channel_low`/`channel_high` fields describe the rows the matrix holds.
Composing after slicing would renormalize the response with the wrong
window; the certificate verifies instead that slicing only removed
non-negative row mass and that the sliced row sums never exceed the full
column sums.

### F-RESP-4 - response parameter Jacobian

**Derivation.** For `q = (c0..c3, b0..b2)` in the reported basis and pair
`(i, j)`, the local derivative through the channel edges is
`u_ij = (dn/de_lo)(de_i/dq_k) + (dn/de_hi)(de_{i+1}/dq_k)` with
`de_l/dq_k = ch_l^k` for the reported calibration parameters (F-MODEL-2);
through the resolution it is `u_ij = (dn/dsigma_j)(dsigma_j/db_k)`
(F-MODEL-4). The normalized derivative follows from the quotient rule with
the global column sums:
`dp_ij/dq_k = u_ij/D_j - n_ij (sum_i' u_i'j)/D_j^2`.
Every column of `dC/dq` therefore sums to zero, which the auxiliary tests
check. The composed Jacobian is `dR/dq_k = (dC/dq_k) . P` with
`P = G . diag(1/N)`.

**Production.** The per-column sums use `numpy.bincount` on the CSR
pattern; seven sparse matrices are returned, never a dense
`n_channels x n_deposition x 7` tensor.

### F-PROJ-1 - overlap projection

**Derivation.** Split the union of the source and target edge arrays into
segments; every segment lies inside exactly one source and one target bin.
`W[t, s]` is the segment length in target bin `t` divided by the source bin
width, so each source bin is distributed exactly. `build_projection_plan`
requires full coverage of every source bin (within `1e-9`) and fails loudly
otherwise.

**Production.** `numpy.union1d` + `searchsorted` over segment midpoints;
sparse CSR weights.

### F-PROJ-2 - projection of values and variances

Values project as `W v` (mass conserving); independent variances project as
`W**2 Var`. For identical binnings `W = I`, so the projection is idempotent.
Covariances between source bins are not modeled (documented limitation).

### F-CAL-1 - calibration fit weights and folded-prediction MC variance

**Model (D-100).** For dataset `d`, with the shared internal core
`q = (c0, k1, k2, k3, b0, b1, b2)`, the deposition-to-channel matrix
`C_fit(q)` and the per-dataset scale `s_d(ch)` (F-CAL-2),

    prediction_d = s_d . (C_fit(q) @ mc_d),
    data_d ~= prediction_d + noise.

`C_fit` is evaluated on the fixed uniform deposition axis `0..4096 keV /
4096 bins` (D-33/D-101); the fit only contracts it with `mc_d` and its
variance, so the implementation builds it on the contiguous hull of the
non-zero MC bins, which is exactly the corresponding sub-matrix of the fixed
axis (each column's kernel and renormalization depend on that column alone).

**Weights (D-48).** With `stat` the data variance `fSumw2`, `syst_frac` the
per-dataset fractional systematic and `MC` the folded prediction's MC
variance,

    var_d = max(stat_d, 1) + (syst_frac_d * data_d)**2 + MC_d,
    MC_d = (s_d * sqrt((C_fit**2) @ var_mc_d))**2.

The `max(stat, 1)` floor is the frozen weight convention: bins with fewer than
one unit of variance are treated as unit variance. With no MC `fSumw2` the
source variance falls back to `max(mc, 0)` (Poisson).

**Limitations.** The floor is a documented approximation: for Poisson data
with `pred < 1` the realized `(data - pred)**2 / max(data, 1)` has expectation
below `pred`, so `chi2/dof` can sit below one on low-count spectra. The
covariance is defined for the weights actually used (F-CAL-5).

### F-CAL-2 - per-dataset quadratic Bezier scale

With `x` the channel and `[x_lo, x_hi]` the fit window, the scale is the
quadratic Bezier with control abscissae `(x_lo, s0, x_hi)` and ordinates
`(s1, s2, s3)`:

    x(t) = x_lo + 2 (s0 - x_lo) t + (x_lo - 2 s0 + x_hi) t^2,
    s(t) = (1-t)^2 s1 + 2 (1-t) t s2 + t^2 s3,
    t(x) = d / (a + sqrt(a^2 + c d)),  a = s0 - x_lo, c = x_lo - 2 s0 + x_hi, d = x - x_lo.

`x(t)` is strictly increasing for `s0` strictly inside the window, so `t` is
the unique in-interval root; the rationalized form is exact at both endpoints
and free of cancellation.

**Derivatives.** `dx/dt = 2(s0 - x_lo) + 2(x_lo - 2 s0 + x_hi) t`,
`ds/dt = 2(s2 - s1) + 2(s1 - 2 s2 + s3) t`, and

    ds/ds0 = (ds/dt) * (-2 t (1-t) / (dx/dt)),
    ds/ds1 = (1-t)^2,  ds/ds2 = 2(1-t)t,  ds/ds3 = t^2.

**Parameter structure (D-103).** For a constant scale (`s1 = s2 = s3`) the
derivative with respect to `s0` vanishes identically; at
`s0 = (x_lo + x_hi)/2` the abscissa parametrization becomes linear and the
four-parameter family collapses onto the three-parameter quadratic-polynomial
subfamily (the degree-2 Bernstein basis), so the `s0` direction is an exact
gauge. For any other `s0` the family is a genuine parabola and the four
parameters are locally independent (rank 4). `s0` is therefore kept free, and
the model is seeded with a non-constant scale so the fit does not start on the
gauge plateau (F-CAL-3); the scale block is marginalized stably in F-CAL-5.

### F-CAL-3 - fit start values and bounds

The internal core starts at `(-180, 1.5, 2.5, 3.5, 2, 20, 40)` with bounds
`c0 in (-300, -100)`, `k1 in (1, 2)`, `k2 in (2, 3)`, `k3 in (3, 4)`,
`b0 in (0, 10)`, `b1 in (0, 80)`, `b2 in (0, 100)`; these are the frozen ranges
(specification reference, D-103) and keep `E(ch)` monotone (F-MODEL-3).
The scale values are bounded to `[0.01, 3]` times the dataset's overall
normalization `sum(data) / sum(model)`; `s0` is bounded strictly inside the
window (F-CAL-2). A constant scale makes `ds/ds0` vanish, so the seed is
non-constant: the data/model ratio is averaged over the lower, middle and
upper thirds of the window (weighted by `model**2 / var`) to give `(s1, s2,
s3)`, with `s0` at the midpoint. The fit then moves `s0` along the genuine
parabola family.

### F-CAL-4 - calibration prediction Jacobian and gradient

Differentiating F-CAL-1, with `C_k = dC/dq_k` from F-RESP-4 (chained into the
internal basis through the F-MODEL-2 Jacobian) and `d s / d s_p` from
F-CAL-2:

    d prediction_d / d q_k = s_d . (C_k[window] @ mc_d),
    d prediction_d / d s_p = (d s_d / d s_p) . (C_fit[window] @ mc_d).

The exact chi-square gradient (the variance depends on the parameters) is

    d chi2 / d theta = -2 J^T ((data - p)/v) - (dv/dtheta)^T ((data - p)^2 / v^2),
    dv/dq_k = s_d^2 * [2 (C_fit . C_k)[window] @ var_mc_d],
    dv/ds_p = 2 s_d (d s_d/d s_p) * [(C_fit^2)[window] @ var_mc_d].

All derivatives are analytic; no finite differences enter production. The
optimizer uses the residual `r = (data - p)/sigma` and its Jacobian
`dr/dtheta = -J/sigma - r * (dv/dtheta)/(2 v)`, so its gradient is the exact
gradient of the objective it minimizes.

### F-CAL-5 - calibration covariance extraction

With the full-parameter prediction Jacobian `J` (core and scale columns,
F-CAL-4) and `W = diag(1/v)` (F-CAL-1), let `F = J^T W J` and split its
parameters into the reported core `c = (c0..c3, b0..b2)` and the scale `s`:

    cov_core = (chi2/dof) * (F_cc - F_cs F_ss^+ F_sc)^-1,
    cov_reported = T cov_core T^T,  T = diag(internal_jacobian(channel_max), I_3).

The Schur complement is the `(c, c)` block of `F^-1` (the scale marginalized);
`F_ss^+` is the Moore-Penrose inverse of the scale block, which projects out
the `s0` gauge when the fitted scale is (nearly) polynomial. Jacobi
(diagonal) preconditioning is applied before the factorization. `chi2/dof` is
the single global scale (F-COV-2). A non-positive-definite core Schur
complement is a hard failure (no pseudo-inverse fallback for the core); the
estimator string recorded with the product is
`fisher-x2dof-marginalized-reported`. For an invertible `F_ss` this is
algebraically identical to taking the core block of the full inverse, which is
verified in the tests with a well-conditioned Fisher.

### F-SIM-1 - fixed per-column sampling and event accounting

**Sampling.** The matrix primary axis has `n_active` active columns. `n_events` is split as
`base, rem = divmod(n_events, n_active)`; the active columns in ascending order receive
`base + 1` for the first `rem` and `base` otherwise. In column `j`, `N_j` events are generated;
each event's primary energy is `E = lo_j + u (hi_j - lo_j)` with `u` uniform in `[0, 1)` and
`[lo_j, hi_j] = [max(edge_j, 0), edge_{j+1}]`. This is exactly the round-robin
assignment `active[(offset + event) mod n_active]` as a count vector.

**Accounting.** Every event either deposits a positive total in the crystal (filling `G` in
exactly one (deposition, primary) cell) or deposits nothing (counted in `zero_j`). Hence
`sum_d G[d, j] + zero_j = N_j` and `sum_j N_j = n_events`. A positive deposit that left the
deposition axis would break the first identity rather than being silently lost, which is why
`verify_event_accounting` raises F-SIM-1.

### F-SIM-2 - exact fixed-total variance

**Matrix.** Conditional on the fixed column total `N_j`, the deposition-bin counts are
multinomial with probabilities `p_dj`; the binomial marginal is
`Var(G[d, j]) = N_j p (1 - p)` with `p = G[d, j] / N_j`, stored in `fSumw2`. Within a column the
bins are negatively correlated and the correlations are reconstructed downstream from counts and
`N_j`; a Poisson `counts` variance would overstate the high-probability bins.

**Source spectrum.** The source mode fills one entry per merged pulse, so the pulse total `P` is
itself random. Conditional on the recorded `P = sum(counts)`, the per-bin counts are multinomial
and the plug-in marginal is `Var(c) = c (1 - c / P)`. This is the D-127/D-128 convention; it is an
approximation to the full pulse-count distribution and is documented as such.

### F-SIM-3 - detection efficiency

`eta_j = sum_d G[d, j] / N_j`, with `eta_j = 0` for a zero-total column. Because every event is
either detected or counted as zero, this equals `1 - zero_j / N_j` and lies in `[0, 1]`; the
certificate checks the bound.

### F-SIM-4 - plane and sphere surface sampling

**Plane.** The position is uniform on the crystal-face-sized square (`x = (2 u_x - 1) h_x`,
`y = (2 u_y - 1) h_y`, `z = z_plane`). The direction is Lambertian about the inward normal
`-z`: `cos(theta) = sqrt(u_cos)`, `phi = 2 pi u_phi`, and
`direction = (-sin t cos phi, -sin t sin phi, -cos t)`, a unit vector.

**Sphere.** A uniform point on the sphere of radius `R` uses `cos(theta0) = 2 u1 - 1`,
`phi0 = 2 pi u2`, the inward normal `n = (-sin t0 cos phi0, -sin t0 sin phi0, -cos t0)` and
`position = -R n`. A local orthonormal frame `e1 = n x z_hat` (falling back to `(1, 0, 0)` at the
poles) and `e2 = n x e1` carries the same Lambertian direction
`e1 sin t cos phi + e2 sin t sin phi + n cos t`, a unit vector with positive projection on the
inward normal.

### F-SIM-5 - pulse merging

Crystal deposits carry their Geant4 global time. Sorting by time, a pulse is the group
`[t0, t0 + 10 us)`; its energy is the sum of the group's deposits, and groups are cut before the
first deposit outside the window (half-open). Only the source mode merges pulses; the matrix mode
scores one gamma per event.

### F-SIM-6 - physical boundary certificate

A gamma cannot deposit more energy than it carries, so on the binned axes a necessary condition
is `G[d, j] = 0` whenever `deposition_edges[d] > primary_edges[j+1]` (the deposition bin starts
above the primary column's top edge). The check is one vectorized mask and runs in strict mode; a
violation is `CertificateError("F-SIM-6")`. It complements the always-on F-SIM-1 accounting,
which catches deposits that leave the deposition axis entirely.

### F-SIM-7 - deterministic seed derivation

Each matrix primary column has an independent stream seeded by `column_seed(seed, j)`; each
source-mode event block by `block_seed(seed, b)`. Both use a SplitMix64 mixer of `(seed + tag)`
and the index, reduced to `[0, 2**63)`:

    def _stream_seed(base_seed, tag, index):
        mixed = splitmix64((base_seed + tag) & (2**64 - 1))
        mixed = splitmix64((mixed + index) & (2**64 - 1))
        return mixed % (2**63)

with `tag = 0x01` for columns and `0x02` for blocks. For a fixed seed and a fixed worker
partition, merging the summed histograms reproduces the invocation result bit-for-bit (counts are
integer-valued). Source-mode blocks have the fixed size `SOURCE_MODE_EVENT_BLOCK` and are never
split across workers.

**Revision (D-123).** Bit-for-bit equivalence across *different* worker counts is not a
contract: Geant4's engine state (including internal generator caches) makes a mid-run reseed
process-dependent. The matrix per-column streams do not depend on the partition, and the matrix
merged result is worker-count independent in practice (covered by a test), but only the
same-seed/same-partition reproducibility is normative.

### F-SOLVE-1 - Tikhonov objective and its normalization

**Objective.** Minimize `chi2 + alpha * ||D_tilde mu||^2` over `mu >= 0`,
with `chi2 = ||(R mu - y)/sigma||^2`, `W = diag(1/sigma^2)`,
`A = R^T W R`, `b = R^T W y`.

**Normalization (D-80).** `D` is the order-1 `[-1, 1]` or order-2
`[1, -2, 1]` finite-difference operator and
`D_tilde = D . diag(sqrt(diag(A)))`. Since `A` carries `1/counts^2`, the
diagonal scale is `1/counts`; `D_tilde mu` is dimensionless and `alpha` is
dimensionless and comparable across datasets and windows. Zero-curvature
columns (`A_jj = 0`) have a zero scale and are dropped from the penalty;
they carry no data information and the solver fixes them at zero. With the
(default-on) SNIP peak mask the penalty operator becomes
`D_tilde' = diag(rho**0.5) D . diag(sqrt(diag(A)))` (F-SOLVE-6); D-74 is
superseded by D-154.

**Half-gradient.** The half-gradient is `g(mu) = H mu - b` with
`H = A + alpha D_tilde^T D_tilde`; the solver and F-UNC use this `H`
object, so value and error cannot drift apart.

**Discretization.** `D` is a discrete roughness measure; no continuum
approximation is claimed beyond the chosen stencil order.

### F-SOLVE-2 - active-set non-negative QP

Lawson-Hanson active set on `min 1/2 mu^T H mu - b^T mu`, `mu >= 0`.
Zero-curvature, zero-gradient bins are fixed at zero; a zero-curvature bin
with a non-zero gradient is an unbounded problem and raises `SolverError`.
Reduced systems use dense Cholesky for small or dense matrices, banded
Cholesky when the half-bandwidth is below `n/4`, and sparse LU otherwise.
The iteration budget is `10 n + 100`; exhaustion raises `SolverError` in
every mode. Starting point is `mu = 0` (no warm start yet).

### F-SOLVE-3 - KKT certificate

For `r = H mu - b`, the reported metrics are
`max(0, -min_{active} r) / max(1, ||b||_inf)` and
`max_i |mu_i r_i| / (max(1, ||b||_inf) max(1, ||mu||_inf))`, both required
`<= KKT_TOL = 1e-6`. Dividing by the data-gradient scale makes the
certificate independent of the count normalization of the problem. Strict
mode raises `CertificateError("F-SOLVE-3")` on failure.

### F-SOLVE-4 - SNIP baseline estimation (D-155/D-157/D-188)

**Role.** The SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
baseline is used *only* to locate genuine peaks in the measured spectrum. The
mask built from it (F-SOLVE-5) relaxes the roughness penalty at those peaks,
which widens the admissible range of `alpha`: a larger global `alpha` can
suppress noise-induced (spurious) oscillatory structure without eroding real
peaks. SNIP does not identify spurious peaks; it identifies real ones.

**Non-negativity.** Background-subtracted spectra may contain negative bins.
The estimator operates on `y+ = max(y, 0)`; the number and index range of
clipped bins are recorded (`snip_clipped_bins`, `snip_clipped_index_range`).
No shift or other imputation is applied (no silent handling).

**LLS transform.** With `y = y+`,
`v_i = ln( ln( sqrt(y_i + 1) + 1 ) + 1 )` and inverse
`y_i = ( exp( exp(v_i) - 1 ) - 1 )**2 - 1` (Morháč's log-log-sqrt transform).
The transform compresses the dynamic range so clipping acts on relative
structure.

**Iteration.** For `p = 1 .. m`:
`v_i <- min( v_i, (v_{i-p} + v_{i+p}) / 2 )` for interior bins only. A bin whose
`i - p` or `i + p` neighbor does not exist keeps its value (edge-preserving): an
average against a missing neighbor would pull a boundary value down toward its
in-range neighbor. The baseline is `b = inverse(v)` after the last iteration.

**Iteration count (D-157/D-188).** `m` is resolution-derived, not a free knob. At
the reported-window midpoint energy `E_mid`, `FWHM_bins = FWHM(E_mid) / Delta_E`
with `FWHM = 2 sqrt(2 ln 2) sigma_E(E_mid)` (F-MODEL-4); then
`m = clip(round(FWHM_bins / 2), 1, m_max)`, with `m_max` a documented safety
cap. The reference bin is passed in by the caller
(`iteration_reference_index`, the bin whose center is nearest the reported
window midpoint in the shipped pipeline); `None` keeps the middle bin of the
array handed to the mask, which is what the unit tests use. An explicit override
(`snip_iterations`) is allowed and recorded.

**Limits of the derived `m`.** `m_max = 32` (D-191; it was 8 under D-162)
saturates the rule only when `round(FWHM_bins / 2) > m_max`, i.e. when
`sigma_bins > (m_max + 0.5) / sqrt(2 ln 2)` = 27.6 bins at the reference bin
(the strict inequality follows from `int(round(...))` with round-half-to-even:
at the exact tie the even-valued cap does not bind),
which is above roughly 3.5 MeV on the production 2048-bin axis with the shipped
resolution model. The derived count is therefore used un-clipped throughout a
30-3000 keV window: `m` is 3 at 30 keV, 6 at 150 keV, 13 at the 609 keV line,
17 at 1 MeV and 21 at that window's midpoint (the realized value recorded by the
shipped Th-232 run), and the clipping window `2m+1` then matches the peak's own
`FWHM_bins` instead of stopping inside it, which is the intended "remove the
detector peak before estimating the continuum" behavior. Under the former
`m_max = 8` the rule saturated above roughly 260 keV: SNIP removed only
structures narrower than `2m+1 = 17` bins while the peak itself is 18-47 bins
wide between 300 keV and 1.8 MeV, so the baseline sat *inside* the peak
(measured: 59% of the 609 keV peak top on the validation dataset). Either way
the mask is a peak-*locator*, not a background estimate; the significance map is
a broad pedestal rather than a sharp peak indicator, which is acceptable because
the protection width is fixed in bins (F-SOLVE-5).

**Limits.** SNIP removes structures narrower than about `2m+1` bins; tying `m`
to the resolution width removes the detector peak before estimating the
continuum, which is the intended behavior. On very low-count bins the
transform is near-linear and the baseline is stable. No continuum model is
claimed: `b` is a robust local lower envelope, and the mask derived from it is
a *structural prior*, not a background measurement.

### F-SOLVE-5 - Resolution-matched significance and peak mask (D-156/D-188)

**Residual.** `r_i = y+_i - b_i`.

**Matched filter.** The detector width `sigma_E(E_i)` is known (F-MODEL-4).
With bin width `Delta_i` and `s_i = sigma_E(E_i) / Delta_i`, a normalized
Gaussian kernel `g` of width `s_i` is applied around each bin:
`M_i = sum_k g_k r_{i+k}` and `V_i = sum_k g_k**2 sigma_{y,i+k}**2`, so
`z_i = M_i / sqrt(V_i)`. The kernel is truncated at `SNIP_FILTER_SIGMA = 3`
resolution widths (`|k| <= ceil(3 s_i)`, renormalized over the truncated
support), which is the matched-filter support and the only place the truncation
enters. The matched filter suppresses single-bin noise spikes
(the dominant spurious-peak seed) that a per-bin threshold would misclassify.

**Candidates.** Bin `i` is a peak candidate when `z_i >= k` (default
`k = 5`, D-156) and `z_i` is a local maximum against its **two immediate
neighbors** (the filter window is not used for the local-maximum test).

**Mask (revised by D-188).** Around every candidate center, all bins within
`protect_bins` **primary bins** (default `3`, the order-2 stencil width) are
marked as peaks; overlapping intervals merge into one cluster, so the protected
count is bounded by `(2 protect_bins + 1) * n_candidates`. The fixed diagonal
mask is `W = diag(w)` with `w_i = floor` (default `0.01`, D-191) on marked bins and
`w_i = 1` elsewhere. A zero-variance or non-positive-width bin is a hard
`ValidationError` rather than an unmarked bin, so the only bins that keep
`w_i = 1` are those no candidate marks.

**Why the width is in bins, not resolution widths.** The protection must cover
the *unfolded* peak core, whose width is set by the primary binning, not by the
detector: a line occupies one primary bin, and the order-2 penalty smooths over
three. A resolution-scaled width describes the *measured* peak and grows with
the detector width in bins (`sigma_E / Delta_E` is 5.5 bins at 150 keV and 20
bins at 1.8 MeV on the validation dataset), so it stopped being local: the old
rule marked 46.8% of a 2048-bin axis (80% of 130-650 keV, runs up to 96 bins),
and the roughness penalty vanished over whole regions. The resolution still
enters where it belongs, in the matched filter above. A capped resolution rule
was rejected as well: on this deployment (`sigma_E / Delta_E` >= 5.5 bins) the
cap would always bind, so `protect_sigma` would become a setting without an
effect, while on a deployment with `sigma_E / Delta_E <= 1.5` bins the detector
no longer resolves the peak and a bin-scale protection is the only meaningful
one. One knob with a bin-scale meaning replaces a knob whose meaning changed
with the detector.

**Limits and multiplicity.** Thresholding many bins inflates the family-wise
false-positive rate; the 5-sigma threshold plus the resolution-width matched
filter keeps it small but not zero. A wrongly protected noise spike is *less*
smoothed than before; that is the known failure mode and is quantified by the
acceptance metrics (F-SOLVE-6). The mask is a function of `y` only and is
frozen before the solve (D-155), so the masked problem stays convex.

**Measured effect of the redesign (D-188).** On the Ra-226 validation dataset
(`alpha = 10`, mask on vs off, raised from the same fit space): protected bins
126/2048 = 6.2%, single maximum per line, peak-height inflation 1.20-1.23x, and
chi2 276.5 vs 308.8. The former rule on the same data set produced a four-spike
comb around 609 keV (589/606/618/641 keV, single-bin heights up to 1.32e6 vs a
2.76e5 single peak with the mask off in the same padded fit space, i.e. 4.8x;
the same rule inflated the single-bin height to 2.3x in the reported-window fit
space, 1.0e6 against 4.4e5), split 186/242 keV peaks, and halved the
number of non-zero bins at constant total strength -- the degenerate-vertex
signature of a locally vanishing penalty.

### F-SOLVE-6 - Masked operator, objective and certificates (D-154/D-158/D-188)

**Masked difference operator.** `D' = diag(rho**0.5) D` with `D` the order-1
`[-1, 1]` or order-2 `[1, -2, 1]` finite-difference operator (F-SOLVE-1) and
`rho_r = min_{j=0}^{order} w_{r+j}` the **minimum** of the mask weights over the
stencil of difference row `r` (D-188). A protected peak bin therefore relaxes
every difference row that touches it, and each such row carries exactly the
configured floor: `rho_r = floor` for any row whose stencil contains a protected
bin, `rho_r = 1` otherwise. The former product rule reached
`rho = floor**(order+1) = 1e-3` for interior rows, i.e. a relaxation 100x
stronger than the value the setting and its CLI help name, which made the masked
normal matrix locally singular (a family of near-optimal solutions, of which the
active set returns an arbitrary vertex). `D'^T D' = D^T diag(rho) D` stays
symmetric positive semidefinite and banded (same half-bandwidth as `D`). (The
rectangular `W**0.5 D W**0.5` form does not typecheck for `n_rows = n - order`;
the row form is the correct symmetric weighting.)

**Normalization (D-80 carried through).** `D_tilde' = D' .
diag(sqrt(diag(A)))` with `A = R^T W_data R`, `W_data = diag(1/sigma**2)`;
`alpha` stays dimensionless. Zero-curvature columns (`A_jj = 0`) are dropped
exactly as in F-SOLVE-1.

**Objective and normal equations.**
`min_{mu >= 0} ||(R mu - y)/sigma||^2 + alpha * ||D_tilde' mu||^2`,
`H = A + alpha * D_tilde'^T D_tilde'`, `b = R^T W_data y`. The Lawson-Hanson
active-set solve and the F-SOLVE-3 KKT certificate are unchanged because
`D_tilde'` is fixed before solving. `alpha` is optional and defaults to
`DEFAULT_ALPHA = 1.0` (D-191); the mask only enlarges the useful `alpha` range,
it does not choose `alpha`.

**Uncertainty.** F-UNC-1/F-UNC-2 use the same `H`, so the bands include the
mask effect automatically. The mask is data-derived (plug-in): the reported
covariance is conditional on the realized mask, and the mask-selection
uncertainty is not propagated (D-159). That conditionality is documented and
its coverage is validated on synthetic pulls with the mask on and off.

**Certificate F-SOLVE-6 (implemented scope).** Strict mode recomputes the mask
from the recorded spectrum, the settings and the `iteration_reference_index`
(which the unfold layer derives from the reported-window midpoint), requires the
stored weights to equal it exactly, and always enforces `w_i in [0, 1]` (the
two-point `floor`/1 set follows from the construction). Failure raises
`CertificateError("F-SOLVE-6")`. **Not checked at runtime**, and not to be
relied on: the symmetry and half-bandwidth of `D_tilde'^T D_tilde'` (both hold
by construction, so the check would be tautological) and a read-back comparison
of the recorded baseline/mask sha256 against the product meta (the hashes are
written for provenance; the mask itself is not stored in the product, so there
is nothing to compare against on read). README section 7.3 states the same
scope.

**Determinism.** Given `y`, calibration and parameters, the baseline, the mask
and hence the solution are deterministic; the mask and baseline sha256 are
recorded (D-160).

**Acceptance metrics (synthetic, D-161): the committed gate is still pending.**
The registered metric set is (a) spurious-peak suppression (the fraction of
injected noise peaks that survive, mask on vs off); (b) true-peak area bias;
(c) pull coverage with the mask on/off; (d) robustness to threshold and
iteration perturbations; (e) bitwise reproducibility; no golden or reference
outputs. As of D-188 only (e) is covered by the suite: the mask-on end-to-end
tests pin structural properties (validity, strict certificates, the D-157
reference bin, the resolution-independent protection footprint) but none of the
acceptance metrics below, and the closure/pull studies still run with the mask
off, so (a)-(d) are documented-but-open. The numbers quoted in this document under "Measured effect
of the redesign" and in F-UNF-1 come from the ad-hoc validation described there
(the Ra-226 dataset and the forward-closure fixture), not from a committed
study; D-162 keeps the study as the acceptance gate.

**Default-parameter study (preliminary, pre-D-188).** The numbers below were
measured under the former resolution-scaled protection rule
(`protect_sigma = 2` resolution widths, floor applied as a stencil product), so
they document how the adopted defaults were reached, not the current rule. A
sensitivity grid was run on
the real Th232 window (1259 bins, alpha = 0.1, strict). Mask off gave
chi2 = 315.9, 537 active bins, max(mu) = 2.73e5. The adopted defaults
(5 sigma / 2 sigma_E / floor 0.1, resolution-derived m) gave chi2 = 256.2,
314 active bins, max(mu) = 7.53e5 with an unchanged total. Sweeping the
parameters moved chi2 within 252-285 and the active count within 200-412:
an over-strong floor (0.3) over-smooths peaks (max(mu) 5.1e5), a weak floor
(0.05) inflates them (1.14e6), and an explicit m = 2 under-smooths
(chi2 = 314). The derived m sits near the best tested value (m = 4 gives 254).
These are *sensitivity* observations on real data, not truth-based tuning:
the synthetic closure study of D-161 (injected peaks with known amplitudes,
pull coverage) remains the gate that may revise the defaults. The realized
values are always recorded in the product meta (D-160), so a re-tuned default
never invalidates an existing product. (D-191 later re-tuned the shipped
defaults to `alpha = 1.0`, `snip_floor = 0.01` and `m_max = 32`; the grid above
documents how the D-162 defaults were reached.)

### F-COV-1 - Fisher information

`F = J^T W J` with `J` the analytic model Jacobian (never finite
differences) and `W = diag(1/sigma^2)` the same weights as the fit. Shapes
and positivity of `sigma` are validated.

### F-COV-2 - chi2/dof scaling

`cov = s^2 F^-1` with `s^2 = chi2/dof` (PDG convention), `dof >= 1`. The
inverse goes through Cholesky; a singular or non-positive-definite Fisher
matrix raises `SolverError` (no pseudo-inverse fallback). The PSD
certificate checks symmetry and the smallest eigenvalue against a relative
tolerance. For a quadratic chi2 the scale is approximately one.

### F-COV-3 - profile covariance (optional diagnostic)

For each parameter, profile points are evaluated at
`p_i +- 4 sqrt(2/H_ii)`; the remaining parameters are re-optimized with a
deterministic Nelder-Mead and the `dchi2 = 1` crossing is solved with
`brentq`. The covariance is `D_w C D_w` where `D_w` holds the profile
widths and `C` is the correlation matrix of the inverse numerical Hessian
`(H/2)^-1`; for a quadratic objective this reproduces `2 A^-1` exactly, the
correct chi2 covariance. `dof` is reported as 0 and the scale as 1 because
the diagnostic does not use a reduced-chi2 scaling. Limitation: the crossing
must lie inside the scan window, otherwise the diagnostic raises
`SolverError`.

### F-UNC-1 - statistical propagation

At a fixed active set, the free variables satisfy `H_FF mu_F = b_F` and the
active ones stay at zero, so `d mu_F = H_FF^-1 (R^T W)_F dy` and
`d mu_A = 0`. The covariance is
`Cov(mu) = H_FF^-1 (R^T W)_F Sigma_stat (W R)_F H_FF^-1` extended by zeros,
and the band is the square root of its diagonal. Using the full `H^-1`
would overstate every free direction whenever a constraint is active; the
implementation always solves the reduced system on the free set
(`_reduced_columns`). `Sigma_stat = diag(max(stat, 1))` is the pure data
statistical variance (F-CAL-1); the systematic part of the fit weights
belongs to F-UNC-2 (D-50). `fisher` is the full half-Hessian from F-SOLVE-1;
when omitted, the data-only `R^T W R` is used and the caller accepts the
documented difference. Reduced systems use the same factorization policy as
the solver.

### F-UNC-2 - systematic propagation

Three contributions, all first-order at fixed active set:

1. **Data-side `syst_frac`.** Same linearization as F-UNC-1 with
   `Sigma = diag((syst_frac * y)^2)`.
2. **Calibration.** With `Q_k = dR/dq_k` (F-RESP-4) and
   `r = R mu - y`, the half-gradient derivative is
   `g_k = Q_k^T W r + R^T W Q_k mu`; the sensitivity columns are the reduced
   solves `V_k = -H_FF^-1 (g_k)_F` (zero on the active set) and
   `Cov_calib = V Sigma_q V^T`.
3. **MC variance (simulation-derived).** With `R = C G diag(1/N)` and `N_j` fixed by the
   sampling design (F-SIM-1), `dR/dG_js = C_j e_s^T / N_s`. Differentiating
   the half-gradient `g = H mu - b` at fixed `mu` gives the **full vector**

   `dg_a/dG_js = [ delta_{a,s} (C_j^T W r) + (R^T W C_j)_a mu_s ] / N_s`

   (D-119). A rank-one form `d_js e_s^T` drops the second term;
   finite differences on random problems show it contributes at the same order
   as the first (~50% of the derivative), so it must be kept. With the
   recorded-bin multinomial covariance `Cov(G_s) = N_s (diag(p_s) - p_s p_s^T)`
   and the reduced free-set inverse `U = H_FF**-1`, the propagated covariance is

   `Cov(mu) = sum_s (1/N_s) U A_s U^T`,
   `A_s = sum_j p_js v_js v_js^T - (sum_j p_js v_js)(sum_k p_ks v_ks)^T`,
   `v_js = (C_j^T W r) e_s + (R^T W C_j) mu_s`.

   The band is the square root of its diagonal:

   `Var_i = sum_s (1/N_s) [ sum_j p_js X_js_i**2 - Xbar_s_i**2 ]`,
   `X_js = U v_js`, `Xbar_s = sum_j p_js X_js`. This term is named
`mc_variance` throughout the code (`propagate_systematic(..., mc_variance=...)`,
`BandComponent(name="mc_variance")`); the helper that computes it keeps the
descriptive name `simulation_mc_variance` because the variance comes from the
finite Monte-Carlo statistics of the simulated response (D-167).

   The unrecorded zero-deposition category has `v = 0`, so it cancels from
   `A_s` (the sums run over recorded deposition bins only); dropping it from a
   centered form would omit its `p_zero Xbar**2` contribution. `Var_i >= 0`
   is checked against a small relative tolerance.

   **Evaluation (D-173, §1.16).** Expanding `X_js_i = a_j U[i,s] +
   mu_s m_{i,j}` with `a_j = C_j^T W r` and `m = U (R^T W C)` gives

   `Var_i = sum_s (1/N_s) [ u_si^2 dA_s + 2 u_si mu_s (t1_si - abar_s t2_si)
   + mu_s^2 (wv_si - t2_si^2) ]`,
   `dA_s = sum_j p_js a_j^2 - abar_s^2`, `t1_si = sum_j p_js a_j g_ij`,
   `t2_si = sum_j p_js g_ij`, `wv_si = sum_j p_js g_ij^2`, `g_i = mixed^T u_i`,
   `u_i = U e_i`. Only the free-set columns `u_i` are solved (in blocks of
   `MC_BLOCK_COLUMNS = 128`) and the three `P^T` contractions are formed per
   block, so neither `H_FF**-1` nor `U (R^T W C)` is materialized as a whole;
   the only dense `O(n^2)` object is the data-side `R^T W C`. The result equals
   the direct linearization above at `rtol = 1e-9`, including active bins
   (where the reduced free-set inverse is the boundary convention).

### F-UNC-3 - strict band decomposition

`total = hypot(stat, syst)`; the certificate verifies
`total^2 = stat^2 + syst^2` with a relative tolerance of `1e-9` and fails
strict mode otherwise. Components record their name, kind and formula ID so
a product can be audited without re-deriving the split.

### F-IO-1 - atomic write, reopen validation and provenance protocol

**Protocol.** A write goes to `<target>.part`; the writer closes the file, the
same process reopens it with uproot and validates every object against the
product contract, and only then is the part moved onto `<target>` with
`os.replace` (atomic on one filesystem). An existing target is refused unless
`force` is set; the parent directory is created on demand; a stale `.part` is
removed before writing; any failure removes the part and leaves an existing
target untouched. The writer and verifier therefore share one implementation
(`write_product` calls `verify_product` on the part), which is what makes the
protocol self-checking.

**Validation tiers.** Always-on checks (never disableable, D-62): the
`product_kind` is known, `format_version == 1`, the on-disk object set equals
`OBJECT_NAMES` exactly (no extra, no missing, no duplicate cycle), each object
has the right uproot type (TH1/TH2/RNTuple), axes are strictly increasing with
a legal unit, values/variances match the axis shapes and are finite, the
required variance buffers exist, and the `meta` field set matches the frozen
type table with length-1 entries. Strict-only product certificates
(`strict=True`, D-61): F-RESP-1/F-RESP-2 (C/R column sums, composition and
efficiency identities), F-COV-2 (param_cov symmetry, PSD and bin labels),
F-SIM-1/F-SIM-2 (count/accounting bounds and `fSumw2 = N_j p (1 - p)`;
`mc_spectrum` uses the source variant `c (1 - c/P)`, D-128),
F-UNC-3 (band decomposition), D-15 band storage (`fSumw2 = sigma**2`) and
F-IO-1 (positive `daq_time_s`, non-negative raw `fSumw2`). The physical-layer
F-SIM-6 boundary certificate is owned by `kc761tool/sim` and runs under strict mode
when a matrix product is produced. Every failure is a `CertificateError`
carrying the formula ID and the offending value.

**Provenance.** `build_provenance` records the git revision and dirty flag,
Python version, the versions of an ordered, single-source dependency list,
the sha256 of every input file, the ordered CLI arguments and a UTC
timestamp. `fingerprint_for`/`input_sha256` look up a recorded digest by exact
or resolved path (no basename fallback, to avoid ambiguous matches). In a
non-git directory the revision is `"unknown"`, the dirty flag is 0 and one
warning is emitted instead of failing (D-93).

**Encoding.** JSON blobs are compact and ASCII: `arguments_json` is an ordered
list of `[name, value]` pairs, `inputs_json` a list of
`{"path", "sha256"}`, `dependency_versions` a list of `[name, version]`;
calibration parameter vectors are JSON number lists in the frozen
`PARAM_NAMES_REPORTED` order. Variances are read back from the raw `fSumw2`
buffer (not `errors()**2`), so a write/read cycle is bit exact. The
`calib_sha256`/`sim_sha256` and `geometry_param` fields are not part of the
contract (D-88/D-89).

### F-UNF-1 - energy window to channel and primary selection

**Assumptions.** The calibration `E(ch)` is strictly increasing on the
acquisition range (F-MODEL-3), the data channel axis is uniform, and the
primary axis is the simulation's variable energy axis (`G.y`).

**Derivation.** Let `centers_i = E(i)` for `i = 0..n-1`. For a requested
energy window `[elo, ehi]` with `elo <= ehi`:

    chlo = clip(first i with centers_i >= elo, 0, n-1),
    chhi = clip(last  i with centers_i <= ehi, 0, n-1).

At least one center must fall inside, otherwise the window is empty at the
channel resolution and a `ValidationError` is raised. `elo`/`ehi` must lie
inside `[E_min, E_max]`, the primary axis range, otherwise the requested
window cannot be represented by the simulation and a `ValidationError` is
raised. The reported `mu` covers the primary bins whose **centers** lie in
`[elo, ehi]`, and that set is also the fit space (D-187): the same index range
supplies the solved primary columns and the reported product axis. The former
F-BIN-3 row padding is retired.

**Limits.** Centers (not bin edges) define membership on both axes; this is a
half-bin boundary convention, documented here so it is not re-derived. A
window narrower than the channel pitch selects at most one channel.

**Edge limitation (D-187).** The response of a primary bin just inside the
window extends below `chlo` (or above `chhi`); those rows are measurement, but
they are outside the requested window and are no longer fitted, so the *data*
that would constrain the leaked part of an edge line's response is not used.
Near the window edge the adjacent primary columns are also strongly collinear
through the response (measured on the fixture: `cos(R[:,50], R[:,51]) = 0.994`,
against 0.945 six bins inside), and that near-degenerate direction is resolved
by the roughness penalty toward the edge. The consequence is therefore not a
one-row attenuation but a boundary layer whose width follows the regularization
smoothing scale and the local collinearity, not the detector resolution alone:

* in a narrow window (fixture, 220-520 keV, 12 fitted channel rows) a truth line
  in the second-to-last reported primary bin recovers 0.20 of its amplitude
  (0.87 with the F-BIN-3 padded rows fitted) and its strength moves *outward*
  into the last reported bin (78% of the recovered sum), while a line six
  reported bins inside loses 15% (0.75 against 0.92 in the wide window);
* widening the window so the line has a few bins of room (200-560 keV) restores
  it to 0.88-0.91, i.e. the interior value;
* for a window many resolution widths wide - the production 30-3000 keV Ra-226
  case, 1336 fitted rows - a forward closure gives a total flux ratio of
  0.9998, so the layer is negligible there.

The regularization dependence is small (the edge/interior ratio stays 0.26-0.44
over `alpha` from 1e-4 to 3e-2) but the absolute ratios are not: the test that
pins this uses the ratio, not an absolute recovery. An analysis that needs quantitative bins at the edge of
its region of interest must request a window wider than that region and quote
only the interior, which is the standard unfolding convention and the reason
the padding existed before D-187.

### F-UNF-2 - data-side unfold fit weights

**Derivation.** With `stat` the per-channel data variance (`fSumw2`), `data`
the measured counts and `syst_frac` the fractional data-side systematic, the
unfold fit uses

    sigma_fit**2 = max(stat, 1) + (syst_frac * data)**2.

The `max(stat, 1)` floor and the `syst_frac = 0.05` default (D-169) are the frozen
F-CAL-1 weight conventions. The simulation-MC term is deliberately **excluded**
from the weights (D-112): it depends on the unknown deposition distribution and
would make the objective iterative. It enters only the systematic band through
F-UNC-2, so the fit stays a single non-negative QP.

**Limits.** The floor is a documented approximation (F-CAL-1 limitations); for
Poisson data with `pred < 1` it biases `chi2/dof` low. The systematic term is
linearized in `data`, so it does not model the Poisson variance of the
systematic itself.

### F-UNF-3 - exact-zero column pruning

**Derivation.** `R = C . G . diag(1/N)` is non-negative, so a column is exactly
zero if and only if its sum is exactly zero. Columns with `sum_i R_ij == 0.0`
carry no data information: their gradient entry `(R^T W y)_j` is zero and their
normal-matrix diagonal `(R^T W R)_jj` is zero, so the non-negative solver fixes
them at zero in every mode. They are removed from the solved matrix (D-110) and
re-inserted as zeros in the reported solution; the removal uses exact `== 0.0`
comparison, never a tolerance.

**Limits.** Removing columns re-indexes the finite-difference penalty
(F-SOLVE-1). In this pipeline exact-zero columns are the leading/trailing part
of the primary range (depositions that cannot reach any channel), so the kept
range is contiguous and the reduced penalty is the exact sub-block of the
full-axis penalty. An interior zero column would make the reduced penalty a
re-indexed operator; the definition here is the reduced operator (`D` on the
kept columns), which is the frozen meaning of "solve on the remaining columns".

### F-UNF-4 - unfold diagnostics

**Derivation.** With the fit rows `F = [chlo, chhi]` (the reported channel rows,
D-187; formerly padded), the reduced
response `R_FK` (kept columns `K`), the solution `mu_K`, weights
`sigma_fit` (F-UNF-2) and data `y_F`:

    chi2 = sum_F ((y - R mu)**2 / sigma_fit**2),
    n_active = #{k in K : mu_k > 0},
    dof = |F| - n_active,
    covariance_scale = 1.

`covariance_scale` is fixed to one because the unfold reports the analytic
first-order propagation (F-UNC-1/F-UNC-2) and never rescales it by a reduced
chi-square; `dof` is the number of fit rows minus the number of active
(positive) solution bins. `dof` can be zero or negative for a heavily
regularized / underdetermined problem; it is then reported as computed and
`chi2/dof` is not used.

**Limits.** `n_active` counts strictly positive bins, so a bin that sits at the
constraint is not counted as a fitted parameter even though its uncertainty is
zero (F-UNC-1); this is the active-set convention of F-SOLVE-2/F-SOLVE-3.

### F-UNF-5 - refolded channel spectrum

**Derivation.** With the full-axis composed response `R` (F-RESP-2), the
full-axis solution `mu` (zeros on pruned columns) and the reported channel
window `[chlo, chhi]`,

    refolded_i = sum_j R_ij mu_j,   i in [chlo, chhi].

This is the model prediction of the measured channel spectrum under the
unfolded primary spectrum; it uses the same full-axis composition as the solve and
is sliced to the reported channel window only for storage.

### F-UNF-6 - calib-only channel relabeling

**Derivation.** Calibration-only mode does not invert the response. It reuses
the channel-derived deposition axis stored in `C.y`, whose edges are
`E(i +- 1/2)` (F-RESP-1), as the new energy axis of the measured spectrum:

    values'_i = values_i,   variances'_i = variances_i,
    edges'_i = C.y.edges_i,  unit = kev.

Counts and `fSumw2` are copied bin by bin (no interpolation, no rebinning), so
the operation is exactly invertible by the axis alone.

**Limits.** The data channel axis must equal `C.x` bitwise (D-114); otherwise
the relabeling would attach the wrong energies to the counts.


### F-SPEC-1 - spectrum addition

**Derivation.** Two measured spectra `a` and `b` of the same source over the
same channel axis are two independent counting experiments, so their bin
contents and their Poisson variances add:

    values_i = a_i + b_i,
    e_a,i = max(sqrt(fSumw2_a,i), 1),   e_b,i = max(sqrt(fSumw2_b,i), 1),
    variances_i = e_a,i**2 + e_b,i**2,
    daq_time_s = t_a + t_b.

The one-count error floor follows the convention of F-SPEC-2 and of the fit
weights (F-CAL-1/F-UNF-2): a bin with zero recorded variance still carries a
one-count uncertainty, so a sum never claims zero uncertainty. For a Poisson
product (`csv2root`) `fSumw2 = counts`, so the floor is a no-op on every
non-empty bin and only lifts the exactly-zero variance of an empty bin, which
it must: two empty bins must not sum to a bin with zero uncertainty.

**Preconditions (always on).** Both operands are `spectrum` products, their
channel axes are equal bitwise (unit and edges) and both `daq_time_s` values
are finite and positive; a violation is a `SchemaError`/`ValidationError` in
both modes. The output is a `spectrum` product on the shared axis whose
`daq_time_s` is the sum and whose audit `source_file` is `<a> + <b>` (D-185).

**Limits.** The addition is exact in the stored representation: no rebinning,
no interpolation and no clamping. `fSumw2` is taken to be the per-bin variance
already. Values, variances and the DAQ time are commutative; only the audit
`source_file` string keeps the positional order.

### F-SPEC-2 - DAQ-time scaled spectrum subtraction

**Derivation.** A background spectrum `b` recorded for `t_b` differs from the
background accumulated during a measurement `a` of duration `t_a` by the
DAQ-time ratio `r = t_a / t_b`. Subtracting the scaled background bin by bin
gives

    r = t_a / t_b,
    values_i = a_i - r b_i,
    e_a,i = max(sqrt(fSumw2_a,i), 1),   e_b,i = max(sqrt(fSumw2_b,i), 1),
    variances_i = e_a,i**2 + r**2 e_b,i**2,
    daq_time_s = t_a.

The two measurements are independent, so their errors add in quadrature with
the scale factor applied to the background only; the net spectrum represents
the source measured for `t_a`, which the output therefore inherits.

**Preconditions (always on).** Identical to F-SPEC-1: both operands are
`spectrum` products, their channel axes are equal bitwise and both DAQ times
are finite and positive, so `r` is finite and positive. The output is a
`spectrum` product on the shared axis with the audit `source_file` `<a> - <b>`
(D-185).

**Limits.** Negative net bins are physical outcomes of the subtraction and are
stored unchanged; the fail-loud rule of plan section 5 forbids clamping them.
The one-count error floor prevents a zero-variance bin from contributing zero
uncertainty. `r` is a ratio of recorded DAQ times: dead time, rate-dependent
effects and any change of background composition between the two runs are not
modeled.
