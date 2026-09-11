# KC761 toolkit

Simulation, calibration and unfolding toolkit for the MEASALL KC761x/KC761
gamma spectrometer. Pure Python plus `geant4-pybind`: Geant4 source and matrix
simulations, an energy/resolution calibration with a statistical response
matrix, and non-negative regularized spectrum unfolding with propagated
uncertainties.

This README is organized around the **mathematics and algorithms**. Every
formula carries a registry ID (`F-...`) whose full derivation, approximations
and guarding certificate live in [docs/derivations.md](docs/derivations.md);
the frozen decisions referenced as `D-nnn` are in
[docs/plan.md](docs/plan.md).

---

## 1. The problem this solves

A scintillation detector does not measure gamma energies; it measures a
smeared, efficiency-limited, background-contaminated channel histogram. The
toolkit estimates three mathematical objects and composes them into one linear
inverse problem:

| Object | Symbol | Meaning | Built by |
|--------|--------|---------|----------|
| Energy map | $E(\mathrm{ch})$ | channel $\to$ keV, cubic, strictly increasing | `calib` (F-MODEL-1) |
| Resolution | $\sigma(E)$ | detector width in keV ($\mathrm{FWHM} = 2.355\ \sigma$) | `calib` (F-MODEL-4) |
| Response | $R$ | primary energy $\to$ expected channel counts per primary | `compose` (F-RESP-2) |

The measured spectrum $y$ is then modeled as

$$
y \ \approx\  R\ \mu + \text{noise}, \qquad \mu \ge 0,
$$

where $\mu$ is the primary (incident) energy spectrum. $\mu$ is recovered by
`unfold` (F-SOLVE-1..6) and reported with strictly split uncertainty bands
(F-UNC-1..3). The full data flow is

```
raw CSV --csv2root--> spectrum --subbkg--> data
                                             |
             Geant4 source mode --> mc_spectrum
                                             |
                                    calib <--+        (fits E, sigma, scales)
                                             |
             Geant4 matrix mode --> G  --compose--> R
                                             |
                              unfold <-------+-------> mu, stat/syst bands
```

Every stage is a matrix or a low-dimensional optimization; no stage compares
against stored reference output. Correctness is defined by the derivations and
the runtime certificates of section 11.

---

## 2. Detector model: energy scale and resolution

### 2.1 Dual-basis cubic energy calibration (F-MODEL-1/F-MODEL-2)

$E(\mathrm{ch})$ is cubic on the acquisition range
$`[0, \mathrm{ch}_{\max}]`$. The fit does **not** use the plain coefficients: it
uses the value at the origin and the slopes at three nodes,

$$
c_0 = E(0), \qquad k_1 = E^{\prime}(0), \qquad
k_2 = E^{\prime}(\mathrm{ch}_{\max}/2), \qquad k_3 = E^{\prime}(\mathrm{ch}_{\max}),
$$

which gives the numerically well-scaled internal form

$$
E(\mathrm{ch}) = c_0 + k_1\ \mathrm{ch} + \frac{4k_2 - 3k_1 - k_3}{2\ \mathrm{ch}_{\max}}\ \mathrm{ch}^2 + \frac{2\ (k_1 - 2k_2 + k_3)}{3\ \mathrm{ch}_{\max}^2}\ \mathrm{ch}^3.
$$

Products store the plain cubic $`(c_0, c_1, c_2, c_3)`$. The two bases are related
by an **affine** map, so its Jacobian is constant in the parameters
(`internal_jacobian`), and covariance transforms between bases by
$T \mathrm{cov} T^{\mathsf{T}}$. Round-tripping is exact to round-off.
This basis choice is what keeps the fit conditioning sane: the slope
coordinates are $O(1)$ where the raw cubic coefficients differ by orders of
magnitude.

### 2.2 Resolution as a quadratic Bernstein form (F-MODEL-4)

With $`t = \max(E, 0) / E_{\mathrm{REF}}`$ and $`E_{\mathrm{REF}} = 2000`$ keV, the
variance is the degree-2 Bernstein polynomial with control values
$`(b_0^2, b_1^2, b_2^2) \ge 0`$:

$$
\sigma^2(t) = (1-t)^2 b_0^2 + 2(1-t)\ t\ b_1^2 + t^2 b_2^2.
$$

For $t \in [0, 1]$ this is a **convex combination** of non-negative controls,
so positivity is structural rather than checked. Only $`b_k^2`$ enters, so the
sign of $`b_k`$ is irrelevant. Exact derivatives
$`\partial\sigma / \partial b_k`$ come from the same symbolic expression (§10).
For $`E > E_{\mathrm{REF}}`$ the form is extrapolated and the cross term can
drive $\sigma^2$ negative — that is a physical statement about the model class,
and it is guarded by a certificate rather than hidden (§2.3).

### 2.3 Certificates rather than assumptions

* **Monotonicity (F-MODEL-3).** $E^{\prime}$ is quadratic, so its minimum on
  $`[0, \mathrm{ch}_{\max}]`$ lies at an endpoint or at the exact vertex. The
  certificate reconstructs the quadratic coefficients by finite differences and
  evaluates the vertex — a sampling grid could hide a narrow dip, an exact
  check cannot. A zero slope is rejected: a plateau makes the channel-energy
  map non-invertible.
* **Positivity (F-MODEL-5).** Strict mode raises when
  $\sigma^2 < -10^{-9}\ \mathrm{keV}^2$. Outside strict mode a non-positive
  variance is clamped to $`\sigma_{\text{floor}} = 10^{-3}`$ keV; a strictly
  negative variance emits a `RuntimeWarning`, and the affected energies are
  recorded in the product `meta` as `resol_clamp_count`,
  `resol_clamp_energy_low_kev` and `resol_clamp_energy_high_kev`. The clamp is
  explicit, logged and recorded — never silent.

### 2.4 Working window and padding (F-BIN-3)

The solver models
$`[\mathrm{ch}_{\mathrm{lo}} - \mathrm{pad},\  \mathrm{ch}_{\mathrm{hi}} + \mathrm{pad}]`$
and reports $`[\mathrm{ch}_{\mathrm{lo}}, \mathrm{ch}_{\mathrm{hi}}]`$. The pad
covers the smearing of events whose true channel lies outside the reported
window:

$$
\mathrm{pad} = \left\lceil
  \frac{n_{\mathrm{pad}}\ \sigma(E_{\mathrm{edge}})}{w_{\mathrm{edge}}}
\right\rceil,
$$

with $`w_{\mathrm{edge}} = E(i + \tfrac12) - E(i - \tfrac12)`$ the local channel
width from the calibration. $`n_{\mathrm{pad}} = 0`$ is legal and disables
padding.

---

## 3. The response kernel: exact Gaussian bin integrals

### 3.1 Exact bin probability, no midpoint approximation (F-KERN-1)

For a source at energy $`c_j`$ with width $`\sigma_j`$, the probability that the
smeared energy lands in channel bin $i$ with edges $`[e_i, e_{i+1}]`$ is the
Gaussian integral over the bin,

$$
P(i \mid j) = \Phi\left(\frac{e_{i+1} - c_j}{\sigma_j}\right) - \Phi\left(\frac{e_i - c_j}{\sigma_j}\right),
$$

with $\Phi$ the standard normal CDF evaluated through $\mathrm{erf}$. The
expression is analytic; the only error is floating-point round-off. There is no
midpoint approximation and no truncation to a tabulated kernel.

### 3.2 C1 smoothstep support taper and exact renormalization (F-KERN-2)

A Gaussian has infinite support; the kernel is truncated with a compactly
supported $C^1$ taper. With $x$ the **bin-center** offset from the source
(D-83), $`s = \mathrm{clip}\big((n_\sigma \sigma - \lvert x \rvert)/\sigma,\  0,\  1\big)`$ and

$$
w(x) = 3s^2 - 2s^3.
$$

$w = 1$ on the plateau $`\lvert x \rvert \le (n_\sigma - 1)\sigma`$, decays
smoothly to $0$ at $`\lvert x \rvert = n_\sigma \sigma`$, and is exactly $0$
beyond. Its first derivative $(6s - 6s^2)\ \mathrm{d}s/\mathrm{d}x$ vanishes at
both clip boundaries, so the taper is $C^1$ and adds **no kink** to the fit
objective. The tapered column is then renormalized exactly:

$$
n_{ij} = P(i \mid j)\  w_{ij}, \qquad
D_j = \sum_i n_{ij}, \qquad
p_{ij} = \frac{n_{ij}}{D_j}.
$$

Every non-empty column therefore sums to exactly $1$; a column with $`D_j = 0`$
is exactly zero. Renormalization is what makes the taper an approximation of
the *shape* rather than a silent loss of probability.

### 3.3 Sparse assembly with exact-zero pruning (F-KERN-3)

Only bin centers strictly inside
$`(c_j - n_\sigma \sigma_j,\  c_j + n_\sigma \sigma_j)`$ are evaluated
(`searchsorted`). Everywhere else the taper is *exactly* zero and all its first
derivatives vanish, so the pruned sum is identical to the full sum for
**every** parameter value. The pattern has $`O(\sum_j \mathrm{reach}_j)`$ entries
with $`\mathrm{reach}_j \sim 2 n_\sigma \sigma_j / w`$; no dense
$`n_{\text{channels}} \times n_{\text{deposition}}`$ intermediate is ever created,
and the assembled matrix is CSR.

This is the mechanism behind D-79: the response geometry is
**parameter-independent** (full deposition axis $\times$ requested channel rows,
no freeze, no bin selection), so the fit objective is continuous in the
parameters *by construction* while remaining sparse.

### 3.4 Kernel derivatives and center folding (F-KERN-4)

The generated module provides $`\partial n/\partial e_{\mathrm{lo}}`$,
$`\partial n/\partial e_{\mathrm{hi}}`$, $\partial n/\partial c$,
$\partial n/\partial \sigma$ and $`\partial n/\partial c_{\text{ctr}}`$. Because
the bin center is $`(e_{\mathrm{lo}} + e_{\mathrm{hi}})/2`$, moving either edge
displaces the taper argument by half the displacement, so the kernel adds
$`\tfrac12\ \partial n/\partial c_{\text{ctr}}`$ to each edge derivative. The
clipped smoothstep is differentiated by the chain rule with the clipping
prefactor, which vanishes outside the transition band — the $C^1$ extension is
exact.

---

## 4. Response composition and the analytic Jacobian

### 4.1 The matrices $C$ and $R = C\ G\mathrm{diag}(1/N)$ (F-RESP-1/F-RESP-2)

$C[i,j]$ is the probability that a gamma depositing energy in deposition bin
$j$ lands in channel bin $i$; the channel edges are $E(i - \tfrac12)$ from
F-MODEL-1. With $G$ the matrix-mode deposition-by-primary count matrix, $`S_j`$
its column sums and $`N_j`$ the per-column generated-event totals,

$$
\tilde{p} = \frac{G}{S_j}, \qquad \eta_j = \frac{S_j}{N_j}, \qquad
R = C\ \tilde{p}\ \mathrm{diag}(\eta) = C\ G\ \mathrm{diag}(1/N).
$$

The two forms are algebraically identical because
$\tilde{p}\ \eta = G/N$. The implementation uses the second: it has no
intermediate $0/0$ and one fewer normalization step. $`\eta_j`$ is the
**detection efficiency** (F-SIM-3) and is validated to lie in $[0, 1]$; it
follows the identity $`\eta_j = 1 - \mathrm{zero}_j / N_j`$.

Read that identity precisely: $`\mathrm{zero}_j`$ counts every generated primary
of column $j$ that did **not** land in an in-range deposition bin, which
includes both events that deposited nothing in the crystal and events whose
total deposit fell outside the deposition axis. $`\eta_j`$ is therefore the
fraction of primaries with an in-range deposit — the containment the matrix can
represent, not a claim about the crystal's physical detection threshold.

### 4.2 Compose on the full axis, then slice (F-RESP-3)

Composition runs over the **full** primary axis and window slicing happens
afterward. `column_sums` and `efficiency` keep their full-axis meaning, while
`channel_low`/`channel_high` describe the rows the matrix actually holds.
Composing after slicing would renormalize the response with the wrong window;
the certificate instead verifies that slicing only removed non-negative row
mass and that sliced row sums never exceed full column sums.

### 4.3 Response Jacobian by chaining (F-RESP-4)

For $`q = (c_0, c_1, c_2, c_3, b_0, b_1, b_2)`$ in the reported basis, a channel
edge contributes through its energy and a deposition column through its width:

$$
\begin{aligned}
u_{ij} &= \frac{\partial n}{\partial e_{\mathrm{lo}}}\ \frac{\partial e_i}{\partial q_k} + \frac{\partial n}{\partial e_{\mathrm{hi}}}\ \frac{\partial e_{i+1}}{\partial q_k} && \text{(calibration)}, \\
u_{ij} &= \frac{\partial n}{\partial \sigma_j}\ \frac{\partial \sigma_j}{\partial b_k} && \text{(resolution)},
\end{aligned}
$$

with $`\partial e_l / \partial q_k = \mathrm{ch}_l^{\ k}`$ for the reported basis
(F-MODEL-2). The **quotient rule couples every entry of a column** through the
renormalization denominator:

$$
\frac{\partial p_{ij}}{\partial q_k} = \frac{u_{ij}}{D_j} - \frac{n_{ij} \sum_{i'} u_{i'j}}{D_j^{2}}.
$$

Every column of $\partial C/\partial q$ therefore sums to zero — a conservation
identity the tests check — and the composed Jacobian is
$`\partial R/\partial q_k = (\partial C/\partial q_k)\ P`$ with
$P = G\mathrm{diag}(1/N)$. All seven matrices are returned as sparse CSR;
no dense $`n_{\text{channels}} \times n_{\text{deposition}} \times 7`$ tensor is
materialized.

### 4.4 Overlap projection between binnings (F-PROJ-1/F-PROJ-2)

For rebinning, the union of source and target edges is split into segments,
each lying inside exactly one source and one target bin. $W[t,s]$ is the
segment length inside target bin $t$ divided by the source bin width, so each
source bin is distributed exactly. `build_projection_plan` requires full
coverage (within $10^{-9}$) and fails loudly otherwise — it never silently
drops mass. Values project as $Wv$ (mass conserving) and independent variances
as $W^2\ \mathrm{Var}$; identical binnings give $W = I$, so the projection is
idempotent. Covariances between source bins are not modeled (documented
limitation).

---

## 5. Unfolding I: the regularized non-negative problem

### 5.1 Tikhonov objective with a normalization-invariant `alpha` (F-SOLVE-1)

Unfolding is ill-posed: neighboring primary bins map to nearly the same
channel distribution, so the unregularized least-squares solution oscillates
violently. The toolkit minimizes

$$
\min_{\mu \ge 0}\ \  \chi^2(\mu) + \alpha\ \lVert \tilde{D}\mu \rVert^2,
\qquad
\chi^2(\mu) = \left\lVert \frac{R\mu - y}{\sigma} \right\rVert^2,
$$

with the weight matrix and normal-equation building blocks

$$
W = \mathrm{diag}(1/\sigma^2), \qquad
A = R^{\mathsf{T}} W R, \qquad
b = R^{\mathsf{T}} W y.
$$

Here $D$ is the order-1 $[-1, 1]$ or order-2 $[1, -2, 1]$ finite-difference
operator and

$$
\tilde{D} = D \mathrm{diag}\left(\sqrt{\mathrm{diag}(A)}\right).
$$

The normalization is the point of D-80. Rescale the data by $y \to ky$,
$\sigma \to k\sigma$. Then $W \to W/k^2$, hence $A \to A/k^2$ and
$\mathrm{diag}(\sqrt{A}) \to \mathrm{diag}(\sqrt{A})/k$. Taking the
solution to scale as $\mu \to k\mu$, **both** objective terms are then exactly
invariant:

$$
\chi^2 \to \chi^2, \qquad
\lVert \tilde{D}\mu \rVert^2 \to \lVert \tilde{D}\mu \rVert^2.
$$

The minimizer therefore scales as $\mu \to k\mu$ (counts in, counts out) while
its **shape** is unchanged, and the ratio that $\alpha$ balances is invariant.
That is what makes one $\alpha$ meaningful across datasets, windows and units
(counts vs counts per second): the balance between data fidelity and roughness
does not depend on how the counts happen to be normalized. Note that the
invariances claimed are under count rescaling only: $D$ is the unit-coefficient
difference of adjacent **bins**, so $\alpha$ is not transferable to a different
binning or to a response expressed in different units.

The half-gradient is $g(\mu) = H\mu - b$ with

$$
H = A + \alpha\ \tilde{D}^{\mathsf{T}}\tilde{D}.
$$

$H$ is built once and shared by the solver and the uncertainty propagation, so
values and errors cannot drift apart (D-150). Zero-curvature columns
($`A_{jj} = 0`$) are dropped from the penalty exactly (they carry no data
information) and the solver fixes them at zero.

$\alpha$ is **mandatory** (D-45): there is no default and no automatic
selection, because the choice of regularization strength is a physics
statement, not a numerical detail.

### 5.2 Lawson-Hanson active set on the normal equations (F-SOLVE-2)

The QP $\min \tfrac12 \mu^{\mathsf{T}} H \mu - b^{\mathsf{T}}\mu$ subject to
$\mu \ge 0$ is solved with a self-implemented **Lawson-Hanson active set**
method:

1. start at $\mu = 0$;
2. solve the reduced system $`H_{FF}\ \mu_F = b_F`$ on the free set $F$;
3. if the proposal is positive, accept it; otherwise step from the current
   feasible $\mu$ toward it until a variable hits the boundary, move exactly
   the blocking variables into the active set, and repeat;
4. release the blocked variable with the most negative reduced gradient; stop
   when none has one below the tolerance.

Bins with zero curvature and zero gradient are fixed at zero; a zero-curvature
bin with a non-zero gradient is an unbounded problem and raises `SolverError`.
Reduced systems use dense Cholesky for small or dense matrices, banded
Cholesky when the half-bandwidth is below $n/4$, and sparse LU otherwise
(`core/_linalg.py`, one shared policy). The iteration budget is $10n + 100$ and
exhaustion raises in every mode — it never returns a half-converged answer.

### 5.3 KKT certificate in data-gradient units (F-SOLVE-3)

For $r = H\mu - b$, the reported metrics are

$$
\frac{\max\left(0,\  -\min_{i \in \text{active}} r_i\right)}{\max\left(1, \lVert b \rVert_\infty\right)} \le 10^{-6},
\qquad
\frac{\max_i \lvert \mu_i r_i \rvert}
     {\max\left(1, \lVert b \rVert_\infty\right)\ \max\left(1, \lVert \mu \rVert_\infty\right)} \le 10^{-6}.
$$

Dividing by the data-gradient scale makes the certificate **independent of the
count normalization** of the problem: the same solution quality passes whether
the spectrum is stored in counts or in counts per second. This is the
complementarity-and-dual-feasibility pair of the QP KKT system. Strict mode
raises `CertificateError("F-SOLVE-3")` on failure.

### 5.4 Pruning and diagnostics (F-UNF-3/F-UNF-4)

$R = C\ G\mathrm{diag}(1/N)$ is non-negative, so a column is exactly zero
**iff** its sum is exactly zero (compared with `== 0.0`, never a tolerance).
Such columns have zero gradient and zero normal-matrix diagonal, so the solver
fixes them at zero anyway; they are removed before the solve (D-110) and
re-inserted as zeros. Reported diagnostics are

$$
\chi^2 = \sum_{i \in F} \frac{(y_i - (R\mu)_i)^2}{\sigma_{\mathrm{fit},i}^2},
\qquad
n_{\text{active}} = \lbrace k : \mu_k > 0 \rbrace,
\qquad
\mathrm{dof} = \lvert F \rvert - n_{\text{active}},
$$

with $`\texttt{covariance\_scale} = 1`$.

`covariance_scale` is fixed at one because the unfold reports the analytic
first-order propagation and never rescales it by a reduced chi-square. `dof`
can be zero or negative for a heavily regularized problem; it is reported as
computed and $\chi^2/\mathrm{dof}$ is then not used.

---

## 6. Unfolding II: SNIP peak protection

Regularization suppresses noise-driven oscillations, but a global $\alpha$
large enough to do that also erodes genuine peaks. The toolkit resolves this
with a **data-derived, pre-solve, fixed** diagonal mask that relaxes the
roughness penalty at resolved peaks. Because the mask depends only on the
measured spectrum, the problem stays convex and the KKT certificate is
unchanged (D-155).

### 6.1 SNIP LLS baseline (F-SOLVE-4)

SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) is used **only
to locate genuine peaks**, never as a background measurement. Background
subtraction can leave negative bins, so the estimator operates on
$y^{+} = \max(y, 0)$ and records how many bins were clipped and their index
range — no shift and no imputation. The dynamic range is compressed with
Morháč's log-log-sqrt transform,

$$
v_i = \ln\Big(\ln\big(\sqrt{y_i^{+} + 1} + 1\big) + 1\Big),
\qquad
y_i = \Big(e^{\ e^{v_i} - 1} - 1\Big)^2 - 1,
$$

and the iteration is, for $p = 1 \dots m$,

$$
v_i \ \leftarrow\  \min\left(v_i,\  \frac{v_{i-p} + v_{i+p}}{2}\right),
$$

with out-of-range neighbors replaced by $`v_i`$ (edge-preserving). The iteration
count $m$ is **resolution-derived, not a free knob** (D-157):

$$
\mathrm{FWHM}_{\text{bins}} = \frac{2\sqrt{2\ln 2}\ \sigma_E(E_{\text{mid}})}{\Delta_E},
\qquad
m = \mathrm{clip}\left(\mathrm{round}\left(\tfrac12 \mathrm{FWHM}_{\text{bins}}\right),\  1,\  m_{\max} = 8\right).
$$

$`E_{\text{mid}}`$ is the midpoint energy of the spectrum handed to the mask (the
full measured axis in the shipped pipeline; `solver.snip_peak_mask` uses the
middle bin index) and $`\Delta_E`$ the local bin width at that energy, so
$`\mathrm{FWHM}_{\text{bins}}`$ is the detector peak width measured in bins.
Tying $m$ to the detector width is what removes the detector peak before
estimating the continuum — the intended behavior. An explicit override is
allowed and recorded. The cap $`m_{\max} = 8`$ usually clips the derived value,
so this is a coarse resolution matching rather than a precise one.

### 6.2 Resolution-matched significance and the peak mask (F-SOLVE-5)

The residual is $`r_i = y_i^{+} - b_i`$. Since the detector width is known,
significance is computed with a **matched filter** rather than a per-bin
threshold: with $`s_i = \sigma_E(E_i)/\Delta_i`$ and a normalized Gaussian kernel
$g$ of width $`s_i`$,

$$
M_i = \sum_k g_k\  r_{i+k}, \qquad
V_i = \sum_k g_k^2\ \sigma_{y,i+k}^2, \qquad
z_i = \frac{M_i}{\sqrt{V_i}}.
$$

The matched filter suppresses single-bin noise spikes — the dominant
spurious-peak seed — which a per-bin threshold would misclassify. A bin is a
candidate when $`z_i \ge k`$ (default $k = 5$, D-156) **and** $`z_i`$ is a local
maximum, tested against its two immediate neighbors. All bins within
$`n_{\text{protect}}\ \sigma_E(E_i)`$ (default $2$) of a candidate are marked, and

$$
w_i = \begin{cases}
w_{\text{floor}} \ (\text{default } 0.1) & \text{on marked bins},\\
1 & \text{elsewhere}.
\end{cases}
$$

Bins without data support keep $`w_i = 1`$, i.e. they are smoothed normally.

### 6.3 Masked operator, still symmetric and banded (F-SOLVE-6)

The masked difference operator scales each **row** by the square root of the
product of the mask weights over its stencil:

$$
D' = \mathrm{diag}(\rho^{1/2})\ D, \qquad
\rho_r = \prod_{j=0}^{\text{order}} w_{r+j}.
$$

A protected peak bin therefore relaxes every difference row that touches it: a
row fully inside a peak carries
$`\rho = w_{\text{floor}}^{\ \text{order}+1}`$, a row at a peak/continuum boundary
$`\rho = w_{\text{floor}}^{\ \text{order}}`$. Crucially,

$$
D'^{\mathsf{T}} D' = D^{\mathsf{T}} \mathrm{diag}(\rho)\  D
$$

stays symmetric positive semidefinite and banded with the same half-bandwidth
as $D$. (The rectangular $W^{1/2} D W^{1/2}$ form does not even typecheck for
$`n_{\text{rows}} = n - \text{order}`$; the row form is the correct symmetric
weighting.) The penalty operator is then
$\tilde{D}' = D'\mathrm{diag}\big(\sqrt{\mathrm{diag}(A)}\big)$ —
the same diagonal scaling as F-SOLVE-1, so $\alpha$ keeps the same meaning —
and

$$
H = A + \alpha\ \tilde{D}'^{\mathsf{T}}\tilde{D}', \qquad
b = R^{\mathsf{T}} W_{\text{data}}\  y.
$$

**Strict-mode certificate.** `verify_snip_mask` recomputes the mask from the
recorded spectrum and settings and requires the stored weights to match it
exactly; the $`w_i \in [0, 1]`$ bound is always-on. The baseline and mask sha256
are written into the product `meta`, so a re-tuned default never invalidates an
existing product and the realized values remain auditable. The further checks
sketched in F-SOLVE-6 (symmetry and half-bandwidth of $\tilde{D}'$, hash
comparison on read-back) are **not yet implemented**; do not rely on them.

**Scope of the claim.** The mask is a *structural prior*, not a background
measurement. Thresholding many bins inflates the family-wise false-positive
rate; the $5\sigma$ threshold plus the resolution-width matched filter keeps it
small but not zero. A wrongly protected noise spike is *less* smoothed than
before — the known failure mode, quantified by the synthetic acceptance metrics
of F-SOLVE-6 (spurious-peak suppression, true-peak area bias, pull coverage
mask-on vs mask-off, robustness to the threshold and iteration perturbations).
The reported covariance is conditional on the realized mask; mask-selection
uncertainty is not propagated (D-159).

---

## 7. Uncertainty propagation with a strict stat/syst split

### 7.1 Statistical band on the free set (F-UNC-1)

At a fixed active set, the free variables satisfy $`H_{FF}\mu_F = b_F`$ and the
active ones stay at zero, so
$`\mathrm{d}\mu_F = H_{FF}^{-1}(R^{\mathsf{T}}W)_F\ \mathrm{d}y`$ and
$`\mathrm{d}\mu_A = 0`$:

$$
\mathrm{Cov}(\mu) = H_{FF}^{-1}\ (R^{\mathsf{T}}W)_F\ \Sigma_{\text{stat}}\ (WR)_F\ H_{FF}^{-1},
\quad\text{extended by zeros},
\qquad
\Sigma_{\text{stat}} = \mathrm{diag}\big(\max(\text{stat}, 1)\big).
$$

Using the full $H^{-1}$ instead would **overstate every free direction whenever
a constraint is active**, so the implementation always solves the reduced
system on the free set. $H$ is the same half-Hessian that solved the problem,
so values and errors cannot drift apart.

### 7.2 Systematic contributions (F-UNC-2)

Three contributions, all first-order at fixed active set.

**Data-side `syst_frac`** (default 0.05, D-169) is the same linearization as
F-UNC-1 with
$`\Sigma = \mathrm{diag}\big((\texttt{syst\_frac}\cdot y)^2\big)`$.

**Calibration.** With $`Q_k = \partial R/\partial q_k`$ from F-RESP-4, the
half-gradient derivative is the **full** expression

$$g_k = Q_k^{\mathsf{T}} W r + R^{\mathsf{T}} W Q_k \mu,$$

with sensitivity columns $`V_k = -H_{FF}^{-1}(g_k)_F`$ (zero on the active set)
and $`\mathrm{Cov}_{\text{calib}} = V\Sigma_q V^{\mathsf{T}}`$. Both terms
are kept: the first is the direct response perturbation at the data residual,
the second the response perturbation acting on the current solution.

**Simulation MC.** $`N_j`$ is fixed by the sampling design (F-SIM-1), so the
deposition counts are multinomial with
$`\mathrm{Cov}(G_s) = N_s\big(\mathrm{diag}(p_s) - p_s p_s^{\mathsf{T}}\big)`$.
Differentiating the half-gradient gives the full vector

$$\frac{\partial g_a}{\partial G_{js}} = \frac{\delta_{a,s}\ \big(C_j^{\mathsf{T}} W r\big) + \big(R^{\mathsf{T}} W C_j\big)_a \mu_s}{N_s},$$

and with $`U = H_{FF}^{-1}`$ and
$`v_{js} = \big(C_j^{\mathsf{T}} W r\big)e_s + \big(R^{\mathsf{T}} W C_j\big)\mu_s`$,

$$
\mathrm{Cov}(\mu) = \sum_s \frac{1}{N_s}\ U A_s U^{\mathsf{T}},
\qquad
A_s = \sum_j p_{js} v_{js} v_{js}^{\mathsf{T}} - \Big(\sum_j p_{js} v_{js}\Big)\Big(\sum_k p_{ks} v_{ks}\Big)^{\mathsf{T}}.
$$

A rank-one $`d_{js} e_s^{\mathsf{T}}`$ form drops the second term, which finite
differences show contributes at the same order as the first (\ 50%), so it must
be kept (D-119). The unrecorded zero-deposition category has $v = 0$ and cancels
from the *centered* form $`A_s`$, which is why the sums run over recorded
deposition bins only.

### 7.3 Streaming evaluation, no $n \times n$ inverse (D-173)

Expanding $`X_{js,i} = a_j U[i,s] + \mu_s m_{i,j}`$ with
$`a_j = C_j^{\mathsf{T}} W r`$ and $m = U(R^{\mathsf{T}} W C)$ gives the three-term
form actually evaluated,

$$
\begin{aligned}
\mathrm{Var}_i = \sum_s \frac{1}{N_s}\Big[\ & u_{si}^2\ \mathrm{d}A_s + 2\ u_{si}\mu_s\ \big(t_{1,si} - \bar{a}_s t_{2,si}\big) \\
& + \mu_s^2\ \big(w_{v,si} - t_{2,si}^2\big) \Big],
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{d}A_s &= \sum_j p_{js} a_j^2 - \bar{a}_s^2, &
\bar{a}_s &= \sum_j p_{js} a_j, \\
t_{1,si} &= \sum_j p_{js} a_j g_{ij}, &
t_{2,si} &= \sum_j p_{js} g_{ij}, \\
w_{v,si} &= \sum_j p_{js} g_{ij}^2, &
g_i &= \text{mixed}^{\mathsf{T}} u_i, \\
u_i &= U e_i, &
\text{mixed} &= R^{\mathsf{T}} W C.
\end{aligned}
$$

Only the free-set columns $`u_i`$ are solved, in blocks of
`MC_BLOCK_COLUMNS = 128`, and the contractions are formed per block. Neither
$`H_{FF}^{-1}`$ nor $U(R^{\mathsf{T}} W C)$ is ever materialized; the only dense
$O(n^2)$ object is the data-side `mixed`, which carries no inverse. The result
equals the direct linearization at $\mathrm{rtol} = 10^{-9}$, active bins
included.

### 7.4 The certificate that makes the split auditable (F-UNC-3)

$$
\sigma_{\text{total}} = \mathrm{hypot}\big(\sigma_{\text{stat}}, \sigma_{\text{syst}}\big),
$$

verified as
$`\sigma_{\text{total}}^2 = \sigma_{\text{stat}}^2 + \sigma_{\text{syst}}^2`$ to a
relative $10^{-9}$ in strict mode. Each `BandComponent` records its name, kind
(`stat`/`syst`) and formula ID, so a product can be audited without
re-deriving the split.

---

## 8. Calibration: a joint fit over datasets

### 8.1 The forward model (F-CAL-1)

For each dataset $d$, with the shared internal core
$`q = (c_0, k_1, k_2, k_3, b_0, b_1, b_2)`$ and a per-dataset scale
$`s_d(\mathrm{ch})`$:

$$
\text{prediction}_d = s_d \odot \big(C_{\mathrm{fit}}(q)\ \mathrm{mc}_d\big),
\qquad
\text{data}_d \approx \text{prediction}_d + \text{noise}.
$$

$`C_{\mathrm{fit}}`$ is evaluated on the **fixed** uniform deposition axis
$0..4096$ keV / 4096 bins (F-BIN-4) so its geometry does not depend on the
fitted parameters (D-79/D-101). The fit only contracts
$`C_{\mathrm{fit}}`$ with the MC spectrum and its variance, so the implementation
builds it on the contiguous hull of the non-zero MC bins — exactly the
corresponding sub-matrix of the fixed axis, because each column's kernel and
renormalization depend on that column alone. The export matrix is rebuilt
separately on the channel-derived axis $E(i \pm \tfrac12)$ (D-101).

Weights follow the frozen convention (D-48):

$$
\mathrm{var}_d = \max(\mathrm{stat}_d, 1) + \big(\mathrm{systFrac}_d \cdot \mathrm{data}_d\big)^2 + \mathrm{MC}_d,
\qquad
\mathrm{MC}_d = \left(s_d\sqrt{(C_{\mathrm{fit}}^2)\ \mathrm{varMC}_d}\right)^2.
$$

Here $`\mathrm{stat}_d`$ is the data variance `fSumw2`, $`\mathrm{systFrac}_d`$ the
per-dataset fractional systematic (`syst_frac`), $`\mathrm{varMC}_d`$ the MC
spectrum variance (`var_mc`), and $`\mathrm{MC}_d`$ the folded prediction's MC
variance.

The $\max(\mathrm{stat}, 1)$ floor is a **documented approximation**: for
Poisson data with $\text{pred} < 1$ the realized
$(\text{data} - \text{pred})^2 / \max(\text{data}, 1)$ has expectation below
$\text{pred}$, so $\chi^2/\mathrm{dof}$ can sit below one on low-count spectra.
The covariance is defined for the weights actually used.

### 8.2 Per-dataset quadratic Bezier scale (F-CAL-2)

The scale corrects the simulated/real normalization difference across a
dataset's fit window $`[x_{\mathrm{lo}}, x_{\mathrm{hi}}]`$. It is the quadratic
Bezier curve with control abscissae $`(x_{\mathrm{lo}}, s_0, x_{\mathrm{hi}})`$
and ordinates $`(s_1, s_2, s_3)`$:

$$
\begin{aligned}
x(t) &= x_{\mathrm{lo}} + 2(s_0 - x_{\mathrm{lo}})\ t + (x_{\mathrm{lo}} - 2s_0 + x_{\mathrm{hi}})\ t^2, \\
s(t) &= (1-t)^2 s_1 + 2(1-t)\ t\ s_2 + t^2 s_3, \\
t(x) &= \frac{d}{a + \sqrt{a^2 + cd}}, \qquad
a = s_0 - x_{\mathrm{lo}},\quad c = x_{\mathrm{lo}} - 2s_0 + x_{\mathrm{hi}},\quad d = x - x_{\mathrm{lo}}.
\end{aligned}
$$

$x(t)$ is strictly increasing for $`s_0`$ strictly inside the window, so $t$ is
the unique in-interval root; the rationalized form is exact at both endpoints
and free of catastrophic cancellation. The middle control **abscissa $`s_0`$ is a
free parameter** (D-103). For a constant scale ($`s_1 = s_2 = s_3`$) the
derivative with respect to $`s_0`$ vanishes identically, and at
$`s_0 = (x_{\mathrm{lo}} + x_{\mathrm{hi}})/2`$ the parametrization becomes
linear, collapsing the four-parameter family onto the three-parameter quadratic
(degree-2 Bernstein) subfamily — the $`s_0`$ direction is then an exact **gauge**.
So $`s_0`$ is kept free, the model is seeded with a non-constant scale to avoid
starting on the gauge plateau, and the scale block is marginalized stably
(§8.4). Derivatives:

$$
\frac{\partial s}{\partial s_0} = \frac{\mathrm{d}s}{\mathrm{d}t}\cdot\frac{-2t(1-t)}{\mathrm{d}x/\mathrm{d}t},
\qquad
\frac{\partial s}{\partial s_1} = (1-t)^2, \qquad
\frac{\partial s}{\partial s_2} = 2(1-t)t, \qquad
\frac{\partial s}{\partial s_3} = t^2.
$$

### 8.3 Analytic Jacobian and the exact chi-square gradient (F-CAL-4)

Differentiating the forward model, with
$`C_k = \partial C/\partial q_k`$ from F-RESP-4 chained into the internal basis
through the F-MODEL-2 Jacobian:

$$
\frac{\partial\ \text{prediction}_d}{\partial q_k} = s_d \odot \big(C_k[\text{window}]\ \mathrm{mc}_d\big),
\qquad
\frac{\partial\ \text{prediction}_d}{\partial s_p} = \frac{\partial s_d}{\partial s_p} \odot \big(C_{\mathrm{fit}}[\text{window}]\ \mathrm{mc}_d\big).
$$

The variance depends on the parameters, so the **exact** gradient carries an
extra term:

$$
\frac{\mathrm{d}\chi^2}{\mathrm{d}\theta} = -2 J^{\mathsf{T}} \frac{\text{data} - p}{v} - \left(\frac{\mathrm{d}v}{\mathrm{d}\theta}\right)^{\mathsf{T}} \frac{(\text{data} - p)^2}{v^2},
$$

$$
\frac{\partial v}{\partial q_k} = s_d^2\Big[2\ (C_{\mathrm{fit}} \cdot C_k)[\text{window}]\ \mathrm{varMC}_d\Big],
\qquad
\frac{\partial v}{\partial s_p} = 2 s_d \frac{\partial s_d}{\partial s_p}\Big[(C_{\mathrm{fit}}^2)[\text{window}]\ \mathrm{varMC}_d\Big].
$$

The optimizer then sees the residual
$r = (\text{data} - p)/\sigma$ and its Jacobian
$\mathrm{d}r/\mathrm{d}\theta = -J/\sigma - r\ (\mathrm{d}v/\mathrm{d}\theta)/(2v)$,
so its gradient is the exact gradient of the objective it minimizes. **No
finite differences enter production.** The fit itself is a single bounded
trust-region (reflective) least-squares stage via
`scipy.optimize.least_squares` with `x_scale="jac"`, started from the frozen
bounds and seeds of F-CAL-3.

### 8.4 Covariance with the scale marginalized (F-CAL-5)

With $J$ the full-parameter Jacobian, $W = \mathrm{diag}(1/v)$ (F-CAL-1)
and $F = J^{\mathsf{T}} W J$, split the parameters into the reported core $c$
and the scale $s$:

$$
\mathrm{cov}_{\text{core}}
  = \frac{\chi^2}{\mathrm{dof}}\ \Big(F_{cc} - F_{cs} F_{ss}^{+} F_{sc}\Big)^{-1},
\qquad
\mathrm{cov}_{\text{reported}} = T \mathrm{cov}_{\text{core}} T^{\mathsf{T}},
$$

with $`T = \mathrm{diag}\big(\texttt{internal\_jacobian}(\mathrm{ch}_{\max}), I_3\big)`$.

The Schur complement $`F_{cc} - F_{cs}F_{ss}^{+}F_{sc}`$ is the $(c,c)$ block of
$F^{-1}$, i.e. the scale **marginalized** rather than fixed; $`F_{ss}^{+}`$ is the
Moore-Penrose inverse, which projects out the $`s_0`$ gauge when the fitted scale
is (nearly) polynomial. Jacobi (diagonal) preconditioning is applied before the
factorization. $\chi^2/\mathrm{dof}$ is the single global scale (PDG convention,
F-COV-2): $\mathrm{cov} = s^2 F^{-1}$ with $s^2 = \chi^2/\mathrm{dof}$; a
singular or non-positive-definite Fisher matrix is a **hard failure** — there is
no pseudo-inverse fallback for the core. For invertible $`F_{ss}`$ this is
algebraically identical to taking the core block of the full inverse, which the
tests verify on a well-conditioned Fisher. The estimator string recorded with
the product is `fisher-x2dof-marginalized-reported`.

An optional **profile-covariance diagnostic** (F-COV-3) profiles each parameter
at $`p_i \pm 4\sqrt{2/H_{ii}}`$, re-optimizes the rest, solves the
$\Delta\chi^2 = 1$ crossing with `brentq`, and takes correlations from the
numerical Hessian. It never replaces the analytic estimate.

---

## 9. Simulation: sampling, exact variances and seeding

### 9.1 Fixed per-column sampling and event accounting (F-SIM-1)

The matrix primary axis has $`n_{\text{active}}`$ active columns.
$`n_{\text{events}}`$ is split as
$`\text{base},\  \text{rem} = \mathrm{divmod}(n_{\text{events}}, n_{\text{active}})`$;
the active columns receive $\text{base} + 1$ for the first `rem` and `base`
otherwise — exactly the round-robin assignment
$`\text{active}[(\text{offset} + \text{event}) \bmod n_{\text{active}}]`$ written
as a count vector. Within column $j$, each primary energy is drawn uniformly in
its bin,

$$
E = \mathrm{lo}_j + u\ (\mathrm{hi}_j - \mathrm{lo}_j), \qquad u \sim U[0, 1).
$$

Every event is scored into exactly one cell: either an in-range positive
deposit fills one $(\text{deposition}, \text{primary})$ cell of $G$, or the
event is counted in $`\mathrm{zero}_j`$. Hence the accounting identity holds
exactly by construction:

$$
\sum_d G[d,j] + \mathrm{zero}_j = N_j
\qquad\text{and}\qquad
\sum_j N_j = n_{\text{events}}.
$$

The scoring rule is $0 < \mathrm{totalKev} < \mathrm{high}$ **and**
$\mathrm{totalKev} \ge \mathrm{low}$ for the in-range branch (exactly as
implemented), so an event whose total deposit falls outside the deposition axis
is *routed to* $`\mathrm{zero}_j`$ rather than lost. The identity therefore
cannot detect an out-of-range deposit — it holds by construction, and
$`\mathrm{zero}_j`$ is "no in-range deposit", not "nothing deposited" (§4.1).
What the identity does guarantee is that no event is dropped, double-counted or
invented.

### 9.2 Exact fixed-total variance (F-SIM-2)

Conditional on the fixed column total $`N_j`$, the deposition-bin counts are
multinomial with probabilities $`p = G[d,j]/N_j`$, so the binomial marginal

$$
\mathrm{Var}\big(G[d,j]\big) = N_j\ p\ (1 - p)
$$

is stored in `fSumw2`. Within a column the bins are **negatively correlated**
and those correlations are reconstructed downstream from counts and $`N_j`$. A
Poisson `counts` variance would overstate the high-probability bins; this is
why the exact fixed-total form is used. The source mode fills one entry per
merged pulse, so the pulse total $P$ is itself random; conditional on the
recorded $P = \sum \text{counts}$, the plug-in marginal is
$\mathrm{Var}(c) = c\ (1 - c/P)$ (documented approximation to the full
pulse-count distribution).

### 9.3 Lambertian surface sampling (F-SIM-4)

**Plane.** Position uniform on the crystal-face-sized square,

$$
x = (2u_x - 1)h_x, \qquad y = (2u_y - 1)h_y, \qquad z = z_{\text{plane}}.
$$

Direction Lambertian about the inward normal $-z$:
$`\cos\theta = \sqrt{u_{\cos}}`$ (the correct cosine-weighted law, not
$\cos\theta = u$), $`\varphi = 2\pi u_\varphi`$, giving the unit vector
$(-\sin\theta\cos\varphi,\  -\sin\theta\sin\varphi,\  -\cos\theta)$.

**Sphere.** A uniform point uses $`\cos\theta_0 = 2u_1 - 1`$,
$`\varphi_0 = 2\pi u_2`$; a local orthonormal frame
$`e_1 = n \times \hat{z}`$ (falling back to $(1,0,0)$ at the poles) and
$`e_2 = n \times e_1`$ carries the same Lambertian direction. The result is a unit
vector with positive projection on the inward normal.

### 9.4 Pulse merging and deterministic seeds (F-SIM-5/F-SIM-7)

Crystal deposits carry their Geant4 global time; sorting by time, a pulse is the
group accumulated while $`t \le t_0 + 10\ \mu\mathrm{s}`$, i.e. the **closed**
window $`[t_0,\  t_0 + 10\ \mu\mathrm{s}]`$ (the boundary deposit at exactly
$`t_0 + 10\ \mu\mathrm{s}`$ is included). Its energy is the sum of the group's
deposits. Only the source mode merges — the matrix mode scores one gamma per
event.

Each matrix primary column gets an independent stream from a SplitMix64 mixer
of $(\text{seed} + \text{tag})$ and the column index, reduced to $[0, 2^{63})$;
source-mode event blocks use the same construction with a different tag. For a
fixed seed and a fixed worker partition, merging the summed histograms
reproduces the invocation result **bit-for-bit** (counts are integer-valued).
Bit-for-bit equivalence across *different* worker counts is explicitly **not** a
contract: Geant4's engine state makes a mid-run reseed process-dependent
(D-123).

### 9.5 Physical boundary certificate (F-SIM-6)

A gamma cannot deposit more energy than it carries, so on the binned axes a
necessary condition for every **scored** cell is $G[d,j] = 0$ whenever
$\mathrm{depositionEdges}[d] > \mathrm{primaryEdges}[j+1]$ (the deposition bin
starts above the primary column's top edge). The check is one vectorized mask
and runs in strict mode.

Its scope is bounded by the routing rule of §9.1: an out-of-range total never
reaches a $G$ cell at all, so this certificate proves that the cells that *were*
scored respect energy conservation — it is **not** a detector of escaped
deposits. Those are absorbed into $`\mathrm{zero}_j`$ by construction, which is
why the $`\eta_j`$ of F-SIM-3 must be read as the in-range containment fraction.

---

## 10. Why the formulas cannot drift

The mathematical claims above are only credible if the code cannot quietly
diverge from them. Four mechanisms enforce that:

1. **Symbolic generation (D-77).** `tools/generate_kernels.py` renders the
   energy model, resolution model, kernel and all their derivatives from
   `sympy.diff` into `kc761/core/_gen/`, with a header recording the sympy
   version, the formula IDs and the exact command line. **Value and derivative
   always come from the same symbolic expression**; a hand-written derivative
   of a registered formula is a contract violation. The generator writes no
   timestamps, so repeated runs are byte-identical, and `_gen` verifies a
   manifest at import time.
2. **Mechanical single-source gate.** `tools/check_single_source.py` fails the
   build when a registered expression body appears outside its `_gen` module or
   its owning implementation.
3. **Runtime certificates (D-61).** Section 11 lists the suites. They fail fast
   with the offending formula ID and the offending value.
4. **No silent failures (AGENTS.md hard rule 12, plan §5.5).** No bare
   `except`, no silent clamping, no NaN substitution. A numerically necessary
   clamp must be explicit, warned about, and recorded in the product `meta` —
   as the F-MODEL-5 resolution clamp is.

**Tests are auxiliary.** Correctness is defined by the derivations and the
certificates, never by stored reference outputs, golden files or regression
baselines (D-65). pytest/hypothesis encode invariants; they do not define them.

---

## 11. Runtime certificates

Strict mode (`--strict` or `KC761_STRICT=1`) runs every suite below and aborts
on violation. Always-on schema, shape, unit and finiteness validation runs in
both modes and cannot be disabled (D-62).

| Certificate | Formula ID | Check |
|-------------|------------|-------|
| Energy monotonicity | F-MODEL-3 | $\mathrm{d}E/\mathrm{d}\mathrm{ch} > 0$ exactly at endpoints or the quadratic vertex |
| Resolution positivity | F-MODEL-5 | $\sigma^2 \ge -10^{-9}$ on the export grid |
| Kernel column sums | F-KERN-2 | every non-empty column sums to $1$ within $10^{-10}$ |
| Response columns | F-RESP-1 | every $C$ column sums to $1$ or is exactly $0$ |
| Composition identity | F-RESP-2 | $R$ column sums equal detected mass $`/N_j`$ (the $\eta \in [0,1]$ bound is always-on) |
| Slice integrity | F-RESP-3 | sliced row sums never exceed full column sums |
| KKT | F-SOLVE-3 | scaled reduced gradient and complementarity $\le 10^{-6}$ |
| Mask reproducibility | F-SOLVE-6 | stored mask equals the mask recomputed from the recorded spectrum and settings |
| Covariance PSD | F-COV-1/2 | symmetric, smallest eigenvalue $\ge -\mathrm{tol}$ |
| Band decomposition | F-UNC-3 | $`\sigma_{\text{total}}^2 = \sigma_{\text{stat}}^2 + \sigma_{\text{syst}}^2`$ within $10^{-9}$ relative |
| Event accounting | F-SIM-1 | $`\sum \text{counts} + \mathrm{zero} = N_j`$; $`\sum_j N_j = n_{\text{events}}`$ |
| Simulation variance | F-SIM-2 | `fSumw2` $`= N_j\ p\ (1-p)`$ (matrix) or $c\ (1-c/P)$ (source) |
| Efficiency bounds | F-SIM-3 | derived $`\eta_j = \text{column sum}_j / N_j`$ lies in $[0,1]$ |
| Physical boundary | F-SIM-6 | impossible deposition cells are exactly $0$ (strict only) |
| Finiteness | all | no NaN/Inf; shape and unit checks in every mode |

---

## 12. Running

```bash
python kc761.py --help
python -m kc761 --help
```

| Subcommand | Mathematical role |
|------------|-------------------|
| `csv2root` | parse a raw spectrometer CSV into a `spectrum` product |
| `subbkg` | scale by DAQ time and subtract a background spectrum |
| `sim` | Geant4 source mode (`mc_spectrum`) or matrix mode (`G`) |
| `calib` | joint fit of $`(c_0 \dots c_3, b_0 \dots b_2)`$ and the per-dataset Bezier scales |
| `compose` | compose $R = C\ G\mathrm{diag}(1/N)$ for inspection |
| `unfold` | solve the non-negative Tikhonov problem (full, or `--calib-only`) |

The chain is `csv2root -> subbkg -> sim` (source) `-> calib -> sim` (matrix)
`-> compose -> unfold`: the calibration consumes measured and source-mode
simulated spectra, and the matrix simulation and unfolding consume the
calibration product that supplies their energy axes.

`--alpha` is required for a full unfold (D-45); it has no default. The SNIP
peak mask is on by default and can be disabled with `--no-snip` or tuned with
the `--snip-*` flags (§6). Calibration prints a pre-fit summary and a progress
line about once per second (`--no-progress` silences them,
`--progress-every SECONDS` retunes). Every product-writing command validates
its output and figure targets before starting work, so an existing file is
refused up front rather than after a long run (D-171).

```bash
python kc761.py unfold --strict ...        # every certificate
KC761_STRICT=1 python kc761.py unfold ...  # same, via the environment
```

## 13. Configuration files

`sim`, `calib`, `compose` and `unfold` accept `-c/--config FILE`, a TOML file
read with the standard library (`config_version = 1`). One file may hold the
`[sim]`, `[calib]`, `[compose]` and `[unfold]` tables; each subcommand reads
only its own table. `[sim]` is a serial batch of `[[sim.runs]]`, each executed
in a fresh child process because a `G4RunManager` can be initialized only once
per process. Relative paths resolve against the current working directory
(D-164).

See [examples/](examples/) for a commented file per subcommand, and
[docs/plan.md](docs/plan.md) section 1.12 for the full rules.

## 14. Requirements

* Python >= 3.12.
* Runtime: `numpy`, `scipy`, `numba`, `uproot`, `sympy`, `matplotlib`.
  `numba` JIT-compiles the generated response kernel and the fused Jacobian
  pass (D-174); the number of parallel threads follows numba's standard
  `NUMBA_NUM_THREADS`. Kernel fills write disjoint slices per column, so
  results are thread-count independent.
* Simulation: `geant4-pybind`.
* Development: `ruff`, `pytest`, `hypothesis`.

Packaging is intentionally not provided (D-7); install the dependencies in a
virtual environment and run from the repository root.

## 15. Data and outputs

* Raw measurements live in `work/data/<campaign>/` (for example
  `work/data/2609a/`); `work/` is not committed.
* Default products are written under `work/<subcommand>/`; `csv2root` and
  `subbkg` default next to their input file (D-165) and `compose` next to its
  `--sim` input (D-166).
* Writes are atomic and self-validating (F-IO-1): the product goes to
  `<target>.part`, is closed, reopened and validated against the product
  contract, and only then moved onto `<target>` with `os.replace`. An existing
  target is refused unless `--force` is passed, and any failure removes the
  part and leaves an existing target untouched.
* Every product records full provenance: git revision and dirty flag, Python
  and dependency versions, the sha256 of every input file, the ordered CLI
  arguments and a UTC timestamp.

## 16. Development checks

```bash
ruff check .
pytest -q -m "not g4 and not root"        # 356 passed, 1 bench case skipped
pytest -q tests/test_sim_g4.py            # needs geant4-pybind
python tools/check_single_source.py       # the single-source gate
python tools/generate_kernels.py          # regenerate kc761/core/_gen (committed)
python tools/benchmarks.py --scenario all # wall-clock, never a gate
python kc761.py sim --dry-run ...         # print the resolved run, no side effects
```

`g4`- and `root`-marked tests are skipped when the corresponding framework is
unavailable, and `bench`-marked cases only run with `KC761_RUN_BENCH=1`. The
performance crossovers and their measured before/after numbers are registered
in [docs/plan.md](docs/plan.md) section 1.16.

For reference, the verification commands CI runs are:

```bash
ruff check .
pytest -q -m "not g4 and not root"
python tools/check_single_source.py
python kc761.py --help
python -m kc761 --help
for c in calib compose sim unfold; do python kc761.py "$c" -c "examples/$c.toml" --dry-run; done
```

## 17. Repository layout

| Path | Content |
|------|---------|
| `kc761.py`, `kc761/__main__.py` | entry points |
| `kc761/core/model.py` | energy calibration (dual basis), resolution, certificates (F-MODEL-1..5) |
| `kc761/core/binning.py` | channel/energy grids, working window and pad, fixed MC axis (F-BIN-1..4) |
| `kc761/core/kernel.py` | exact Gaussian bin integrals, taper, sparse assembly (F-KERN-1..4) |
| `kc761/core/response.py` | `C`, `R`, window slicing, response Jacobian (F-RESP-1..4) |
| `kc761/core/solver.py` | Tikhonov objective, SNIP mask, active-set QP, KKT (F-SOLVE-1..6) |
| `kc761/core/covariance.py` | Fisher information, `s^2` scaling, profile diagnostic (F-COV-1..3) |
| `kc761/core/uncertainty.py` | stat/syst propagation, streaming MC term, bands (F-UNC-1..3) |
| `kc761/core/projection.py` | overlap projections and variance propagation (F-PROJ-1..2) |
| `kc761/core/_linalg.py` | shared SPD factorization policy (dense/banded/sparse crossovers) |
| `kc761/core/_gen/` | committed sympy-generated kernels, manifest and freshness check |
| `kc761/schema/` | product contracts, axes, uproot IO and certificates (F-IO-1) |
| `kc761/calib/` | calibration fit, Bezier scale, covariance, product export (F-CAL-1..5) |
| `kc761/unfold/` | selection, compose, solve, orchestration (F-UNF-1..6) |
| `kc761/sim/` | Geant4 detector, sources, sampling, certificates (F-SIM-1..7) |
| `kc761/*/plot.py` | self-contained figures (D-153) |
| `kc761/cli/` | CLI, config-file mode and per-command wiring |
| `tools/` | kernel generation, the single-source gate and the timing tool |
| `examples/` | shipped TOML configuration examples |
| `tests/` | auxiliary tests and deterministic fixtures |
| `docs/` | plan, architecture, formats, derivations |
| `AGENTS.md` | hard rules, file ownership, contract-change process |

## 18. Documentation

* [docs/derivations.md](docs/derivations.md) - **the mathematics**: formula
  registry, full derivations, stated limitations, certificates.
* [docs/plan.md](docs/plan.md) - final plan, decision register (D-nnn) and
  contract points; section 4 summarizes the mathematics and statistics.
* [docs/architecture.md](docs/architecture.md) - layering and module map.
* [docs/formats.md](docs/formats.md) - product schemas and units.
* [AGENTS.md](AGENTS.md) - hard rules, file ownership, contract-change process.
