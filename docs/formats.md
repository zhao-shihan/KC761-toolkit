# Product formats

Status: **W2 implemented**. The container names and the `meta` field list below
are the frozen contract implemented by `kc761/schema/products.py` and
`kc761/schema/io.py`; the low-level primitives in `kc761/schema/_uproot.py`
were verified by the W0 spike (section 6). Object names and units are single
sourced in `products.py` / `axes.py` (AGENTS hard rule 11).

## 1. Envelope

Every product file contains its data objects plus exactly one `meta` RNTuple.
Files are written to `<target>.part`, closed, reopened and validated object by
object, then renamed onto `<target>`. An existing target is refused unless
`--force` is given.

`format_version` starts at 1 for the new schema (independent of the legacy
versioning; see plan Appendix A item 7). Readers must reject any other value
with `SchemaError`.

## 2. Axes, units and orientation

* Channel axis: uniform bins of width 1, edges `-0.5 .. n - 0.5`, unit
  `channel`.
* Energy axes: strictly increasing variable edges in `keV`, unit `kev`.
* Matrix orientation (D-20): `x = output side`, `y = input side`.
  * `deposition_to_channel`: x channel, y deposition energy.
  * `primary_to_deposition`: x deposition energy, y primary energy.
  * `response_matrix`: x channel, y primary energy.
* Units are part of key names (`energy_kev`, `daq_time_s`, `z_mm`); matrices
  carry units in their human-readable axis titles as well.
* No unit conversion happens in the schema layer: arrays are stored in the
  units named by their keys.

## 3. Objects per product

| Product | Objects |
|---------|---------|
| calib | `deposition_to_channel` TH2D; `param_cov` TH2D 7x7 with axis labels `c0 c1 c2 c3 b0 b1 b2`; `meta` |
| sim | `primary_to_deposition` TH2D (counts); `primary_column_totals` TH1D (N_j); `meta` |
| compose | `response_matrix` TH2D; `deposition_to_channel`; `primary_to_deposition`; `primary_column_totals`; `primary_efficiency` TH1D; `meta` |
| unfold | `kc761_spectrum_unfolded`; `sigma_statistical`; `sigma_systematic`; `sigma_total`; `kc761_spectrum_refolded`; `meta` |
| unfold (calib-only) | `kc761_spectrum_calibrated`; `meta` (with `mode = calib_only`) |
| spectrum | `kc761_spectrum` TH1D; `meta` |
| mc_spectrum | `kc761_mc_spectrum` TH1D; `meta` |

Notes:

* C columns sum to 1 or are exactly 0 (F-RESP-1); per-element C sigma is not
  stored and is recomputed from `param_cov` and the analytic Jacobian when
  needed. `param_cov` is a 7x7 TH2D whose two axes carry the bin labels
  `c0 c1 c2 c3 b0 b1 b2` (from `PARAM_NAMES_REPORTED`).
* Simulation counts carry the exact binomial variance in `fSumw2`:
  `N_j p (1 - p)` with `p = counts_j / N_j`. Row/column correlations are
  reconstructed downstream from counts and `N_j`.
* Detection efficiency is derived: `eta_j = column_sum_j / N_j`. It is not
  stored in the sim product; the compose product stores it as
  `primary_efficiency`. `R` full-axis column sums equal the reachable
  detected mass/N (F-RESP-2), so they equal `eta` only when no deposition
  column has an exactly-zero C column (F-RESP-1 allows one; D-99).
* sigma bands: `content = sigma`, `fSumw2 = sigma**2` (D-15), so `values()`
  and `errors()` both return the 1-sigma band.
* `mc_spectrum` is the source-mode simulated pulse spectrum (D-120): a TH1D on
  the fixed uniform energy axis `0..4096 keV / 4096 bins`
  (`core.binning.source_mode_deposition_edges_kev()`), with the required
  conditional-binomial `fSumw2 = c (1 - c/P)` where `P = sum(counts)` is the
  recorded pulse total (D-128). It is a distinct product kind from the measured
  `spectrum`; the CLI connects `calib --mc` to it (D-144).
* Required variance buffers (always-on check, D-98): `primary_to_deposition`,
  `sigma_statistical`, `sigma_systematic`, `sigma_total`, `kc761_spectrum` and
  `kc761_mc_spectrum`. All other variance buffers are optional.

## 4. `meta` RNTuple

All fields are length-1 scalars or strings (one entry). The object is written
with `file.mkrntuple("meta", {field: np.array([value])})`, pinning the RNTuple
type explicitly (D-12 as revised; uproot's dict assignment already defaults to
RNTuple). Native ROOT reading requires ROOT >= 6.30; uproot is unaffected.
`keys()` returns fields in name-sorted order, so always look up meta fields by
name, never by position. The reader requires the field set to match exactly:
no missing and no extra field (`SchemaError` otherwise). The complete list:

* Common (`product_kind` dispatches the reader; W2 D-87): `format_version`
  (int, currently 1), `product_kind` (str), `producer` (str), `created_utc`
  (str), `git_revision` (str), `git_dirty` (int 0/1), `python_version` (str),
  `dependency_versions` (str, JSON list of `[name, version]` in the frozen
  dependency order), `command` (str), `arguments_json` (str, JSON ordered list
  of `[name, value]`), `inputs_json` (str, JSON list of
  `{"path": ..., "sha256": ...}`).
* calib: `channel_max` (float), `params_reported_json` (str, JSON list of 4 in
  the `c0..c3` order), `resol_params_json` (str, JSON list of 3 in the
  `b0..b2` order), `chi2` (float), `dof` (int), `covariance_scale` (float),
  `fit_status` (str, `converged` or `stopped-early`), `scales_json` (str, JSON
  list of `[label, [s0, s1, s2, s3]]`), `scale_bound_flags_json` (str, JSON
  list of `[label, [f0, f1, f2, f3]]` with 0/1 flags marking scale parameters
  sitting on a fit bound, e.g. the `s0` polynomial-stratum limit; D-103),
  `resol_clamp_count` (int),
  `resol_clamp_energy_low_kev`, `resol_clamp_energy_high_kev` (float; both 0
  when the count is 0). These are the W3 fit diagnostics of D-49/D-105/D-106.
* sim: `mode` (int), `mode_name` (str), `geometry_name` (str),
  `geometry_param_mm` (float), `angular_distribution` (str), `seed` (int),
  `n_events` (int), `workers` (int). There is no per-file `calib_sha256`: the
  calibration input is recorded in `inputs_json`.
* compose: only the common fields. The calibration and simulation inputs and
  their sha256 digests live in `inputs_json`.
* unfold: `mode` (str, `unfold` or `calib_only`), `alpha` (float),
  `difference_order` (int), `energy_low_kev`, `energy_high_kev` (float),
  `channel_low`, `channel_high` (int), `pad_nsigma` (float), `syst_frac`
  (float), then the SNIP mask settings `snip_enabled` (int 0/1, default 1,
  D-154), `snip_threshold_sigma` (float, default 5.0), `snip_protect_sigma`
  (float, default 2.0), `snip_floor` (float, default 0.1), `snip_iterations`
  (int, resolution-derived per run, D-157), `snip_max_iterations` (int,
  default 8), `snip_clipped_bins` (int), `snip_clipped_index_range` (str),
  `snip_baseline_sha256` (str), `snip_mask_sha256` (str), and finally the
  diagnostics `chi2` (float), `dof` (int), `covariance_scale` (float). The
  `calib_only` variant carries only the common fields plus `mode`. Per D-118,
  `chi2` is the weighted residual sum of squares over the solver rows,
  `dof = n_fit_rows - n_active` (the number of strictly positive solution bins)
  may be zero or negative for a heavily regularized problem, and
  `covariance_scale` is fixed to `1.0` (the analytic F-UNC-1/F-UNC-2 bands are
  never rescaled). When the SNIP mask is enabled the reported bands are
  conditional on the realised mask (D-159) and the baseline/mask hashes pin the
  exact solve.
* spectrum: `daq_time_s` (float), `source_file` (str).
* mc_spectrum: `source_key` (str), `mode_name` (str), `geometry_name` (str),
  `geometry_param_mm` (float), `n_events` (int), `seed` (int), `workers` (int)
  (D-120).

The former `calib_sha256`/`sim_sha256` fields are removed (D-89). Every input
fingerprint is in `inputs_json`; `kc761/schema/io.py` exposes `fingerprint_for`
and `input_sha256` to look a digest up by path.

Array-valued metadata (for example per-column information) must use a
dedicated histogram object, never a multi-entry RNTuple field (all fields of
one RNTuple must have equal length).

## 5. Provenance

Every product records (D-18): git revision and dirty flag, Python version,
versions of the dependencies actually used, sha256 of every input file, the
full command line, and a UTC timestamp. Provenance is written into the `meta`
RNTuple; `kc761/schema/io.py` assembles it in one place. A non-git working
directory records `git_revision = "unknown"` and `git_dirty = 0` with one
warning and does not abort (D-93).

## 6. uproot spike results (W0)

Verified with `uproot 5.7.6`, `numpy 2.5.3`, Python 3.14.7 (the code targets
>= 3.12). Results are frozen by `tests/test_uproot_spike.py`:

1. `file[name] = (values, edges)` writes a TH1D **without** a variance array;
   `(values, xedges, yedges)` writes a TH2D without variance. A trailing title
   string is accepted.
2. There is **no tuple form that carries `fSumw2`**. `(values, edges,
   variances)` raises `TypeError`, and a four-element 2D form is interpreted as
   a 3D histogram.
3. `fSumw2` works through the model constructors
   `uproot.writing.identify.to_TH1x` / `to_TH2x` (with `to_TAxis` for the
   axes). `data` and `fSumw2` must use the same flow-padded layout; for TH2D
   the layout is the transposed, flattened array
   (`with_flow.T.reshape(-1)`). `kc761/schema/_uproot.py` encapsulates this.
4. Reading: `values()` excludes flow bins; `errors()` returns
   `sqrt(fSumw2)` when the object has a variance buffer. `member("fSumw2")`
   never raises; use `len(hist.member("fSumw2")) > 0` to detect the buffer.
5. `TParameter`, `TNamed` and `TMatrixDSym` have no writable uproot model. All
   scalar/string metadata therefore lives in the `meta` RNTuple; covariance
   matrices are TH2D objects.
6. Metadata writing uses `file.mkrntuple(name, {field: np.array([value])})`.
   Assigning a dict to `file[name]` also writes an RNTuple in uproot 5.7.6,
   but the explicit call pins the type in case uproot changes its default
   (D-12 as revised). Iterating an RNTuple yields RField objects, so field
   names come from `keys()`; `tree[field].array()` reads the values and
   length-1 arrays are unwrapped. Native ROOT reading of the `meta` object
   requires ROOT >= 6.30 (uproot is unaffected).
7. TAxis bin labels are writable through
   `uproot.writing.identify.to_THashList` / `to_TObjString`; `param_cov` uses
   them for `c0 c1 c2 c3 b0 b1 b2`. The canonical axis name travels in the axis
   `fName` and its unit is derived from that name through
   `kc761.schema.axes.AXIS_UNITS`; the axis `fTitle` is a human-readable display
   label (`Energy (keV)`, `Channel`) and is never parsed (D-170). ROOT object
   titles are human-readable as well (`HUMAN_TITLES`). **Reading variances uses the raw `fSumw2`
   buffer, not `errors()**2`, so `write -> read` is bit exact.**

Constraints respected in W2: `meta` fields all have length 1; variable length
data uses separate histogram objects; the atomic-write protocol reopens with
uproot before renaming; the object set on disk must equal `OBJECT_NAMES` for
the product kind exactly; duplicate on-disk object versions are rejected.

## 7. CLI outputs, CSV import and background subtraction (W6)

### 7.1 Command surface and exit codes

`kc761` has six subcommands (`calib`, `unfold`, `sim`, `compose`, `csv2root`,
`subbkg`; D-66). Exit codes are 0 success, 1 runtime failure, 2 usage error
(D-68); every failure is logged as `[kc761.<command>] error: ...` (D-67).
`sim`, `calib`, `compose` and `unfold` additionally accept `-c/--config FILE`
and `--dry-run`; `csv2root` and `subbkg` take no config file (D-129).

### 7.2 Default outputs (D-19/D-138)

When `-o/--output` is omitted the product is written under `work/<command>`
(except `csv2root`/`subbkg`, D-165, and `compose`, D-166, which default next
to their input):

| Command | Default file name |
|---------|-------------------|
| `sim` source | `<source>-n<N>-s<SEED>.root` |
| `sim` matrix | `<calib-stem>-<mode>-n<N>-s<SEED>.root` (`mode` is `plane-front-gamma` or `sphere-gamma`) |
| `calib` | `calib-<label>[-<label>...].root` |
| `compose` | `compose-<calib-stem>-<sim-stem>.root` |
| `unfold` | `unfold-<data-stem>-<sim-stem>-a<alpha>.root` |
| `unfold --calib-only` | `unfold-<data-stem>-calibonly.root` |
| `csv2root` | `<input-stem>.root` |
| `subbkg` | `<signal-stem>-subbkg.root` |

An existing target is refused unless `--force` is given (D-17).

### 7.3 `csv2root` strict grammar

The parser is strict (D-72; resolves Appendix A item 8). The input is UTF-8
(an optional BOM is accepted).

* The first non-empty line is the header. It must match
  `Channel,Count #<D>d<H>h<M>m<S>s` (for example
  `Channel,Count #0d0h2m35s`).
* `D`, `H` and `M` are non-negative integers and `S` is a non-negative decimal
  number; `H <= 23`, `M <= 59`, `S < 60`, and the total acquisition time must
  be positive. It is stored as `daq_time_s = D*86400 + H*3600 + M*60 + S`.
* Each data line has exactly three comma-separated fields, `channel,count,`,
  where the third field is empty. `channel` and `count` are non-negative
  integers; channels start at 0 and increase by exactly 1; blank lines are
  skipped; at least one data row is required.
* Malformed/empty input, an invalid header or time, a wrong column count, a
  negative or non-integer value, and a duplicated/non-monotonic/non-contiguous
  channel are errors (exit code 1).

The output is a `spectrum` product: a uniform `channel` axis (`-0.5 ..
n - 0.5`), `values = counts`, `fSumw2 = counts` (Poisson), `daq_time_s` from
the header and `source_file` set to the input path.

### 7.4 `subbkg` semantics

The net spectrum is `S - r B` with `r = t_sig / t_bkg` (both `daq_time_s`
must be positive). Each input bin error is floored at one count before the
combination, and the net `fSumw2` is `sig_err**2 + r**2 * bkg_err**2`. The
signal and background channel axes must match bitwise (unit and edges). The
output is a `spectrum` product that inherits the signal `daq_time_s` and names
the signal as `source_file`; provenance lists both inputs.

