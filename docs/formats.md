# Product formats

Status: W0 contract skeleton. The functions in `kc761/schema/io.py` that
implement this document are stubs until W2; the low-level primitives in
`kc761/schema/_uproot.py` are implemented and verified (spike, section 6).

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
  carry units in their axis titles as well.
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

Notes:

* C columns sum to 1 by construction; per-element C sigma is not stored and is
  recomputed from `param_cov` and the analytic Jacobian when needed.
* Simulation counts carry the exact binomial variance in `fSumw2`:
  `N_j p (1 - p)` with `p = counts_j / N_j`. Row/column correlations are
  reconstructed downstream from counts and `N_j`.
* Detection efficiency is derived: `eta_j = column_sum_j / N_j`. It is not
  stored in the sim product; the compose product may store it as a derived
  convenience object.
* sigma bands: `content = sigma`, `fSumw2 = sigma**2` (D-15), so `values()`
  and `errors()` both return the 1-sigma band.

## 4. `meta` RNTuple

All fields are length-1 scalars or strings (one entry). The object is written
with `file.mkrntuple("meta", {field: np.array([value])})`, pinning the RNTuple
type explicitly (D-12 as revised; uproot's dict assignment already defaults to
RNTuple). Native ROOT reading requires ROOT >= 6.30; uproot is unaffected.
`keys()` returns fields in name-sorted order, so always look up meta fields by
name, never by position. Initial field list:

* Common: `format_version` (int), `producer` (str), `created_utc` (str),
  `git_revision` (str), `git_dirty` (int), `python_version` (str),
  `dependency_versions` (str, JSON), `command` (str),
  `arguments_json` (str), `inputs_json` (str, path + sha256 list).
* sim: `mode` (int), `mode_name` (str), `geometry_name` (str),
  `geometry_param_mm` (float), `angular_distribution` (str), `seed` (int),
  `n_events` (int), `workers` (int), `calib_sha256` (str).
* unfold: `alpha` (float), `difference_order` (int), `energy_low_kev`,
  `energy_high_kev`, `channel_low`, `channel_high`, `pad_nsigma`,
  `syst_frac`, `chi2`, `dof`, `covariance_scale`, `calib_sha256`,
  `sim_sha256`.
* spectrum: `daq_time_s` (float), `source_file` (str).

Array-valued metadata (for example per-column information) must use a
dedicated histogram object, never a multi-entry RNTuple field (all fields of
one RNTuple must have equal length).

## 5. Provenance

Every product records (D-18): git revision and dirty flag, Python version,
versions of the dependencies actually used, sha256 of every input file, the
full command line, and a UTC timestamp. Provenance is written into the `meta`
RNTuple; `kc761/schema/io.py` assembles it in one place.

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

Constraints to respect in W2: `meta` fields must all have length 1; variable
length data must be separate histogram objects; the atomic-write protocol must
reopen with uproot before renaming.
