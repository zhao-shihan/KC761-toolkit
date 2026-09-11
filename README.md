# KC761 toolkit

Simulation, calibration and unfolding toolkit for the MEASALL KC761x/KC761
gamma spectrometer. Pure-Python plus `geant4-pybind`: Geant4 source and matrix
simulations, an energy/resolution calibration with a statistical response
matrix, and non-negative regularized spectrum unfolding with propagated
uncertainties.

**Status: final.** The specification is [docs/plan.md](docs/plan.md), the
product schemas are in [docs/formats.md](docs/formats.md) and the formulas are
in [docs/derivations.md](docs/derivations.md). The repository contains the
`kc761/` implementation, the two entry points, tests, tools, examples and docs.

## Requirements

* Python >= 3.12.
* Runtime: `numpy`, `scipy`, `numba`, `uproot`, `sympy`, `matplotlib`.
  `numba` JIT-compiles the generated response kernel (D-174); the number of
  parallel threads follows numba's standard `NUMBA_NUM_THREADS`.
* Simulation: `geant4-pybind`.
* Development: `ruff`, `pytest`, `hypothesis`.

Packaging is intentionally not provided (D-7); install the dependencies in a
virtual environment and run from the repository root.

## Running

The two supported entry points are:

```bash
python kc761.py --help
python -m kc761 --help
```

| Subcommand | Purpose |
|------------|---------|
| `csv2root` | parse a raw spectrometer CSV into a `spectrum` product |
| `subbkg`   | scale and subtract a background spectrum |
| `sim`      | Geant4 source-mode (`mc_spectrum`) or matrix-mode (`sim`) simulation |
| `calib`    | fit energy/resolution and export the calibration product |
| `compose`  | compose the full response `R = C p diag(eta)` for inspection |
| `unfold`   | unfold a measured spectrum (full or `--calib-only`) |

A typical chain is `csv2root -> subbkg -> sim` (source mode) `-> calib ->
sim` (matrix mode) `-> compose -> unfold`: the calibration consumes measured
and source-mode simulated spectra, and the matrix simulation and unfolding
consume the calibration product that supplies their energy axes.

Calibration prints a pre-fit summary and a progress line about once per
second (`--no-progress` silences them, `--progress-every SECONDS` retunes).
Every product-writing command validates its output and figure targets before
starting work, so an existing file is refused up front rather than after a long
run (D-171).

Unfolding applies a default-on SNIP peak mask that relaxes the smoothing penalty
at resolved peaks while damping noise-induced structure (D-154); disable it with
`--no-snip` or tune it with the `--snip-*` flags.

Strict mode runs every runtime certificate (D-61) and fails fast:

```bash
python kc761.py unfold --strict ...
KC761_STRICT=1 python kc761.py unfold ...
```

## Configuration files

`sim`, `calib`, `compose` and `unfold` accept `-c/--config FILE`, a TOML file
read with the standard library (`config_version = 1`). One file may hold the
`[sim]`, `[calib]`, `[compose]` and `[unfold]` tables; each subcommand reads
only its own table. `[sim]` is a serial batch of `[[sim.runs]]`, each executed
in a fresh child process because a `G4RunManager` can be initialized only once
per process. Relative paths resolve against the current working directory (D-164).

See [examples/](examples/) for a commented file per subcommand, and
[docs/plan.md](docs/plan.md) section 1.12 for the full rules.

## Data and outputs

* Raw measurements live in `work/data/<campaign>/` (for example
  `work/data/2609a/`); `work/` is not committed.
* Default products are written under `work/<subcommand>/`; `csv2root` and
  `subbkg` default next to their input file (D-165) and `compose` next to its
  `--sim` input (D-166). Existing files are refused unless `--force` is passed,
  and writes are atomic.

## Development checks

```bash
ruff check .
pytest -q -m "not g4 and not root"      # 356 passed, 1 bench case skipped
pytest -q tests/test_sim_g4.py           # needs geant4-pybind
python tools/check_single_source.py
python tools/benchmarks.py --scenario all       # wall-clock, never a gate
python kc761.py sim --dry-run ...         # print the resolved run, no side effects
```

`g4`- and `root`-marked tests are skipped when the corresponding framework is
unavailable, and `bench`-marked cases only run with `KC761_RUN_BENCH=1`. Tests
are auxiliary: correctness is defined by the derivations and the runtime
certificates, not by stored reference outputs. The performance crossovers and
their measured before/after numbers are registered in
[docs/plan.md](docs/plan.md) section 1.16.

## Repository layout

| Path | Content |
|------|---------|
| `kc761.py`, `kc761/__main__.py` | entry points |
| `kc761/core/` | pure numerics: models, binning, kernels, response, solver, covariance, uncertainty |
| `kc761/schema/` | product contracts, axes, uproot IO and certificates |
| `kc761/calib/`, `kc761/unfold/`, `kc761/sim/` | calibration, unfolding and Geant4 packages |
| `kc761/*/plot.py` | self-contained figures (D-153) |
| `kc761/cli/` | CLI, config-file mode and per-command wiring |
| `tools/` | sympy kernel generation, the single-source gate and the `benchmarks.py` timing tool |
| `examples/` | shipped TOML configuration examples |
| `tests/` | auxiliary tests and deterministic fixtures |
| `docs/` | plan, architecture, formats, derivations |
| `AGENTS.md` | hard rules, file ownership, contract-change process |

## Documentation

* [docs/plan.md](docs/plan.md) - final plan, decision register and contract points.
* [docs/architecture.md](docs/architecture.md) - layering and module map.
* [docs/formats.md](docs/formats.md) - product schemas and uproot spike results.
* [docs/derivations.md](docs/derivations.md) - formula registry and derivations.
