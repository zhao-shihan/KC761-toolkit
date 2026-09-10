# KC761 toolkit (rewrite)

Simulation, calibration and unfolding toolkit for the MEASALL KC761x/KC761
gamma spectrometer.

**Status: W0 contract skeleton.** The rewrite is specified in
[docs/plan.md](docs/plan.md). The frozen signatures live under `kc761/` and
raise `NotImplementedError` until their workstream (W1-W6) lands. The legacy
packages (`kc761calib/`, `kc761sim/`, `kc761unfold/`, `kc761util/`, `app/`)
are untouched until W7 and are not part of the new contract.

## Requirements

* Python >= 3.12.
* Core contract: `numpy`, `scipy`, `uproot`.
* Later workstreams: `numba` and `sympy` (generated kernels, W1),
  `matplotlib` (plots, W3/W4), `geant4-pybind` (simulation, W5).
* Development: `ruff`, `pytest`, `hypothesis`.

## Running

Packaging is intentionally not provided (decision D-7). The two supported
entry points are:

```bash
python kc761.py --help
python -m kc761 --help
```

Subcommands: `calib`, `unfold`, `sim`, `compose`, `csv2root`, `subbkg`.

Strict mode (runtime certificates, decision D-61) is enabled per invocation:

```bash
python kc761.py unfold --strict ...
KC761_STRICT=1 python kc761.py unfold ...
```

## Development checks

```bash
ruff check .
pytest -m "not g4 and not root"
```

## Repository layout

| Path | Content |
|------|---------|
| `kc761.py`, `kc761/__main__.py` | entry points |
| `kc761/core/` | pure numerics: models, binning, kernels, response, solver, covariance |
| `kc761/schema/` | product contracts, axes, uproot IO |
| `kc761/calib/`, `kc761/unfold/`, `kc761/sim/` | workstream packages (W3/W4/W5) |
| `kc761/plotting/`, `kc761/cli/` | shared plotting and CLI |
| `tests/` | auxiliary tests and deterministic fixtures |
| `docs/` | plan, architecture, formats, derivations |
| `AGENTS.md` | hard rules, file ownership, contract-change process |

## Documentation

* [docs/plan.md](docs/plan.md) - frozen rewrite plan and decision register.
* [docs/architecture.md](docs/architecture.md) - layering and module map.
* [docs/formats.md](docs/formats.md) - product schemas and uproot spike results.
* [docs/derivations.md](docs/derivations.md) - formula ID registry and derivations.
