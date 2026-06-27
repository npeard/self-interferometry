# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Use [Pixi](https://pixi.prefix.dev) for all development tasks. Pixi manages an
isolated Python interpreter and lockfile-backed dependencies under `.pixi/`, so
always invoke Python and tools through `pixi run` (or from inside `pixi shell`)
-- never use system Python or a separate venv.

- `pixi run all` - Run full pipeline (format, lint, ascii, typecheck, test)
- `pixi run format` - Format code with ruff and auto-fix issues
- `pixi run lint` - Check code style with ruff
- `pixi run ascii` - Fail on any non-ASCII character in `*.py`/`*.md`
- `pixi run typecheck` - Type-check with ty
- `pixi run test` - Run pytest test suite
- `pixi run spell` - Run codespell for spell checking
- `pixi run precommit` - Run all pre-commit hooks
- `pixi run -e test pytest` - Run pytest in the minimal `test` environment (CI-equivalent)

Environments: `default` (library), `dev` (= dev + test features; tooling), `test`
(CI-equivalent). `ruff`/`codespell` live in the default env so the
`format`/`lint`/`spell` tasks use the pinned, lockfile-backed tools (not
whatever is on PATH).

For installation and setup:

```bash
pixi install                  # Solve and materialize the default environment (editable install)
pixi run pre-commit install   # One-time: set up git hooks
```

Tasks are defined in `[tool.pixi.tasks]` in `pyproject.toml`; dependencies live
in `[tool.pixi.*]` feature/environment tables. The lockfile `pixi.lock` is
committed for reproducibility.

**NumPy is pinned `<2`** (in `[project.dependencies]` and `[tool.pixi.dependencies]`)
because this machine is locked to the torch 2.2 wheels (the last x86-64 macOS
build), which use the NumPy 1.x C-ABI; NumPy 2.x breaks torch and segfaults the
suite. If torch is ever upgraded to a NumPy-2 build, this pin can be relaxed.

**Important**: Always run `pixi run format` first when encountering linting/formatting issues before making manual edits. This auto-fixes most formatting problems and saves time.

## Package layout

The importable package is `smi/` (distribution name `smi`). Top-level repo
layout: `smi/` (library), `tests/`, `notebooks/` (marimo apps), `scripts/`
(`check_ascii.py`), `docs/`.

Inside `smi/`:

- `analysis/` - ML pipeline.
  - `features/` - Polars-based `FeatureRegistry` (`registry.py`), feature
    definitions (`features.py`; `register_feature` decorator auto-registers on
    the module-level `default_registry`), the stats script (`compute_norm_stats.py`),
    and the **generated, version-controlled** per-feature mean/std
    (`normalization.py` - do not hand-edit; regenerate with the script).
  - `models/` - architectures (`tcn`, `scnn`, `tcan`, `lstm`, `mamba`,
    `barland_cnn`) plus `base.py`, which defines `Model(nn.Module)` (the
    normalization wrapper) and `FeatureMap`.
  - `datamodule.py` - `VelocityDataModule` (LightningDataModule).
  - `datasets.py`, `lit_module.py`, `synthetic_lit_module.py`, `training_interface.py`.
- `synthetic/` - physics simulation (coil driver, interferometers, waveform);
  was `acquisition/simulations/`.
- `redpitaya/` - hardware control (manager, scpi, config); was
  `acquisition/redpitaya/`. Imports from `smi.synthetic`; import its manager via
  `from smi.redpitaya.manager import RedPitayaManager` (not re-exported from the
  package `__init__`, to avoid an init-time import cycle).

## Normalization & model wrapping

Normalization stats are NOT computed in the dataset. Instead:

1. `compute_norm_stats.py` points at an HDF5 dataset, evaluates the
   `FeatureRegistry` (inputs = photodiode channels; targets = velocity,
   displacement), and writes per-feature mean/std to `features/normalization.py`.
2. `Model(nn.Module)` wraps an inner architecture and bakes those stats into
   `register_buffer`s. Its `forward` is raw-units in / raw-units out: raw input
   -> `FeatureMap` (identity for now) -> input-normalize -> inner model ->
   output de-normalize. So the internal network sees zero-mean/unit-variance
   features while the rest of the codebase (loss, plotting) stays in raw physical
   units. `LitModule` does the wrapping; `create_model` still returns the bare
   inner model.

Models are TorchScript-scriptable (`Model.to_torchscript()`), verified by
`tests/test_torchscript.py`. `torch.compile` remains the training-speed path;
TorchScript is for packaging the trained model. HDF5 (gzip-4, chunked per shot)
is the chosen storage format - see `docs/data-storage-evaluation.md`.

## Hyperparameter / architecture search

Two-level search (design: `docs/hyperparameter-search-pipeline-design.md`):

- **Ray Tune** (`analysis/tune_search.py`, `main.py --search`) replaces the old
  YAML grid: list-valued YAML fields become `tune.choice` dimensions; trials run
  via `train_func` (reusing `TrainingInterface`) with an ASHA scheduler and
  fractional-GPU packing. `TrainingConfig.from_yaml` now returns one config.
- **vmap-ensemble** (`analysis/ensemble.py`) trains K same-architecture models on
  one shared on-GPU minibatch via `torch.func` for efficient within-architecture
  sweeps; `generate_synthetic_batch` (in `synthetic_lit_module.py`) is shared.

Deps live in the `tune` pixi feature (`ray[tune]`, `optuna`), included in all
test-running envs so the search path is CPU-tested in CI as well as on the GPU rig.

## Code Quality

- Ruff formatting and linting with Google-style docstrings
- Type hints required (Python 3.12+)
- Pre-commit hooks: ruff, ty, ascii-only, codespell, nbstripout, standard checks
- Spell checking with codespell

### ASCII-only source convention

**Never use Unicode characters in source files (this file included).** Math
notation in docstrings, comments, and string literals must be written in
**plain text or LaTeX**, not Unicode glyphs. Use `rho` (or `\rho`), `psi`,
`tau`, `sum_n`, `<psi|k>`, `A (x) B`, `->`, `<=`, `+/-`, `d^2`, `A^T`
instead of the corresponding Greek letters, angle brackets, arrows, and
relation glyphs. This keeps source grep-able, diff-able, and free of
homoglyph ambiguity.

Enforcement: `scripts/check_ascii.py` (run via `pixi run ascii`, part of
`pixi run all`, and the `ascii-only` pre-commit hook) fails on any non-ASCII
codepoint in `*.py`/`*.md`. Ruff's `RUF001/2/3` additionally flag the
confusable subset.
