#!/usr/bin/env python
"""Ray Tune driver for the SMI hyperparameter / architecture search pipeline.

This module replaces the YAML Cartesian-product grid search that used to live in
:class:`smi.analysis.training_interface.TrainingConfig`. A search config is an
ordinary training YAML in which any field may be a list; list-valued fields in
the ``model`` / ``training`` / ``loss`` / ``data`` / ``synthetic`` sections are
interpreted as a ``ray.tune.choice([...])`` search dimension, and scalar fields
are fixed. Ray Tune then *samples* configurations (with an ASHA scheduler for
early stopping and fractional-GPU packing) instead of exhaustively enumerating
the grid.

The same :func:`train_func` is used on the dev Mac (CPU, tiny sizes, for tests)
and on the CUDA rig (GPU); the only difference is the ``accelerator`` / ``devices``
fields in the config and the ``gpu_fraction`` resource request.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from ray import tune
from ray.tune import RunConfig, TuneConfig, Tuner
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from smi.analysis.training_interface import TrainingConfig, TrainingInterface

logger = logging.getLogger(__name__)

#: The metric ASHA optimizes and the best config is selected by.
SEARCH_METRIC = 'val/total_unweighted_loss'
SEARCH_MODE = 'min'

#: Sections of the YAML config that contribute to the search space.
_CONFIG_SECTIONS = ('model', 'training', 'loss', 'data', 'synthetic')


def build_param_space(config_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a loaded YAML config dict to a Ray Tune ``param_space``.

    For each of the ``model`` / ``training`` / ``loss`` / ``data`` / ``synthetic``
    sections, list-valued fields become ``ray.tune.choice([...])`` (one sampled
    element per trial) and scalar fields are passed through unchanged. This
    preserves the historical YAML grid semantics -- a field written as a list is
    a search dimension -- while letting Ray sample / early-stop rather than
    enumerate the full Cartesian product.

    A field whose intended value is itself a list (e.g. ``temporal_channels`` or
    ``wavelengths_nm``) is written in the YAML as a singleton list-of-lists,
    exactly as in the existing grid configs, so it round-trips as a single
    ``choice`` over inner lists.

    Args:
        config_dict: Parsed YAML config with top-level section keys.

    Returns:
        A ``param_space`` dict keyed by section name; each value is a dict of
        ``{field: value_or_tune.choice}``. Sections absent from the config
        (e.g. ``synthetic`` for real-data runs) are omitted.
    """
    param_space: dict[str, dict[str, Any]] = {}
    for section in _CONFIG_SECTIONS:
        raw = config_dict.get(section)
        if raw is None:
            continue
        section_space: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                section_space[key] = tune.choice(value)
            else:
                section_space[key] = value
        param_space[section] = section_space
    return param_space


def _config_from_sampled(sampled: dict[str, dict[str, Any]]) -> TrainingConfig:
    """Rebuild a :class:`TrainingConfig` from one sampled Tune config.

    Ray resolves each ``tune.choice`` to a concrete value before calling
    :func:`train_func`, so ``sampled`` is a plain nested dict mirroring the YAML
    sections. This reassembles the per-section hparams dicts that
    ``TrainingInterface`` / ``LitModule`` consume.

    Args:
        sampled: Nested dict with ``model`` / ``training`` / ``loss`` / ``data``
            and optionally ``synthetic`` sections of concrete values.

    Returns:
        A single-run :class:`TrainingConfig`.
    """
    return TrainingConfig(
        model_config=dict(sampled['model']),
        training_config=dict(sampled['training']),
        loss_config=dict(sampled['loss']),
        data_config=dict(sampled['data']),
        synthetic_config=(
            dict(sampled['synthetic']) if sampled.get('synthetic') is not None else None
        ),
    )


def train_func(config: dict[str, dict[str, Any]]) -> None:
    """Ray Tune trainable: train one sampled configuration.

    Reuses :class:`TrainingInterface` for data-path construction (real data via
    ``VelocityDataModule``; synthetic via ``SyntheticLitModule`` +
    ``SyntheticIndexDataset``), LightningModule creation (``LitModule`` real /
    ``SyntheticLitModule`` synthetic), and trainer setup. The only Tune-specific
    additions are:

    - a :class:`TuneReportCheckpointCallback` reporting :data:`SEARCH_METRIC`,
      injected via ``TrainingInterface(extra_callbacks=...)``;
    - ``check_val_every_n_epoch=1`` so the metric is reported every epoch (the
      single-run default of 5 would never report on a short ASHA budget).

    The same code path runs on CPU (tests) and GPU (rig) -- the accelerator and
    device count come straight from ``config['training']``.

    Args:
        config: One sampled point from the Tune ``param_space`` (nested by
            section). Ray has already resolved all ``tune.choice`` dimensions to
            concrete values.
    """
    training_config = _config_from_sampled(config)

    report_callback = TuneReportCheckpointCallback(
        metrics={SEARCH_METRIC: SEARCH_METRIC},
        on='validation_end',
        save_checkpoints=False,
    )

    experiment_name = training_config.training_config.get(
        'experiment_name', training_config.model_config['type']
    )

    interface = TrainingInterface(
        config=training_config,
        experiment_name=experiment_name,
        extra_callbacks=[report_callback],
        check_val_every_n_epoch=1,
    )
    interface.train()


def run_search(
    config_path: str,
    *,
    num_samples: int = 10,
    gpu_fraction: float = 0.0,
    cpus_per_trial: int = 1,
    max_concurrent_trials: int | None = None,
    max_t: int | None = None,
    grace_period: int = 1,
    storage_path: str | None = None,
) -> tune.ResultGrid | Any:
    """Run a Ray Tune search defined by a YAML config and return the best result.

    Builds the search space from the YAML (see :func:`build_param_space`), runs
    ``num_samples`` trials with an :class:`ASHAScheduler` for early stopping, and
    packs trials onto the GPU via fractional-GPU resources. Exposes the
    single-GPU packing knobs from the design (``gpu_fraction``,
    ``max_concurrent_trials``, ``num_samples``).

    Args:
        config_path: Path to the search YAML (list-valued fields = search dims).
        num_samples: Number of configurations Ray samples and trains.
        gpu_fraction: Fraction of a GPU each trial requests. ``0`` for CPU-only
            (dev Mac / tests); on the 24 GB rig, ``~1/gpu_fraction`` trials pack
            per GPU (memory is not isolated, so cap with ``max_concurrent_trials``).
        cpus_per_trial: CPUs reserved per trial.
        max_concurrent_trials: Hard cap on simultaneous trials (CUDA-context
            overhead, not weights, is the real limit for tiny models).
        max_t: ASHA ``max_t`` (max training iterations / epochs before a trial
            may run to completion). Defaults to the config's ``max_epochs``.
        grace_period: ASHA ``grace_period`` (min iterations before a trial may be
            early-stopped).
        storage_path: Where Ray persists results. Defaults to Ray's default.

    Returns:
        The best :class:`ray.tune.Result` by :data:`SEARCH_METRIC` (``min``).
    """
    with Path(config_path).open() as f:
        config_dict = yaml.safe_load(f)

    param_space = build_param_space(config_dict)

    if max_t is None:
        # max_epochs may be a search dimension; take the largest candidate.
        max_epochs = config_dict['training']['max_epochs']
        max_t = max(max_epochs) if isinstance(max_epochs, list) else int(max_epochs)

    scheduler = ASHAScheduler(
        metric=SEARCH_METRIC, mode=SEARCH_MODE, max_t=max_t, grace_period=grace_period
    )

    trainable = tune.with_resources(
        train_func, {'CPU': cpus_per_trial, 'GPU': gpu_fraction}
    )

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=RunConfig(storage_path=storage_path),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=SEARCH_METRIC, mode=SEARCH_MODE)
    logger.info(
        'Best config (%s=%s): %s',
        SEARCH_METRIC,
        best_result.metrics.get(SEARCH_METRIC),
        best_result.config,
    )
    return best_result
