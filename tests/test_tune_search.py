#!/usr/bin/env python
"""Tests for the Ray Tune search driver (smi.analysis.tune_search).

These run on CPU and must stay fast (seconds): tiny sequence length, batch size,
and 1-2 steps per epoch. The end-to-end smoke test exercises the *same*
``train_func`` used on the GPU rig, just with ``accelerator: cpu`` and
``gpu_fraction=0``.
"""

from pathlib import Path

import pytest
import yaml

ray = pytest.importorskip('ray')

from ray import tune  # noqa: E402  (after importorskip)

from smi.analysis.tune_search import (  # noqa: E402
    SEARCH_METRIC,
    build_param_space,
    run_search,
)

CONFIG_PATH = (
    Path(__file__).parent.parent
    / 'smi'
    / 'analysis'
    / 'models'
    / 'configs'
    / 'tune-example.yaml'
)


def _tiny_config_dict() -> dict:
    """Load the example tune config as a dict."""
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_build_param_space_maps_lists_and_scalars():
    """List fields -> tune.choice; scalars pass through unchanged."""
    config_dict = _tiny_config_dict()
    param_space = build_param_space(config_dict)

    # Sections present in the YAML are present in the space; absent ones omitted.
    assert set(param_space) == {'model', 'training', 'loss', 'data', 'synthetic'}

    # A scalar passes through verbatim.
    assert param_space['model']['kernel_size'] == 3
    assert param_space['model']['sequence_length'] == 256
    assert param_space['training']['accelerator'] == 'cpu'

    # A list-valued field becomes a tune.choice search dimension.
    dropout = param_space['model']['dropout']
    assert isinstance(dropout, tune.search.sample.Categorical)
    assert dropout.categories == [0.0, 0.05]

    # learning_rate is written as YAML scientific notation, which PyYAML parses
    # as strings; LitModule coerces with float(). The choice preserves the
    # original list verbatim.
    lr = param_space['training']['learning_rate']
    assert isinstance(lr, tune.search.sample.Categorical)
    assert lr.categories == ['1e-3', '5e-4']

    # A list-of-lists field round-trips as a single choice over inner lists.
    channels = param_space['model']['temporal_channels']
    assert isinstance(channels, tune.search.sample.Categorical)
    assert channels.categories == [[8, 8, 16]]


def test_build_param_space_omits_missing_sections():
    """Sections absent from the config (e.g. synthetic) are not in the space."""
    config_dict = _tiny_config_dict()
    config_dict.pop('synthetic')
    param_space = build_param_space(config_dict)
    assert 'synthetic' not in param_space
    assert 'model' in param_space


def test_run_search_smoke_returns_best_result():
    """End-to-end: 1 sample, CPU, 1 epoch, tiny synthetic config.

    Exercises the same train_func path used on GPU and asserts a best result
    with the search metric is returned.
    """
    if not ray.is_initialized():
        ray.init(
            local_mode=True, num_cpus=1, include_dashboard=False, log_to_driver=False
        )
    try:
        best = run_search(
            str(CONFIG_PATH),
            num_samples=1,
            gpu_fraction=0.0,
            cpus_per_trial=1,
            max_concurrent_trials=1,
        )
    finally:
        ray.shutdown()

    assert best is not None
    assert SEARCH_METRIC in best.metrics
    assert best.metrics[SEARCH_METRIC] is not None
    # The sampled value came from one of the search dimensions.
    assert best.config['model']['dropout'] in (0.0, 0.05)
    assert best.config['training']['learning_rate'] in ('1e-3', '5e-4')
