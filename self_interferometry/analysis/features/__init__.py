"""Feature engineering registry for self-interferometry.

The :class:`FeatureRegistry` is the single source of truth for how raw Red
Pitaya channel data becomes model inputs and supervision targets. Features are
registered by decorating a compute function with :func:`register_feature`;
importing :mod:`self_interferometry.analysis.features.features` populates the
module-level :data:`default_registry`.

Normalization statistics for every registered feature are computed offline by
``compute_norm_stats.py`` and version-controlled in ``normalization.py``; the
model's normalization wrapper (``analysis/models/base.py``) bakes those stats
into ``register_buffer`` s so the rest of the codebase works in raw units.
"""

from .registry import (
    FeatureRegistry,
    FeatureSpec,
    default_registry,
    register_feature,
)

# Importing features.py registers all built-in features as a side effect.
from . import features  # noqa: E402,F401  (side-effect import after registry)

__all__ = [
    'FeatureRegistry',
    'FeatureSpec',
    'default_registry',
    'register_feature',
]
