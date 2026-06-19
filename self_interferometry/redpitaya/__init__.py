"""Red Pitaya hardware interface module."""

from .redpitaya_config import RedPitayaConfig
from .scpi import SCPI

__all__ = ['SCPI', 'RedPitayaConfig', 'RedPitayaManager']


def __getattr__(name: str) -> object:
    """Expose RedPitayaManager lazily to avoid a circular import.

    The manager module imports from self_interferometry.synthetic, which in turn
    imports RedPitayaConfig from this package. Importing manager eagerly here
    would close that cycle during package initialization. Deferring the manager
    re-export to first attribute access keeps the public API (accessing
    RedPitayaManager off this package) intact while breaking the cycle.
    """
    if name == 'RedPitayaManager':
        from .manager import RedPitayaManager

        return RedPitayaManager
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
