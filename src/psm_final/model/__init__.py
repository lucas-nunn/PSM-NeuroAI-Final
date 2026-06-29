"""Public exports for the psm_final.model subpackage."""

from importlib import import_module

__all__ = [
    "Model",
    "Algonauts",
    "TripleN",
]

_EXPORTS = {
    "Model": ".base",
    "Algonauts": ".algonauts",
    "TripleN": ".triple_n",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
