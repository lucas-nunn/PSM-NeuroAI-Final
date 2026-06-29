"""Public exports for the psm_final.data subpackage."""

from importlib import import_module

__all__ = [
    "Algonauts",
    "TripleN",
    "shared_stimuli",
    "session_macaque",
    "build_crosswalk",
    "load_crosswalk",
    "map_trials",
]

_EXPORTS = {
    "Algonauts": ".algonauts",
    "TripleN": ".triple_n",
    "shared_stimuli": ".util",
    "session_macaque": ".stimulus",
    "build_crosswalk": ".stimulus",
    "load_crosswalk": ".stimulus",
    "map_trials": ".stimulus",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
