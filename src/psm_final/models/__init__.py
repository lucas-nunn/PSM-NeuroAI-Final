"""Public exports for the psm_final.models subpackage."""

from importlib import import_module

__all__ = [
    "BetaVAE",
    "COCODataset",
    "EncodingModel",
    "load_vdvae",
]

_EXPORTS = {
    "BetaVAE": ".beta_vae",
    "COCODataset": ".beta_vae",
    "EncodingModel": ".encoding",
    "load_vdvae": ".vdvae",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
