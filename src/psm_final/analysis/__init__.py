"""Public exports for the psm_final.analysis subpackage."""

from importlib import import_module

__all__ = [
    "noise_ceiling",
    "correlation_rdm",
    "ModelAnalysisBase",
    "BetaVAEAnalysis",
    "VDVAEAnalysis",
    "AutoKL",
    "run_rsa",
    "discover_analyzers",
]

_EXPORTS = {
    "noise_ceiling": ".correlating",
    "correlation_rdm": ".correlating",
    "ModelAnalysisBase": ".model",
    "BetaVAEAnalysis": ".beta_vae_analysis",
    "VDVAEAnalysis": ".vdvae_analysis",
    "AutoKL": ".autoencoderKL",
    "run_rsa": ".runner",
    "discover_analyzers": ".runner",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

