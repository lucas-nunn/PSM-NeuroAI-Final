"""Public exports for psm_final."""

from importlib import import_module

__all__ = [
    "Algonauts",
    "TripleN",
    "SimpleMLP",
    "MLP_3layers",
    "fit_pca",
    "get_pca_activities",
    "plot_batch_predictions",
    "plot_dataset_samples",
    "plot_pca",
    "plot_rsa",
    "plot_tsne",
    "train",
    "test",
    "train_and_test",
    "session_macaque",
    "build_crosswalk",
    "load_crosswalk",
    "map_trials",
    "noise_ceiling",
    "correlation_rdm",
]

_EXPORTS = {
    "Algonauts": ".data.algonauts",
    "TripleN": ".data.triple_n",
    "SimpleMLP": ".helpers.MLP",
    "MLP_3layers": ".helpers.MLP",
    "fit_pca": ".helpers.pca",
    "get_pca_activities": ".helpers.pca",
    "plot_batch_predictions": ".helpers.plotting",
    "plot_dataset_samples": ".helpers.plotting",
    "plot_pca": ".helpers.plotting",
    "plot_rsa": ".helpers.plotting",
    "plot_tsne": ".helpers.plotting",
    "train": ".helpers.training_testing",
    "test": ".helpers.training_testing",
    "train_and_test": ".helpers.training_testing",
    "shared_stimuli": ".data.util",
    "session_macaque": ".data.stimulus",
    "build_crosswalk": ".data.stimulus",
    "load_crosswalk": ".data.stimulus",
    "map_trials": ".data.stimulus",
    "noise_ceiling": ".analysis.correlating",
    "correlation_rdm": ".analysis.correlating",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
