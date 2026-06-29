"""Helper exports for psm_final."""

from importlib import import_module

__all__ = [
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
]

_EXPORTS = {
    "SimpleMLP": ".MLP",
    "MLP_3layers": ".MLP",
    "fit_pca": ".pca",
    "get_pca_activities": ".pca",
    "plot_batch_predictions": ".plotting",
    "plot_dataset_samples": ".plotting",
    "plot_pca": ".plotting",
    "plot_rsa": ".plotting",
    "plot_tsne": ".plotting",
    "train": ".training_testing",
    "test": ".training_testing",
    "train_and_test": ".training_testing",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
