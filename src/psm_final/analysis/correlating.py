import numpy as np

from scipy.stats import spearmanr


def noise_normalize_correlations(correlations, ceilings, *, ceiling_squared=False):
    """Normalize correlations by their attainable correlation ceilings.

    Parameters
    ----------
    correlations
        Raw correlation coefficient(s).
    ceilings
        Matching noise-ceiling value(s). Set ``ceiling_squared=True`` when these
        are reliability / explainable-variance values (as in Algonauts and the
        Triple-N split-half reliability); they are square-rooted before normalizing
        a correlation.
    ceiling_squared
        Whether ``ceilings`` are expressed in squared-correlation units.

    Returns
    -------
    signed, squared
        ``r / r_ceiling`` and ``r**2 / r_ceiling**2``. Invalid, non-positive
        ceilings remain NaN. Values are deliberately not clipped: estimates above
        one are diagnostically useful because empirical ceilings are themselves
        uncertain.
    """
    correlations, ceilings = np.broadcast_arrays(
        np.asarray(correlations, dtype=float),
        np.asarray(ceilings, dtype=float),
    )
    correlation_ceiling = (
        np.sqrt(np.clip(ceilings, 0.0, None))
        if ceiling_squared
        else ceilings
    )
    valid = (
        np.isfinite(correlations)
        & np.isfinite(correlation_ceiling)
        & (correlation_ceiling > 0)
    )
    signed = np.full(correlations.shape, np.nan, dtype=float)
    squared = np.full(correlations.shape, np.nan, dtype=float)
    np.divide(
        correlations,
        correlation_ceiling,
        out=signed,
        where=valid,
    )
    np.divide(
        correlations ** 2,
        correlation_ceiling ** 2,
        out=squared,
        where=valid,
    )
    return signed, squared


def noise_ceiling(rdms: np.ndarray):
    rdms = np.asarray(rdms)
    n = rdms.shape[0]
    if n < 2:
        raise ValueError("need at least 2 RDMs to compute a noise ceiling")
    group_mean = rdms.mean(axis=0)
    uppers, lowers = [], []
    for i in range(n):
        loo_mean = (rdms.sum(axis=0) - rdms[i]) / (n - 1)   # mean of the other subjects
        uppers.append(spearmanr(rdms[i], group_mean)[0])
        lowers.append(spearmanr(rdms[i], loo_mean)[0])

    return float(np.mean(lowers)), float(np.mean(uppers))

def correlation_rdm(patterns, condensed=True):
    """Correlation-distance RDM (1 - Pearson r) for an (items x features) matrix.

    Vectorized as 1 - Gram of row-wise z-scored patterns (a single BLAS matmul),
    ~17x faster than scipy pdist('correlation') and numerically identical. Zero-
    variance rows normalize to zeros (distance treated as 1), avoiding NaNs.
    """
    X = np.asarray(patterns, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)
    rdm = 1.0 - X @ X.T
    return rdm[np.triu_indices(rdm.shape[0], k=1)] if condensed else rdm
