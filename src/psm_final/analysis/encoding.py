"""Cross-validated model-feature to neural-response encoding utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ALPHAS = tuple(np.logspace(-4, 4, 9))
RESULT_COLUMNS = [
    "mean_encoding_score",
    "std_encoding_score",
    "best_alpha",
    "alpha_selection_stability",
    "outer_selected_alphas",
    "n_targets",
    "n_outer_folds",
]


@dataclass(frozen=True)
class EncodingCVResult:
    outer_scores: np.ndarray
    outer_selected_alphas: tuple[float, ...]
    best_alpha: float
    alpha_selection_stability: float
    n_targets: int


def prediction_correlations(predicted, observed):
    """Pearson r per response column across held-out stimuli.

    Constant predictions against varying observations score zero because they
    contain no predictive signal. Constant observed responses remain unscorable
    (NaN).
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError(
            "predicted and observed responses must be 2D arrays with equal shapes"
        )
    if predicted.shape[0] < 2:
        raise ValueError("need at least 2 held-out stimuli to compute Pearson r")

    correlations = np.full(predicted.shape[1], np.nan)
    finite = np.isfinite(predicted).all(axis=0) & np.isfinite(observed).all(axis=0)
    finite_columns = np.flatnonzero(finite)
    if not finite_columns.size:
        return correlations

    predicted_finite = predicted[:, finite_columns]
    observed_finite = observed[:, finite_columns]
    predicted_centered = predicted_finite - predicted_finite.mean(
        axis=0, keepdims=True
    )
    observed_centered = observed_finite - observed_finite.mean(
        axis=0, keepdims=True
    )
    predicted_norm = np.linalg.norm(predicted_centered, axis=0)
    observed_norm = np.linalg.norm(observed_centered, axis=0)

    scorable = observed_norm > 0
    variable_prediction = predicted_norm > 0
    correlations[finite_columns[scorable & ~variable_prediction]] = 0.0

    valid = scorable & variable_prediction
    valid_columns = finite_columns[valid]
    correlations[valid_columns] = np.sum(
        predicted_centered[:, valid] * observed_centered[:, valid], axis=0
    ) / (predicted_norm[valid] * observed_norm[valid])
    correlations[valid_columns] = np.clip(
        correlations[valid_columns], -1.0, 1.0
    )
    return correlations


def _mean_prediction_score(predicted, observed):
    correlations = prediction_correlations(predicted, observed)
    finite = correlations[np.isfinite(correlations)]
    return float(finite.mean()) if finite.size else np.nan


def _validate_cv_inputs(features, responses, alphas, outer_folds, inner_folds, regression):
    features = np.asarray(features, dtype=float)
    responses = np.asarray(responses, dtype=float)
    if features.ndim != 2 or responses.ndim != 2:
        raise ValueError("features and responses must both be 2D arrays")
    if features.shape[0] != responses.shape[0]:
        raise ValueError("features and responses must contain the same stimuli")
    if responses.shape[1] == 0:
        raise ValueError("responses must contain at least one voxel or unit")

    regression = str(regression).lower()
    if regression not in {"ridge", "lasso"}:
        raise ValueError("regression must be 'ridge' or 'lasso'")

    alphas = np.asarray(DEFAULT_ALPHAS if alphas is None else alphas, dtype=float)
    if alphas.ndim != 1 or not alphas.size:
        raise ValueError("alphas must be a non-empty one-dimensional sequence")
    if not np.isfinite(alphas).all() or np.any(alphas <= 0):
        raise ValueError("alphas must contain only finite positive values")
    alphas = np.unique(alphas)

    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")
    n_stimuli = features.shape[0]
    if n_stimuli < 2 * outer_folds:
        raise ValueError(
            f"need at least {2 * outer_folds} stimuli so every outer test fold "
            "can compute Pearson r"
        )
    smallest_outer_train = n_stimuli - int(np.ceil(n_stimuli / outer_folds))
    if smallest_outer_train < 2 * inner_folds:
        raise ValueError(
            "outer training folds need at least "
            f"{2 * inner_folds} stimuli so every inner validation fold can "
            "compute Pearson r"
        )
    return features, responses, alphas, regression


def _regressor(kind, alpha):
    if kind == "ridge":
        estimator = Ridge(alpha=alpha)
    else:
        estimator = Lasso(alpha=alpha, max_iter=10_000)
    return make_pipeline(StandardScaler(), estimator)


def _fit_score(features, responses, train, test, regression, alpha):
    model = _regressor(regression, alpha)
    response_scaler = StandardScaler()
    train_responses = response_scaler.fit_transform(responses[train])
    test_responses = response_scaler.transform(responses[test])
    model.fit(features[train], train_responses)
    predicted = model.predict(features[test])
    return _mean_prediction_score(predicted, test_responses)


def _alpha_scores(features, responses, splits, regression, alphas):
    scores = np.full(len(alphas), np.nan)
    for alpha_index, alpha in enumerate(alphas):
        fold_scores = [
            _fit_score(features, responses, train, validation, regression, alpha)
            for train, validation in splits
        ]
        finite = np.asarray(fold_scores)[np.isfinite(fold_scores)]
        if finite.size:
            scores[alpha_index] = finite.mean()
    return scores


def _select_alpha(features, responses, splits, regression, alphas):
    scores = _alpha_scores(features, responses, splits, regression, alphas)
    if not np.isfinite(scores).any():
        return np.nan, scores
    return float(alphas[np.nanargmax(scores)]), scores


def nested_encoding_cv(
    features,
    responses,
    *,
    regression="ridge",
    alphas=None,
    outer_folds=5,
    inner_folds=3,
    seed=42,
):
    """Nested stimulus-level CV with one alpha shared by all response columns.

    Inner folds select alpha from the mean voxel/unit correlation. Outer folds
    estimate prediction performance. A final full-data inner sweep reports the
    alpha to use when fitting a deployable model after evaluation.
    """
    features, responses, alphas, regression = _validate_cv_inputs(
        features, responses, alphas, outer_folds, inner_folds, regression
    )

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    outer_scores = []
    selected_alphas = []
    for fold_index, (outer_train, outer_test) in enumerate(outer_cv.split(features)):
        inner_cv = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + fold_index + 1,
        )
        inner_splits = list(inner_cv.split(features[outer_train]))
        best_alpha, _ = _select_alpha(
            features[outer_train],
            responses[outer_train],
            inner_splits,
            regression,
            alphas,
        )
        selected_alphas.append(best_alpha)
        if np.isfinite(best_alpha):
            outer_scores.append(
                _fit_score(
                    features,
                    responses,
                    outer_train,
                    outer_test,
                    regression,
                    best_alpha,
                )
            )
        else:
            outer_scores.append(np.nan)

    final_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + 10_000)
    best_alpha, _ = _select_alpha(
        features,
        responses,
        list(final_cv.split(features)),
        regression,
        alphas,
    )
    finite_selected = np.asarray(selected_alphas)[np.isfinite(selected_alphas)]
    stability = (
        float(np.mean(np.isclose(finite_selected, best_alpha)))
        if finite_selected.size and np.isfinite(best_alpha)
        else np.nan
    )
    return EncodingCVResult(
        outer_scores=np.asarray(outer_scores, dtype=float),
        outer_selected_alphas=tuple(float(alpha) for alpha in selected_alphas),
        best_alpha=float(best_alpha),
        alpha_selection_stability=stability,
        n_targets=responses.shape[1],
    )


def _result_row(label, result):
    finite_scores = result.outer_scores[np.isfinite(result.outer_scores)]
    mean_score = float(finite_scores.mean()) if finite_scores.size else np.nan
    std_score = float(finite_scores.std()) if finite_scores.size else np.nan
    selected = ",".join(f"{alpha:g}" for alpha in result.outer_selected_alphas)
    return (
        label,
        mean_score,
        std_score,
        result.best_alpha,
        result.alpha_selection_stability,
        selected,
        result.n_targets,
        len(result.outer_scores),
    )


def _result_frame(rows, index_name):
    return pd.DataFrame(rows, columns=[index_name, *RESULT_COLUMNS]).set_index(index_name)


def _prepare(analyzer, algonauts, triple_n, shared_ids, subjects):
    subjects = list(subjects)
    nsd_ids, stim_index = analyzer.aligned_stimuli(
        algonauts, triple_n, shared_ids, subjects
    )
    features = analyzer.features(indices=np.asarray(stim_index) - 1)
    return subjects, nsd_ids, stim_index, features


def _encoding_algonauts_from_prepared(
    algonauts,
    subjects,
    nsd_ids,
    features,
    *,
    rois,
    regression,
    alphas,
    outer_folds,
    inner_folds,
    seed,
    progress=None,
):
    rois = list(algonauts.ALGO_ROIS if rois is None else rois)
    responses = {
        subject: algonauts.response_matrix(subject, indices=nsd_ids)
        for subject in subjects
    }
    rows = []
    for group_index, roi in enumerate(rois, start=1):
        if progress is not None:
            progress("algonauts", roi, group_index, len(rois))
        roi_responses = []
        for subject in subjects:
            left, right = responses[subject]
            left_mask, right_mask = algonauts.roi_mask(roi, subject)
            matrix = np.concatenate(
                (left[:, left_mask], right[:, right_mask]), axis=1
            )
            if matrix.shape[1]:
                roi_responses.append(matrix)
        if not roi_responses:
            continue
        result = nested_encoding_cv(
            features,
            np.concatenate(roi_responses, axis=1),
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
        )
        rows.append(_result_row(roi, result))
    return _result_frame(rows, "roi")


def _encoding_triple_n_from_prepared(
    triple_n,
    stim_index,
    features,
    *,
    area_labels,
    regression,
    alphas,
    outer_folds,
    inner_folds,
    seed,
    progress=None,
):
    if area_labels is None:
        area_labels = sorted(triple_n.units["area_label"].unique())
    area_labels = list(area_labels)
    rows = []
    for group_index, label in enumerate(area_labels, start=1):
        if progress is not None:
            progress("triple_n", label, group_index, len(area_labels))
        try:
            area_responses = triple_n.response_matrix(
                area_label=label,
                indices=stim_index,
            )
        except ValueError:
            continue
        result = nested_encoding_cv(
            features,
            area_responses,
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
        )
        rows.append(_result_row(label, result))
    return _result_frame(rows, "area_label")


def encoding_algonauts(
    analyzer,
    algonauts,
    triple_n,
    shared_ids,
    *,
    subjects=range(1, 9),
    rois=None,
    regression="ridge",
    alphas=None,
    outer_folds=5,
    inner_folds=3,
    seed=42,
    progress=None,
):
    subjects, nsd_ids, _stim_index, features = _prepare(
        analyzer, algonauts, triple_n, shared_ids, subjects
    )
    return _encoding_algonauts_from_prepared(
        algonauts,
        subjects,
        nsd_ids,
        features,
        rois=rois,
        regression=regression,
        alphas=alphas,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        progress=progress,
    )


def encoding_triple_n(
    analyzer,
    triple_n,
    algonauts,
    shared_ids,
    *,
    subjects=range(1, 9),
    area_labels=None,
    regression="ridge",
    alphas=None,
    outer_folds=5,
    inner_folds=3,
    seed=42,
    progress=None,
):
    _subjects, _nsd_ids, stim_index, features = _prepare(
        analyzer, algonauts, triple_n, shared_ids, subjects
    )
    return _encoding_triple_n_from_prepared(
        triple_n,
        stim_index,
        features,
        area_labels=area_labels,
        regression=regression,
        alphas=alphas,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        progress=progress,
    )


def encoding_tables(
    analyzer,
    algonauts,
    triple_n,
    shared_ids,
    *,
    subjects=range(1, 9),
    rois=None,
    area_labels=None,
    regression="ridge",
    alphas=None,
    outer_folds=5,
    inner_folds=3,
    seed=42,
    progress=None,
):
    """Compute both encoding tables while extracting model features only once."""
    subjects, nsd_ids, stim_index, features = _prepare(
        analyzer, algonauts, triple_n, shared_ids, subjects
    )
    common = dict(
        regression=regression,
        alphas=alphas,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )
    return {
        "algonauts": _encoding_algonauts_from_prepared(
            algonauts,
            subjects,
            nsd_ids,
            features,
            rois=rois,
            progress=progress,
            **common,
        ),
        "triple_n": _encoding_triple_n_from_prepared(
            triple_n,
            stim_index,
            features,
            area_labels=area_labels,
            progress=progress,
            **common,
        ),
    }
