"""Cross-validated model-feature to neural-response encoding utilities."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from psm_final.analysis.correlating import noise_normalize_correlations


DEFAULT_ALPHAS = tuple(np.logspace(-4, 4, 9))
DEFAULT_MIN_TRIPLE_N_RELIABILITY = 0.4
RESULT_COLUMNS = [
    "mean_encoding_score",
    "std_encoding_score",
    "noise_ceiling_r",
    "noise_ceiling_threshold",
    "mean_noise_normalized_r",
    "std_noise_normalized_r",
    "mean_noise_normalized_r2",
    "std_noise_normalized_r2",
    "n_ceiling_targets",
    "best_alpha",
    "alpha_selection_stability",
    "outer_selected_alphas",
    "n_targets",
    "n_outer_folds",
]


@dataclass(frozen=True)
class EncodingCVResult:
    outer_scores: np.ndarray
    outer_target_scores: np.ndarray
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


def _fit_correlations(features, responses, train, test, regression, alpha):
    model = _regressor(regression, alpha)
    response_scaler = StandardScaler()
    train_responses = response_scaler.fit_transform(responses[train])
    test_responses = response_scaler.transform(responses[test])
    model.fit(features[train], train_responses)
    predicted = model.predict(features[test])
    return prediction_correlations(predicted, test_responses)


def _fit_score(features, responses, train, test, regression, alpha):
    correlations = _fit_correlations(
        features, responses, train, test, regression, alpha
    )
    finite = correlations[np.isfinite(correlations)]
    return float(finite.mean()) if finite.size else np.nan


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
    outer_target_scores = []
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
            target_scores = _fit_correlations(
                features,
                responses,
                outer_train,
                outer_test,
                regression,
                best_alpha,
            )
            finite = target_scores[np.isfinite(target_scores)]
            outer_scores.append(float(finite.mean()) if finite.size else np.nan)
            outer_target_scores.append(target_scores)
        else:
            outer_scores.append(np.nan)
            outer_target_scores.append(
                np.full(responses.shape[1], np.nan, dtype=float)
            )

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
        outer_target_scores=np.asarray(outer_target_scores, dtype=float),
        outer_selected_alphas=tuple(float(alpha) for alpha in selected_alphas),
        best_alpha=float(best_alpha),
        alpha_selection_stability=stability,
        n_targets=responses.shape[1],
    )


def _finite_fold_summary(values):
    """Mean/std of per-fold means while preserving missing-fold semantics."""
    values = np.asarray(values, dtype=float)
    fold_means = []
    for fold in values:
        finite = fold[np.isfinite(fold)]
        fold_means.append(float(finite.mean()) if finite.size else np.nan)
    fold_means = np.asarray(fold_means, dtype=float)
    finite = fold_means[np.isfinite(fold_means)]
    if not finite.size:
        return np.nan, np.nan
    return float(finite.mean()), float(finite.std())


def _noise_normalized_result(result, noise_ceiling, *, minimum_ceiling=0.0):
    """Group summaries from per-target outer-fold scores and R² ceilings."""
    minimum_ceiling = float(minimum_ceiling)
    if not np.isfinite(minimum_ceiling) or minimum_ceiling < 0:
        raise ValueError("minimum noise ceiling must be finite and non-negative")
    empty = (
        np.nan,
        minimum_ceiling,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0,
    )
    if noise_ceiling is None:
        return empty
    ceiling = np.asarray(noise_ceiling, dtype=float).reshape(-1)
    if ceiling.size != result.n_targets:
        raise ValueError(
            "noise ceiling must have one value per neural target; "
            f"got {ceiling.size} for {result.n_targets} targets"
        )
    valid_ceiling = np.isfinite(ceiling) & (ceiling > 0)
    if minimum_ceiling:
        valid_ceiling &= ceiling >= minimum_ceiling
    if not valid_ceiling.any():
        return empty

    ceiling_r = np.sqrt(np.clip(ceiling[valid_ceiling], 0.0, None))
    signed, squared = noise_normalize_correlations(
        result.outer_target_scores[:, valid_ceiling],
        ceiling[None, valid_ceiling],
        ceiling_squared=True,
    )
    signed_mean, signed_std = _finite_fold_summary(signed)
    squared_mean, squared_std = _finite_fold_summary(squared)
    return (
        float(ceiling_r.mean()),
        minimum_ceiling,
        signed_mean,
        signed_std,
        squared_mean,
        squared_std,
        int(valid_ceiling.sum()),
    )


def _result_row(label, result, noise_ceiling=None, *, minimum_ceiling=0.0):
    finite_scores = result.outer_scores[np.isfinite(result.outer_scores)]
    mean_score = float(finite_scores.mean()) if finite_scores.size else np.nan
    std_score = float(finite_scores.std()) if finite_scores.size else np.nan
    normalized = _noise_normalized_result(
        result, noise_ceiling, minimum_ceiling=minimum_ceiling
    )
    selected = ",".join(f"{alpha:g}" for alpha in result.outer_selected_alphas)
    return (
        label,
        mean_score,
        std_score,
        *normalized,
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
    ceiling_loader = getattr(algonauts, "noise_ceiling", None)
    warned_missing_ceiling = False
    rows = []
    for group_index, roi in enumerate(rois, start=1):
        if progress is not None:
            progress("algonauts", roi, group_index, len(rois))
        roi_responses = []
        roi_ceilings = []
        for subject in subjects:
            left, right = responses[subject]
            left_mask, right_mask = algonauts.roi_mask(roi, subject)
            left_roi = left[:, left_mask]
            right_roi = right[:, right_mask]
            matrix = np.concatenate((left_roi, right_roi), axis=1)
            if matrix.shape[1]:
                roi_responses.append(matrix)
                try:
                    if not callable(ceiling_loader):
                        raise FileNotFoundError("dataset exposes no ceiling loader")
                    left_ceiling, right_ceiling = ceiling_loader(subject, roi=roi)
                    subject_ceiling = np.concatenate(
                        (
                            np.asarray(left_ceiling, dtype=float).reshape(-1),
                            np.asarray(right_ceiling, dtype=float).reshape(-1),
                        )
                    )
                    if subject_ceiling.size != matrix.shape[1]:
                        raise ValueError(
                            "Algonauts ceiling/response target counts differ for "
                            f"subject {subject}, ROI {roi}: "
                            f"{subject_ceiling.size} != {matrix.shape[1]}"
                        )
                except (FileNotFoundError, OSError) as exc:
                    subject_ceiling = np.full(matrix.shape[1], np.nan)
                    if not warned_missing_ceiling:
                        warnings.warn(
                            "Algonauts noise ceilings unavailable or invalid; "
                            f"normalized encoding metrics will be NaN ({exc})"
                        )
                        warned_missing_ceiling = True
                roi_ceilings.append(subject_ceiling)
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
        rows.append(
            _result_row(roi, result, np.concatenate(roi_ceilings))
        )
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
    min_reliability,
    progress=None,
):
    min_reliability = float(min_reliability)
    if not np.isfinite(min_reliability) or not 0 <= min_reliability <= 1:
        raise ValueError("Triple-N minimum reliability must be between 0 and 1")
    if area_labels is None:
        area_labels = sorted(triple_n.units["area_label"].unique())
    area_labels = list(area_labels)

    rows = []
    for group_index, label in enumerate(area_labels, start=1):
        if progress is not None:
            progress("triple_n", label, group_index, len(area_labels))
        try:
            group_responses = triple_n.response_matrix(
                indices=stim_index, area_label=label
            )
        except ValueError:
            continue
        noise_ceiling = None
        metadata_loader = getattr(triple_n, "unit_metadata", None)
        if callable(metadata_loader):
            metadata = metadata_loader(area_label=label)
            if "reliability_best" in metadata:
                noise_ceiling = metadata["reliability_best"].to_numpy(dtype=float)
                if noise_ceiling.size != group_responses.shape[1]:
                    raise ValueError(
                        "Triple-N response/metadata target counts differ for "
                        f"{label}: {group_responses.shape[1]} != {noise_ceiling.size}"
                    )
                reliable = (
                    np.isfinite(noise_ceiling)
                    & (noise_ceiling >= float(min_reliability))
                )
                if reliable.sum() < 2:
                    continue
                group_responses = group_responses[:, reliable]
                noise_ceiling = noise_ceiling[reliable]
        result = nested_encoding_cv(
            features,
            group_responses,
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
        )
        rows.append(
            _result_row(
                label,
                result,
                noise_ceiling,
                minimum_ceiling=min_reliability,
            )
        )
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
    min_reliability=DEFAULT_MIN_TRIPLE_N_RELIABILITY,
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
        min_reliability=min_reliability,
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
    triple_n_min_reliability=DEFAULT_MIN_TRIPLE_N_RELIABILITY,
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
            min_reliability=triple_n_min_reliability,
            progress=progress,
            **common,
        ),
    }
