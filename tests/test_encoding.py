import unittest

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from psm_final.analysis.encoding import (
    DEFAULT_MIN_TRIPLE_N_RELIABILITY,
    EncodingCVResult,
    RESULT_COLUMNS,
    _noise_normalized_result,
    nested_encoding_cv,
)
from psm_final.analysis.model import ModelAnalysisBase
from psm_final.models.encoding import EncodingModel


class PredictionCorrelationTests(unittest.TestCase):
    def test_matches_scipy_for_variable_columns(self):
        rng = np.random.default_rng(42)
        predicted = rng.normal(size=(20, 4))
        observed = rng.normal(size=(20, 4))

        scores = EncodingModel._prediction_correlations(predicted, observed)
        expected = np.array([
            pearsonr(predicted[:, column], observed[:, column]).statistic
            for column in range(predicted.shape[1])
        ])

        np.testing.assert_allclose(scores, expected)

    def test_scores_variable_constant_and_unscorable_columns(self):
        predicted = np.array([
            [1.0, 5.0, 1.0],
            [2.0, 5.0, 2.0],
            [3.0, 5.0, 3.0],
        ])
        observed = np.array([
            [1.0, 3.0, 7.0],
            [2.0, 2.0, 7.0],
            [3.0, 1.0, 7.0],
        ])

        scores = EncodingModel._prediction_correlations(predicted, observed)

        np.testing.assert_allclose(scores[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(scores[2]))

    def test_requires_matching_2d_arrays(self):
        with self.assertRaisesRegex(ValueError, "2D arrays with equal shapes"):
            EncodingModel._prediction_correlations(
                np.ones((3, 2)), np.ones((3, 1))
            )

    def test_requires_two_held_out_stimuli(self):
        with self.assertRaisesRegex(ValueError, "at least 2 held-out stimuli"):
            EncodingModel._prediction_correlations(
                np.ones((1, 2)), np.ones((1, 2))
            )


class NestedEncodingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.features = rng.normal(size=(60, 4))
        weights = rng.normal(size=(4, 3))
        self.responses = self.features @ weights + rng.normal(
            scale=0.05, size=(60, 3)
        )

    def test_ridge_and_lasso_select_alpha_and_generalize(self):
        for regression in ("ridge", "lasso"):
            with self.subTest(regression=regression):
                result = nested_encoding_cv(
                    self.features,
                    self.responses,
                    regression=regression,
                    alphas=[1e-4, 1e-2, 1e2],
                    outer_folds=5,
                    inner_folds=3,
                    seed=4,
                )

                self.assertIn(result.best_alpha, [1e-4, 1e-2, 1e2])
                self.assertGreater(result.outer_scores.mean(), 0.99)
                self.assertEqual(len(result.outer_selected_alphas), 5)
                self.assertEqual(result.n_targets, 3)
                self.assertEqual(result.outer_target_scores.shape, (5, 3))
                np.testing.assert_allclose(
                    result.outer_scores,
                    np.nanmean(result.outer_target_scores, axis=1),
                )

    def test_rejects_unknown_regression(self):
        with self.assertRaisesRegex(ValueError, "ridge.*lasso"):
            nested_encoding_cv(
                self.features,
                self.responses,
                regression="elasticnet",
                alphas=[0.1],
            )


class EncodingNoiseNormalizationTests(unittest.TestCase):
    def test_result_schema_keeps_raw_and_normalized_metrics(self):
        self.assertEqual(DEFAULT_MIN_TRIPLE_N_RELIABILITY, 0.4)
        self.assertIn("mean_encoding_score", RESULT_COLUMNS)
        self.assertIn("noise_ceiling_r", RESULT_COLUMNS)
        self.assertIn("noise_ceiling_threshold", RESULT_COLUMNS)
        self.assertIn("mean_noise_normalized_r", RESULT_COLUMNS)
        self.assertIn("mean_noise_normalized_r2", RESULT_COLUMNS)
        self.assertIn("n_ceiling_targets", RESULT_COLUMNS)

    def test_summary_normalizes_targets_before_averaging_folds(self):
        result = EncodingCVResult(
            outer_scores=np.array([0.4, 0.8]),
            outer_target_scores=np.array([
                [0.3, 0.4, 0.5],
                [0.6, 0.8, 1.0],
            ]),
            outer_selected_alphas=(0.1, 0.1),
            best_alpha=0.1,
            alpha_selection_stability=1.0,
            n_targets=3,
        )

        summary = _noise_normalized_result(
            result,
            noise_ceiling=np.array([0.36, 0.64, 0.2]),
            minimum_ceiling=0.4,
        )

        (
            ceiling_r,
            threshold,
            mean_r,
            std_r,
            mean_r2,
            std_r2,
            n_targets,
        ) = summary
        self.assertAlmostEqual(ceiling_r, 0.8)
        self.assertAlmostEqual(threshold, 0.4)
        self.assertAlmostEqual(mean_r, 0.75)
        self.assertAlmostEqual(std_r, 0.25)
        self.assertAlmostEqual(mean_r2, 0.625)
        self.assertAlmostEqual(std_r2, 0.375)
        self.assertEqual(n_targets, 1)

    def test_summary_rejects_misaligned_ceiling_vector(self):
        result = EncodingCVResult(
            outer_scores=np.array([0.2]),
            outer_target_scores=np.array([[0.1, 0.3]]),
            outer_selected_alphas=(0.1,),
            best_alpha=0.1,
            alpha_selection_stability=1.0,
            n_targets=2,
        )

        with self.assertRaisesRegex(ValueError, "one value per neural target"):
            _noise_normalized_result(result, np.array([0.5]))


class _FakeAnalyzer(ModelAnalysisBase):
    def __init__(self, features):
        super().__init__(triple_n_path="unused")
        self._features = features
        self.feature_calls = 0

    def aligned_stimuli(self, algonauts, triple_n, shared_ids, subjects):
        indices = list(range(1, len(self._features) + 1))
        return indices, indices

    def features(self, indices=None):
        self.feature_calls += 1
        return self._features[np.asarray(indices)]


class _FakeAlgonauts:
    ALGO_ROIS = ["visual"]

    def __init__(self, responses):
        self.responses = responses

    def response_matrix(self, subject, indices=None, roi=None):
        selected = self.responses[np.asarray(indices) - 1]
        return selected[:, :2], selected[:, 2:]

    def roi_mask(self, roi, subject):
        return np.ones(2, dtype=bool), np.ones(1, dtype=bool)

    def noise_ceiling(self, subject, roi=None):
        return np.array([0.25, 0.64]), np.array([1.0])


class _FakeTripleN:
    def __init__(self, responses):
        self.responses = responses
        self.units = pd.DataFrame({
            "macaque": ["M1"] * 6,
            "area_label": ["V1", "V1", "V1", "IT", "IT", "IT"],
            "region": ["EVC", "EVC", "EVC", "IT", "IT", "IT"],
            "unit_type": [1, 1, 1, 2, 2, 2],
            "reliability_best": [0.81, 0.64, 0.20, 0.64, 0.49, 0.10],
        })

    def _mask(self, **filters):
        mask = np.ones(len(self.units), dtype=bool)
        for name, value in filters.items():
            if value is not None:
                mask &= self.units[name].to_numpy() == value
        return mask

    def response_matrix(self, indices=None, **filters):
        mask = self._mask(**filters)
        if mask.sum() < 2:
            raise ValueError("no matching units")
        return self.responses[np.asarray(indices) - 1][:, mask]

    def unit_metadata(self, **filters):
        return self.units.loc[self._mask(**filters)].reset_index(drop=True)


class AnalyzerEncodingIntegrationTests(unittest.TestCase):
    def test_encoding_tables_extract_features_once_and_report_groups(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(30, 3))
        algonauts_responses = features @ rng.normal(size=(3, 3))
        triple_n_responses = features @ rng.normal(size=(3, 6))
        analyzer = _FakeAnalyzer(features)

        tables = analyzer.encoding_tables(
            _FakeAlgonauts(algonauts_responses),
            _FakeTripleN(triple_n_responses),
            shared_ids=list(range(1, 31)),
            subjects=[1],
            rois=["visual"],
            regression="ridge",
            alphas=[1e-4, 1e2],
            outer_folds=3,
            inner_folds=2,
        )

        self.assertEqual(analyzer.feature_calls, 1)
        self.assertEqual(tables["algonauts"].loc["visual", "n_targets"], 3)
        self.assertEqual(list(tables["triple_n"].index), ["IT", "V1"])
        self.assertEqual(tables["triple_n"].index.name, "area_label")
        self.assertEqual(tables["triple_n"].loc["V1", "n_targets"], 2)
        self.assertEqual(
            tables["triple_n"].loc["V1", "n_ceiling_targets"], 2
        )
        self.assertEqual(
            tables["triple_n"].loc[
                "V1", "noise_ceiling_threshold"
            ],
            DEFAULT_MIN_TRIPLE_N_RELIABILITY,
        )
        self.assertEqual(
            tables["algonauts"].loc["visual", "n_ceiling_targets"], 3
        )
        self.assertGreater(
            tables["algonauts"].loc["visual", "mean_encoding_score"], 0.99
        )
        self.assertGreater(
            tables["triple_n"].loc["IT", "mean_encoding_score"], 0.99
        )
        self.assertGreater(
            tables["algonauts"].loc[
                "visual", "mean_noise_normalized_r"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
