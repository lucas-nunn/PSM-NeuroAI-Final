import unittest

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from psm_final.analysis.encoding import nested_encoding_cv
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

    def test_rejects_unknown_regression(self):
        with self.assertRaisesRegex(ValueError, "ridge.*lasso"):
            nested_encoding_cv(
                self.features,
                self.responses,
                regression="elasticnet",
                alphas=[0.1],
            )


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


class _FakeTripleN:
    def __init__(self, responses):
        self.responses = responses
        self.units = pd.DataFrame({
            "macaque": ["M1", "M1"],
            "area_label": ["IT", "IT"],
        })

    def response_matrix(self, macaque=None, area_label=None, indices=None, **filters):
        if macaque not in (None, "M1") or area_label != "IT":
            raise ValueError("no matching units")
        return self.responses[np.asarray(indices) - 1]


class AnalyzerEncodingIntegrationTests(unittest.TestCase):
    def test_encoding_tables_extract_features_once_and_report_groups(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(30, 3))
        algonauts_responses = features @ rng.normal(size=(3, 3))
        triple_n_responses = features @ rng.normal(size=(3, 2))
        analyzer = _FakeAnalyzer(features)

        tables = analyzer.encoding_tables(
            _FakeAlgonauts(algonauts_responses),
            _FakeTripleN(triple_n_responses),
            shared_ids=list(range(1, 31)),
            subjects=[1],
            rois=["visual"],
            area_labels=["IT"],
            regression="ridge",
            alphas=[1e-4, 1e2],
            outer_folds=3,
            inner_folds=2,
        )

        self.assertEqual(analyzer.feature_calls, 1)
        self.assertEqual(tables["algonauts"].loc["visual", "n_targets"], 3)
        self.assertEqual(tables["triple_n"].loc["IT", "n_targets"], 2)
        self.assertGreater(
            tables["algonauts"].loc["visual", "mean_encoding_score"], 0.99
        )
        self.assertGreater(
            tables["triple_n"].loc["IT", "mean_encoding_score"], 0.99
        )


if __name__ == "__main__":
    unittest.main()
