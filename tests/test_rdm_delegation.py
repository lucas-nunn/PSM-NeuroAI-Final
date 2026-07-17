import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import spearmanr

from psm_final.analysis.correlating import correlation_rdm
from psm_final.analysis.model import (
    ModelAnalysisBase,
    _summarize_individual_rdms,
)
from psm_final.dataset.algonauts import Algonauts
from psm_final.dataset.triple_n import TripleN


class RdmDelegationTests(unittest.TestCase):
    def test_algonauts_compute_rdm_uses_response_matrix(self):
        dataset = Algonauts.__new__(Algonauts)
        left = np.array([[1.0, 2.0], [3.0, 4.0]])
        right = np.array([[5.0], [6.0]])
        dataset.response_matrix = Mock(return_value=(left, right))

        with patch(
            "psm_final.dataset.algonauts.correlation_rdm", wraps=correlation_rdm
        ) as compute:
            result = dataset.compute_rdm(2, indices=[11, 12], roi="V1v")

        dataset.response_matrix.assert_called_once_with(
            2, indices=[11, 12], roi="V1v"
        )
        patterns = compute.call_args.args[0]
        expected_patterns = np.concatenate((left, right), axis=1)
        np.testing.assert_array_equal(patterns, expected_patterns)
        np.testing.assert_allclose(result, correlation_rdm(expected_patterns))
        self.assertTrue(compute.call_args.kwargs["condensed"])

    def test_triple_n_compute_rdm_uses_response_matrix(self):
        dataset = TripleN.__new__(TripleN)
        patterns = np.array([[1.0, 2.0], [3.0, 4.0]])
        dataset.response_matrix = Mock(return_value=patterns)

        with patch(
            "psm_final.dataset.triple_n.correlation_rdm", wraps=correlation_rdm
        ) as compute:
            result = dataset.compute_rdm(
                macaque="M1",
                area=4,
                category="F",
                region="IT",
                preference="B",
                indices=[3, 1],
                unit_type=2,
            )

        dataset.response_matrix.assert_called_once_with(
            macaque="M1",
            area=4,
            category="F",
            region="IT",
            preference="B",
            indices=[3, 1],
            unit_type=2,
        )
        np.testing.assert_array_equal(compute.call_args.args[0], patterns)
        np.testing.assert_allclose(result, correlation_rdm(patterns))


class AlgonautsNoiseCeilingTests(unittest.TestCase):
    def test_discovers_sibling_test_release_and_applies_roi_masks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_root = root / "train"
            training_root.mkdir()
            ceiling_root = (
                root
                / "test"
                / "subj01"
                / "test_split"
                / "noise_ceiling"
            )
            ceiling_root.mkdir(parents=True)
            np.save(ceiling_root / "lh_noise_ceiling.npy", [0.1, 0.2, 0.3])
            np.save(ceiling_root / "rh_noise_ceiling.npy", [0.4, 0.5])

            dataset = Algonauts(str(training_root), nsd_indices=[])
            dataset.roi_mask = Mock(return_value=(
                np.array([True, False, True]),
                np.array([False, True]),
            ))

            left, right = dataset.noise_ceiling(1, roi="V1v")

            self.assertEqual(dataset.noise_ceiling_dir, str((root / "test").resolve()))
            dataset.roi_mask.assert_called_once_with("V1v", 1)
            np.testing.assert_allclose(left, [0.1, 0.3])
            np.testing.assert_allclose(right, [0.5])

    def test_missing_release_has_actionable_error(self):
        with TemporaryDirectory() as tmp:
            dataset = Algonauts(tmp, nsd_indices=[])

            with self.assertRaisesRegex(FileNotFoundError, "noise_ceiling_dir"):
                dataset.noise_ceiling(1)


class RsaNoiseNormalizationTests(unittest.TestCase):
    def test_raw_group_mean_is_preserved_and_normalization_uses_individuals(self):
        model_rdm = np.arange(6, dtype=float)
        individual_rdms = np.array([
            [0.0, 1.0, 2.0, 3.0, 5.0, 4.0],
            [0.0, 2.0, 1.0, 5.0, 3.0, 4.0],
        ])
        table = ModelAnalysisBase._correlate(
            model_rdm,
            {"group": _summarize_individual_rdms(individual_rdms)},
            {"group": (0.6, 0.8)},
            ["group"],
        )

        expected_individual_mean = np.mean([
            spearmanr(model_rdm, rdm).statistic
            for rdm in individual_rdms
        ])
        expected_group_mean = spearmanr(
            model_rdm, individual_rdms.mean(axis=0)
        ).statistic
        self.assertAlmostEqual(
            table.loc["group", "spearman_rho"],
            expected_group_mean,
        )
        self.assertAlmostEqual(
            table.loc["group", "spearman_rho_individual_mean"],
            expected_individual_mean,
        )
        self.assertAlmostEqual(
            table.loc["group", "noise_normalized_spearman_rho"],
            expected_individual_mean / 0.8,
        )

    def test_correlate_retains_raw_rho_and_adds_unclipped_normalization(self):
        model_rdm = np.array([0.0, 1.0, 2.0, 3.0])
        table = ModelAnalysisBase._correlate(
            model_rdm,
            {
                "above ceiling": model_rdm.copy(),
                "missing ceiling": model_rdm[::-1],
            },
            {"above ceiling": (0.4, 0.5)},
            ["above ceiling", "missing ceiling"],
            index_name="roi",
        )

        self.assertAlmostEqual(
            table.loc["above ceiling", "spearman_rho"], 1.0
        )
        self.assertAlmostEqual(
            table.loc[
                "above ceiling", "noise_normalized_spearman_rho"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            table.loc[
                "above ceiling", "noise_normalized_spearman_rho2"
            ],
            4.0,
        )
        self.assertTrue(
            np.isnan(
                table.loc[
                    "missing ceiling", "noise_normalized_spearman_rho"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
