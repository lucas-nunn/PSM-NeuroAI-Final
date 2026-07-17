import unittest

import numpy as np

from psm_final.analysis.correlating import noise_normalize_correlations


class NoiseNormalizeCorrelationsTests(unittest.TestCase):
    def test_returns_signed_and_squared_fractions_of_ceiling(self):
        correlations = np.array([0.3, -0.4])
        ceilings_r = np.array([0.6, 0.8])

        signed, squared = noise_normalize_correlations(
            correlations, ceilings_r
        )

        np.testing.assert_allclose(signed, [0.5, -0.5])
        np.testing.assert_allclose(squared, [0.25, 0.25])

    def test_accepts_reliability_or_explainable_variance_ceilings(self):
        correlations = np.array([[0.3, -0.4], [0.6, 0.8]])
        ceilings_r2 = np.array([0.36, 0.64])

        signed, squared = noise_normalize_correlations(
            correlations,
            ceilings_r2,
            ceiling_squared=True,
        )

        np.testing.assert_allclose(signed, [[0.5, -0.5], [1.0, 1.0]])
        np.testing.assert_allclose(squared, [[0.25, 0.25], [1.0, 1.0]])

    def test_invalid_or_nonpositive_ceilings_are_nan(self):
        signed, squared = noise_normalize_correlations(
            np.ones(5),
            np.array([np.nan, np.inf, 0.0, -0.2, 0.25]),
            ceiling_squared=True,
        )

        self.assertTrue(np.isnan(signed[:4]).all())
        self.assertTrue(np.isnan(squared[:4]).all())
        self.assertEqual(signed[4], 2.0)
        self.assertEqual(squared[4], 4.0)

    def test_does_not_clip_above_ceiling(self):
        signed, squared = noise_normalize_correlations(0.75, 0.5)

        self.assertAlmostEqual(float(signed), 1.5)
        self.assertAlmostEqual(float(squared), 2.25)


if __name__ == "__main__":
    unittest.main()
