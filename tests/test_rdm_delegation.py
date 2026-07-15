import unittest
from unittest.mock import Mock, patch

import numpy as np

from psm_final.analysis.correlating import correlation_rdm
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


if __name__ == "__main__":
    unittest.main()
