"""Backward-compatible adapter for the analyzer-native encoding API."""

from psm_final.analysis.encoding import prediction_correlations


class EncodingModel:
    """Compatibility wrapper around a ``ModelAnalysisBase`` instance.

    Encoding is implemented directly by ``ModelAnalysisBase``. New code should
    call ``analyzer.encoding_algonauts()``, ``analyzer.encoding_triple_n()``, or
    ``analyzer.encoding_tables()``.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    @staticmethod
    def _prediction_correlations(predicted, observed):
        return prediction_correlations(predicted, observed)

    def features(self, indices=None):
        return self.analyzer.features(indices=indices)

    def encoding_algonauts(self, algonauts, triple_n, shared_ids, **kwargs):
        return self.analyzer.encoding_algonauts(
            algonauts, triple_n, shared_ids, **kwargs
        )

    def encoding_triple_n(self, triple_n, algonauts, shared_ids, **kwargs):
        return self.analyzer.encoding_triple_n(
            triple_n, algonauts, shared_ids, **kwargs
        )

    def encoding_tables(self, algonauts, triple_n, shared_ids, **kwargs):
        return self.analyzer.encoding_tables(
            algonauts, triple_n, shared_ids, **kwargs
        )
