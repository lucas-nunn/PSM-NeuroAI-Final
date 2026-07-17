from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from psm_final.analysis.model import ModelAnalysisBase


class BasicBaselineAnalysis(ModelAnalysisBase):
    DEFAULT_PCA_COMPONENTS = 50
    DEFAULT_BATCH_SIZE = 128

    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        def pixel_factory():
            return cls(triple_n_path=triple_n_path, kind="pixel")

        # Flattened full-resolution pixels can materialize several GB once the
        # encoding solver promotes features to float64. Keep the baseline available
        # through an explicit --models Pixel selection, but out of default batches.
        pixel_factory.encoding_default = False

        def pca_factory():
            return cls(
                triple_n_path=triple_n_path,
                kind="pca",
                n_components=cls.DEFAULT_PCA_COMPONENTS,
            )

        # The current PCA baseline fits its transform before CV splits exist.
        pca_factory.encoding_cv_safe = False
        return [
            ("Pixel", pixel_factory),
            (
                f"PCA {cls.DEFAULT_PCA_COMPONENTS}",
                pca_factory,
            ),
        ]

    def __init__(self, triple_n_path, *, kind="pixel", n_components=DEFAULT_PCA_COMPONENTS,
                 batch_size=DEFAULT_BATCH_SIZE):
        super().__init__(triple_n_path)
        self.kind = kind
        # PCA is fitted from ``self.images`` before encoding CV splits are created,
        # so including it would leak held-out stimuli into the feature transform.
        self.encoding_cv_safe = kind != "pca"
        self.n_components = n_components
        self.batch_size = batch_size
        self._scaler = None
        self._pca = None
        self._fit_key = None

    @staticmethod
    def _flatten_image(image):
        return np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1) / 255.0

    @staticmethod
    def _batch_slices(n_items, batch_size, min_size=1):
        start = 0
        while start < n_items:
            stop = min(start + batch_size, n_items)
            if n_items - stop < min_size:
                stop = n_items
            yield slice(start, stop)
            start = stop

    def _load_batch(self, paths):
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rows.append(self._flatten_image(image))
        return np.stack(rows, axis=0)

    def _fit_pca(self):
        fit_key = tuple(path.name for path in self.images)
        if self._fit_key == fit_key:
            return

        n_items = len(self.images)
        if n_items < 2:
            raise ValueError("need at least 2 images to build a PCA baseline")

        self._scaler = StandardScaler()
        for batch_slice in self._batch_slices(n_items, self.batch_size):
            self._scaler.partial_fit(self._load_batch(self.images[batch_slice]))

        n_components = min(self.n_components, n_items)
        self._pca = IncrementalPCA(
            n_components=n_components,
            batch_size=max(self.batch_size, n_components),
        )
        for batch_slice in self._batch_slices(
            n_items,
            max(self.batch_size, n_components),
            min_size=n_components,
        ):
            batch = self._scaler.transform(self._load_batch(self.images[batch_slice]))
            self._pca.partial_fit(batch)
        self._fit_key = fit_key

    def embedding(self, image):
        pixels = self._flatten_image(image)
        if self.kind == "pixel":
            return pixels
        if self.kind != "pca":
            raise ValueError(f"unknown baseline kind: {self.kind!r}")

        self._fit_pca()
        return self._pca.transform(self._scaler.transform(pixels[None, :]))[0]
