"""
encoding.py

Encoding models: model features -> real brain responses (regression)
EncodingModel WRAPS an
existing analyzer instance (any ModelAnalysisBase subclass: BetaVAEAnalysis,
CNNAnalysis, ResNet50Analysis, ...) and reimplements only what it needs
(the stimulus-loading loop) independently, using that analyzer's already-public
embedding() method and triple_n_path attribute.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.linear_model import Lasso


class EncodingModel:
    """Fits and evaluates a linear encoding model on top of any RSA analyzer.

    Usage:
        analyzer = CNNAnalysis(triple_n_path=..., model_path=...)  # unchanged
        enc = EncodingModel(analyzer)
        scores = enc.encoding_algonauts(algonauts, triple_n, shared_ids)
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def features(self, indices=None):
        """Model embeddings for the shared stimulus set, stacked into one array.
        """
        stimuli_dir = Path(self.analyzer.triple_n_path) / "others" / "StimuliNNN"
        digit_only = re.compile(r"^\d+$")
        images = sorted(
            (p for p in stimuli_dir.glob("*.bmp") if digit_only.fullmatch(p.stem)),
            key=lambda p: int(p.stem),
        )
        if indices is not None:
            images = [images[i] for i in indices]

        embeddings = []
        for img_path in images:
            with Image.open(img_path) as img:
                embeddings.append(self.analyzer.embedding(img))
        return np.stack(embeddings, axis=0)

    def encoding_algonauts(self, algonauts, triple_n, shared_ids, subjects=range(1, 9),
                           rois=None, test_split=0.1, seed=42, alpha=1.0, regression=None):
        """Fit a Lasso encoding model (model features -> fMRI) and report mean
        per-ROI prediction accuracy (Pearson r on held-out images), averaged
        across subjects.

        Returns a DataFrame indexed by ROI with columns mean_encoding_score /
        std_encoding_score (across subjects).
        """
        if regression is None:
            regression = Lasso(alpha=alpha)
        rois = list(algonauts.ALGO_ROIS if rois is None else rois)

        # Reuses the colleague's aligned_stimuli() staticmethod unchanged --
        # available on self.analyzer since it inherits from ModelAnalysisBase.
        nsd_ids, stim_index = self.analyzer.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_indices = np.asarray(stim_index) - 1

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(nsd_ids))
        n_test = max(1, int(round(len(nsd_ids) * test_split)))
        test_pos, train_pos = order[:n_test], order[n_test:]

        nsd_train = [nsd_ids[i] for i in train_pos]
        nsd_test = [nsd_ids[i] for i in test_pos]
        features_train = self.features(indices=model_indices[train_pos])
        features_test = self.features(indices=model_indices[test_pos])

        per_subject_roi_scores = []
        for subj in subjects:
            lh_train, rh_train = algonauts.response_matrix(subj, indices=nsd_train)
            lh_test, rh_test = algonauts.response_matrix(subj, indices=nsd_test)

            reg_lh = clone(regression).fit(features_train, lh_train)
            reg_rh = clone(regression).fit(features_train, rh_train)
            lh_pred, rh_pred = reg_lh.predict(features_test), reg_rh.predict(features_test)

            lh_corrs = np.array([pearsonr(lh_pred[:, v], lh_test[:, v])[0] for v in range(lh_pred.shape[1])])
            rh_corrs = np.array([pearsonr(rh_pred[:, v], rh_test[:, v])[0] for v in range(rh_pred.shape[1])])

            roi_scores = {}
            for roi in rois:
                lh_roi, rh_roi = algonauts.roi_mask(roi, subj)
                vertex_scores = np.concatenate([lh_corrs[lh_roi], rh_corrs[rh_roi]])
                if vertex_scores.size:
                    roi_scores[roi] = float(np.nanmean(vertex_scores))
            per_subject_roi_scores.append(roi_scores)

        rows = []
        for roi in rois:
            vals = [d[roi] for d in per_subject_roi_scores if roi in d]
            if vals:
                rows.append((roi, float(np.mean(vals)), float(np.std(vals))))

        return pd.DataFrame(
            rows, columns=["roi", "mean_encoding_score", "std_encoding_score"]
        ).set_index("roi")

    def encoding_triple_n(self, triple_n, algonauts, shared_ids, subjects=range(1, 9),
                          area_labels=None, test_split=0.1, seed=42, alpha=1.0, regression=None):
        """Fit a Lasso encoding model (model features -> macaque unit responses)
        and report mean per-area-label prediction accuracy, averaged across
        macaques.
        """
        if regression is None:
            regression = Lasso(alpha=alpha)
        if area_labels is None:
            area_labels = sorted(triple_n.units["area_label"].unique())

        nsd_ids, stim_index = self.analyzer.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_indices = np.asarray(stim_index) - 1

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(nsd_ids))
        n_test = max(1, int(round(len(nsd_ids) * test_split)))
        test_pos, train_pos = order[:n_test], order[n_test:]

        stim_train = [stim_index[i] for i in train_pos]
        stim_test = [stim_index[i] for i in test_pos]
        features_train = self.features(indices=model_indices[train_pos])
        features_test = self.features(indices=model_indices[test_pos])

        macaques = sorted(triple_n.units["macaque"].unique())
        per_macaque_area_scores = []
        for m in macaques:
            area_scores = {}
            for label in area_labels:
                try:
                    resp_train = triple_n.response_matrix(macaque=m, area_label=label, indices=stim_train)
                    resp_test = triple_n.response_matrix(macaque=m, area_label=label, indices=stim_test)
                except ValueError:
                    continue  # fewer than 2 units for this macaque/area combo

                reg = clone(regression).fit(features_train, resp_train)
                pred = reg.predict(features_test)
                corrs = np.array([pearsonr(pred[:, u], resp_test[:, u])[0] for u in range(pred.shape[1])])
                if corrs.size:
                    area_scores[label] = float(np.nanmean(corrs))
            per_macaque_area_scores.append(area_scores)

        rows = []
        for label in area_labels:
            vals = [d[label] for d in per_macaque_area_scores if label in d]
            if vals:
                rows.append((label, float(np.mean(vals)), float(np.std(vals))))

        return pd.DataFrame(
            rows, columns=["area_label", "mean_encoding_score", "std_encoding_score"]
        ).set_index("area_label")