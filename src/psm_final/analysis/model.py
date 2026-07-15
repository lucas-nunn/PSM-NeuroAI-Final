import re

import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from scipy.stats import spearmanr

from psm_final.analysis.correlating import correlation_rdm, noise_ceiling
from psm_final.analysis.encoding import (
    encoding_algonauts as _encoding_algonauts,
    encoding_tables as _encoding_tables,
    encoding_triple_n as _encoding_triple_n,
    prediction_correlations,
)


class ModelAnalysisBase():
    encoding_cv_safe = True

    # Triple-N groupings each model RDM is compared against, beyond the coarse area.
    # `unit_type` is the per-neuron firing-rate cluster (paper Fig. 3 k-means PSTH type
    # 1/2/3); area and region are each shown coarse AND split by that cluster:
    #   ("area_label",)              -- coarse area (Face/Body/Object/Color/V1/...)
    #   ("area_label", "unit_type")  -- each area split by firing-rate cluster
    #   ("region",)                  -- cortical region (IT vs EVC)
    #   ("region", "unit_type")      -- each region split by firing-rate cluster
    TRIPLE_N_SEGMENTATIONS = (
        ("area_label",), ("area_label", "unit_type"),
        ("region",), ("region", "unit_type"),
    )

    def __init__(self, triple_n_path):
        self.triple_n_path = triple_n_path

    def embedding(self, image):
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        """Enumerate ready-to-run instances of this analyzer for the RSA runner.

        Returns a list of ``(label, factory)`` pairs, where ``factory()`` builds one
        configured analyzer instance (typically one per trained checkpoint found
        under ``checkpoints_root``). Construction is deferred into ``factory`` so
        discovery stays cheap and the runner loads at most one model into memory at
        a time.

        The base returns ``[]`` -- a subclass opts in by locating its own
        checkpoints (see :meth:`BetaVAEAnalysis.discover` for the worked example).
        The runner (`psm_final.analysis.runner`) calls this on every concrete
        subclass to collect the models to compare against the brain data.
        """
        return []

    def features(self, indices=None):
        """Return model embeddings for selected shared stimuli in numeric order."""
        self.shared_stimuli_dir = Path(self.triple_n_path) / "others" / "StimuliNNN"
        digit_only = re.compile(r"^\d+$")

        # Sort by the numeric filename so images[i] is deterministically the
        # stimulus at position i+1 (StimuliNNN/0001.bmp..1000.bmp); glob order is
        # filesystem-dependent, which would otherwise misalign indices=... RDMs.
        self.images = sorted(
            (p for p in self.shared_stimuli_dir.glob("*.bmp")
             if digit_only.fullmatch(p.stem)),
            key=lambda p: int(p.stem),
        )

        if indices is not None:
            self.images = [self.images[i] for i in indices]

        embeddings = []
        for img_path in self.images:
            with Image.open(img_path) as img:
                embeddings.append(self.embedding(img))
        return np.stack(embeddings, axis=0)

    def rdm(self, indices=None):
        return correlation_rdm(self.features(indices=indices))

    # ------------------------------------------------------------------ #
    # RSA: model RDM vs. brain RDMs (Algonauts ROIs / Triple-N areas)
    # ------------------------------------------------------------------ #
    # These methods are model-agnostic: they only rely on `self.rdm()`, so any
    # subclass (Beta-VAE, MLP, CNN, ...) that implements `embedding()` gets them
    # for free. The construction mirrors the tables at the end of `beta_vae.ipynb`
    # and `data_exploration.ipynb`: every RDM -- the model's, each ROI's, each
    # Triple-N area's -- is computed over the SAME shared stimuli in one shared
    # order, so every pair lines up for a Spearman comparison.

    @staticmethod
    def aligned_stimuli(algonauts, triple_n, shared_ids, subjects=range(1, 9)):
        """Stimuli usable by all three modalities, in one shared order.

        Returns ``(nsd_ids, stim_index)`` for the images present in EVERY Algonauts
        `subjects` split AND carrying a Triple-N mapping:

        - ``nsd_ids``    -- 1-based NSD 73k ids; pass as ``indices`` to
          :meth:`Algonauts.compute_rdm`.
        - ``stim_index`` -- 1-based Triple-N ``stim_index`` (1..1000); pass as
          ``indices`` to :meth:`TripleN.compute_rdm`. Subtract 1 for :meth:`rdm`
          (0-based positions into ``StimuliNNN``).

        The two lists are element-aligned: ``nsd_ids[k]``, ``stim_index[k]`` and
        ``StimuliNNN[stim_index[k] - 1]`` all name the same image.
        """
        matched = [set(algonauts.shared_stimuli_indices(s, shared_ids)[0]) for s in subjects]
        common = sorted(set.intersection(*matched))

        stim_idx = triple_n.nsd_to_stim_index(common)   # 1-based, None where unmapped
        keep = [k for k, s in enumerate(stim_idx) if s is not None]
        nsd_ids = [common[k] for k in keep]
        stim_index = [stim_idx[k] for k in keep]
        return nsd_ids, stim_index

    # ------------------------------------------------------------------ #
    # Encoding: model features -> held-out neural responses
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prediction_correlations(predicted, observed):
        return prediction_correlations(predicted, observed)

    def encoding_algonauts(
        self,
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
        """Nested-CV encoding scores for Algonauts ROIs.

        One alpha is selected per ROI using the mean correlation across all of
        that ROI's vertices and subjects.
        """
        return _encoding_algonauts(
            self,
            algonauts,
            triple_n,
            shared_ids,
            subjects=subjects,
            rois=rois,
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
            progress=progress,
        )

    def encoding_triple_n(
        self,
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
        """Nested-CV encoding scores for Triple-N areas.

        One alpha is selected per area using the mean correlation across all of
        that area's units and macaques.
        """
        return _encoding_triple_n(
            self,
            triple_n,
            algonauts,
            shared_ids,
            subjects=subjects,
            area_labels=area_labels,
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
            progress=progress,
        )

    def encoding_tables(
        self,
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
        """Both encoding tables, extracting this model's features only once."""
        return _encoding_tables(
            self,
            algonauts,
            triple_n,
            shared_ids,
            subjects=subjects,
            rois=rois,
            area_labels=area_labels,
            regression=regression,
            alphas=alphas,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
            progress=progress,
        )

    def rsa_algonauts(self, algonauts, triple_n, shared_ids, subjects=range(1, 9), rois=None):
        """Spearman-correlate this model's RDM against every Algonauts fMRI ROI.

        For each ROI, the per-subject RDMs are averaged into a group RDM (dropping
        subjects where the ROI is empty) and correlated with the model RDM. Returns
        a DataFrame indexed by ROI with columns ``spearman_rho`` and the per-ROI
        across-subject ``noise_ceiling_low`` / ``noise_ceiling_high``.
        """
        rois = list(algonauts.ALGO_ROIS if rois is None else rois)
        nsd_ids, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        group_rdms, group_nc = self._algonauts_group_rdms(algonauts, nsd_ids, subjects, rois)
        return self._correlate(model_rdm, group_rdms, group_nc, rois, index_name="roi")

    def rsa_triple_n(self, triple_n, algonauts, shared_ids, subjects=range(1, 9),
                     groupby=("area_label",), groups=None):
        """Spearman-correlate this model's RDM against each Triple-N unit group.

        ``groupby`` selects the segmentation (see :meth:`_triple_n_group_rdms`): by
        coarse area (default), by area x firing-rate cluster (``unit_type``), by
        region, or any other ``units`` column(s). One RDM is built per group
        (units pooled across
        macaques) and correlated with the model RDM; ``groups`` optionally restricts
        to a subset of group labels. Returns a DataFrame indexed by the group label
        with columns ``spearman_rho`` and the per-group across-macaque
        ``noise_ceiling_low`` / ``noise_ceiling_high``.
        """
        _, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        group_rdms, group_nc = self._triple_n_group_rdms(triple_n, stim_index, groupby, groups)
        return self._correlate(model_rdm, group_rdms, group_nc, list(group_rdms),
                               index_name=" | ".join(groupby))

    def rsa_triple_n_segmented(self, triple_n, algonauts, shared_ids, subjects=range(1, 9),
                               segmentations=None):
        """Every Triple-N segmentation at once, over one shared stimulus set.

        Builds the model RDM a single time and correlates it against each grouping in
        ``segmentations`` (default :attr:`TRIPLE_N_SEGMENTATIONS`: coarse area, area x
        firing-rate cluster, region). Returns ``{index_name: DataFrame}`` keyed by the
        grouping (``"area_label"``, ``"area_label | unit_type"``, ``"region"``), each
        frame exactly what :meth:`rsa_triple_n` returns.
        """
        segmentations = self.TRIPLE_N_SEGMENTATIONS if segmentations is None else segmentations
        _, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        tables = {}
        for groupby in segmentations:
            group_rdms, group_nc = self._triple_n_group_rdms(triple_n, stim_index, groupby)
            tables[" | ".join(groupby)] = self._correlate(
                model_rdm, group_rdms, group_nc, list(group_rdms), index_name=" | ".join(groupby))
        return tables

    def rsa_tables(self, algonauts, triple_n, shared_ids, subjects=range(1, 9),
                   rois=None, segmentations=None):
        """Both RSA modalities at once, over one shared stimulus set.

        Convenience wrapper around :meth:`rsa_algonauts` and
        :meth:`rsa_triple_n_segmented` that aligns the stimuli and computes the model
        RDM a single time. Returns ``{"algonauts": <DataFrame>, "triple_n":
        {index_name: <DataFrame>, ...}}`` -- one Triple-N frame per segmentation
        (default: coarse area, area x firing-rate cluster, region).
        """
        rois = list(algonauts.ALGO_ROIS if rois is None else rois)
        segmentations = self.TRIPLE_N_SEGMENTATIONS if segmentations is None else segmentations

        nsd_ids, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        algo_rdms, algo_nc = self._algonauts_group_rdms(algonauts, nsd_ids, subjects, rois)
        tn_tables = {}
        for groupby in segmentations:
            tn_rdms, tn_nc = self._triple_n_group_rdms(triple_n, stim_index, groupby)
            tn_tables[" | ".join(groupby)] = self._correlate(
                model_rdm, tn_rdms, tn_nc, list(tn_rdms), index_name=" | ".join(groupby))
        return {
            "algonauts": self._correlate(model_rdm, algo_rdms, algo_nc, rois, index_name="roi"),
            "triple_n": tn_tables,
        }

    # --- building blocks ------------------------------------------------ #

    @staticmethod
    def _algonauts_group_rdms(algonauts, nsd_ids, subjects, rois):
        """Per-ROI group-mean RDM + across-subject noise ceiling.

        NOTE: ``compute_rdm`` reloads the subject fMRI on each call, so this
        re-reads per (subject, ROI).
        """
        group_rdms, group_nc = {}, {}
        for roi in rois:
            per_subj = [algonauts.compute_rdm(subject=s, indices=nsd_ids, roi=roi) for s in subjects]
            per_subj = [r for r in per_subj if r.std() > 0]   # drop subjects where the ROI is empty
            if len(per_subj) < 2:                             # need >=2 subjects for a noise ceiling
                continue
            per_subj = np.vstack(per_subj)
            group_rdms[roi] = per_subj.mean(axis=0)
            group_nc[roi] = noise_ceiling(per_subj)           # (lower, upper)
        return group_rdms, group_nc

    @staticmethod
    def _group_label(values):
        """Composite label for a groupby combination: the bare value for a single
        column, values joined by ' | ' for several (order == the groupby order)."""
        values = tuple(values)
        return str(values[0]) if len(values) == 1 else " | ".join(map(str, values))

    @staticmethod
    def _triple_n_group_rdms(triple_n, stim_index, groupby=("area_label",), groups=None):
        """Per-group RDM (units pooled across macaques) + across-macaque noise ceiling.

        ``groupby`` names one or more ``units`` columns; one RDM is built per unique
        combination of their values, pooling that group's units across macaques:
        ``("area_label",)`` -> one RDM per coarse area (the original behaviour);
        ``("area_label", "unit_type")`` -> each area split by its units' firing-rate
        cluster (1/2/3); ``("region",)`` -> IT vs EVC. Group labels come from
        :meth:`_group_label` (bare value for one column, ' | '-joined otherwise).
        ``groups`` optionally restricts the result to that set of labels.
        """
        groupby = list(groupby)
        macaques = sorted(triple_n.units["macaque"].unique())
        # every value combination present in the data, in a stable (sorted) order
        combos = (triple_n.units[groupby].drop_duplicates()
                  .sort_values(groupby).itertuples(index=False, name=None))
        group_rdms, group_nc = {}, {}
        for combo in combos:
            label = ModelAnalysisBase._group_label(combo)
            if groups is not None and label not in groups:
                continue
            filt = dict(zip(groupby, combo))                  # column -> value for compute_rdm
            try:
                group_rdms[label] = triple_n.compute_rdm(indices=stim_index, **filt)
            except ValueError:
                continue                                      # fewer than 2 units for this group
            per_macaque = []
            for m in macaques:
                try:
                    r = triple_n.compute_rdm(indices=stim_index, macaque=m, **filt)
                except ValueError:
                    continue                                  # this macaque has too few units here
                if r.std() > 0:
                    per_macaque.append(r)
            if len(per_macaque) >= 2:                          # need >=2 macaques for a noise ceiling
                group_nc[label] = noise_ceiling(np.vstack(per_macaque))
        return group_rdms, group_nc

    @staticmethod
    def _correlate(model_rdm, group_rdms, group_nc, labels, index_name="region"):
        """Spearman-correlate ``model_rdm`` against each available group RDM.

        Iterates ``labels`` in order, skipping any without a computed RDM, and
        attaches that group's noise-ceiling bounds (NaN when unavailable)."""
        rows = []
        for label in labels:
            if label not in group_rdms:
                continue
            rho = spearmanr(model_rdm, group_rdms[label])[0]
            low, high = group_nc.get(label, (np.nan, np.nan))
            rows.append((label, rho, low, high))
        return pd.DataFrame(
            rows, columns=[index_name, "spearman_rho", "noise_ceiling_low", "noise_ceiling_high"]
        ).set_index(index_name)

    @staticmethod
    def plot_corr_table(tables, xlabel="brain region", title="", ax=None):
        """Annotated heatmap of model(s) (rows) x brain regions (cols).

        ``tables``: a single DataFrame from :meth:`rsa_algonauts`/:meth:`rsa_triple_n`,
        or a ``{model_label: DataFrame}`` mapping to stack several models as rows.
        A noise-ceiling row (per-column upper bound, shared across models) is appended
        when any region carries one. Composite ``"primary | secondary"`` column labels
        (e.g. ``"Face | 2"`` from an ``area x unit_type`` segmentation) are drawn as a
        two-level x-axis -- the secondary value per tick, the shared primary bracketed
        beneath. Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        if isinstance(tables, pd.DataFrame):
            tables = {"model": tables}
        model_labels = list(tables)

        # Column order/union from the tables (first occurrence wins).
        cols = list(dict.fromkeys(c for t in tables.values() for c in t.index))

        corr = np.array([[tables[m]["spearman_rho"].get(c, np.nan) for c in cols]
                         for m in model_labels])

        # Per-column noise-ceiling upper bound (group-level, so identical across models).
        ceil = np.array([next((t["noise_ceiling_high"].get(c, np.nan) for t in tables.values()
                               if c in t.index and np.isfinite(t["noise_ceiling_high"].get(c, np.nan))),
                              np.nan) for c in cols])
        show_ceiling = bool(np.isfinite(ceil).any())

        M = np.full((len(model_labels) + int(show_ceiling), len(cols)), np.nan)
        M[:len(model_labels), :] = corr
        row_labels = list(model_labels)
        if show_ceiling:
            M[-1, :] = ceil
            row_labels = row_labels + ["noise ceiling"]

        vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("lightgray")                              # NaN cells shown gray
        # composite "primary | secondary" labels -> a grouped two-level x-axis
        hierarchical = any(" | " in str(c) for c in cols)
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 3,
                                            0.5 * len(row_labels) + 2 + 1.1 * hierarchical))
        else:
            fig = ax.figure
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ModelAnalysisBase._grouped_xaxis(ax, cols, xlabel)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
        if show_ceiling:
            ax.axhline(len(model_labels) - 0.5, color="k", lw=1.5)   # separate the ceiling row
        ax.set_ylabel("model")
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label="Spearman rho")
        fig.tight_layout(rect=[0, 0.06 if hierarchical else 0, 1, 1])   # reserve room for group labels
        return fig

    @staticmethod
    def plot_encoding_table(tables, xlabel="brain region", title="", ax=None,
                            vmax=None):
        """Heatmap of mean held-out encoding scores by model and neural subset."""
        import matplotlib.pyplot as plt

        if isinstance(tables, pd.DataFrame):
            tables = {"model": tables}
        if not tables:
            return None

        model_labels = list(tables)
        cols = list(dict.fromkeys(column for table in tables.values()
                                  for column in table.index))
        if not cols:
            return None

        scores = np.array([
            [tables[model]["mean_encoding_score"].get(column, np.nan)
             for column in cols]
            for model in model_labels
        ])
        if vmax is None:
            vmax = np.nanmax(np.abs(scores)) if np.isfinite(scores).any() else 1.0
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("lightgray")
        hierarchical = any(" | " in str(column) for column in cols)
        if ax is None:
            fig, ax = plt.subplots(figsize=(
                0.62 * len(cols) + 3,
                0.5 * len(model_labels) + 2 + 1.1 * hierarchical,
            ))
        else:
            fig = ax.figure
        image = ax.imshow(scores, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ModelAnalysisBase._grouped_xaxis(ax, cols, xlabel)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        ax.set_ylabel("model")
        ax.set_title(title)
        for row in range(scores.shape[0]):
            for column in range(scores.shape[1]):
                value = scores[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.6 * vmax else "black",
                    )
        fig.colorbar(image, ax=ax, label="Mean held-out Pearson r")
        fig.tight_layout(rect=[0, 0.06 if hierarchical else 0, 1, 1])
        return fig

    @staticmethod
    def _grouped_xaxis(ax, cols, xlabel):
        """Label the heatmap x-axis. Plain labels get a single rotated row; composite
        ``"primary | secondary"`` labels (e.g. ``"Face | 2"``) get a two-level axis --
        the secondary value as the inner tick, the shared primary drawn once as a
        bracketed group label beneath, with a white divider between groups."""
        import matplotlib.transforms as mtransforms

        n = len(cols)
        ax.set_xticks(range(n))
        if not any(" | " in str(c) for c in cols):
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_xlabel(xlabel)
            return

        inner = [str(c).split(" | ")[-1] for c in cols]
        primary = [" | ".join(str(c).split(" | ")[:-1]) for c in cols]
        ax.set_xticklabels(inner, rotation=0, fontsize=8)

        # contiguous runs of the same primary label -> one bracketed group each
        groups, start = [], 0
        for j in range(1, n + 1):
            if j == n or primary[j] != primary[start]:
                groups.append((primary[start], start, j - 1))
                start = j

        fig = ax.figure
        xtrans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        bracket = mtransforms.offset_copy(xtrans, fig=fig, y=-18, units="points")
        for label, s, e in groups:
            ax.plot([s - 0.4, e + 0.4], [0, 0], transform=bracket,
                    color="0.35", lw=1.1, clip_on=False)                # spanning underline
            ax.annotate(label, xy=(0.5 * (s + e), 0), xycoords=xtrans,
                        xytext=(0, -21), textcoords="offset points",
                        ha="center", va="top", fontsize=9, fontweight="bold",
                        annotation_clip=False)
        for _label, s, _e in groups[1:]:
            ax.axvline(s - 0.5, color="white", lw=2)                    # divider between groups
        ax.annotate(xlabel, xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -40), textcoords="offset points",
                    ha="center", va="top", fontsize=10, annotation_clip=False)
