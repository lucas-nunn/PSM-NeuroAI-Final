import re

import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from scipy.stats import spearmanr

from psm_final.analysis.correlating import correlation_rdm, noise_ceiling


class ModelAnalysisBase():
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

    def rdm(self, indices=None):
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
        embeddings = np.stack(embeddings, axis=0)
        return correlation_rdm(embeddings)

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

    def rsa_triple_n(self, triple_n, algonauts, shared_ids, subjects=range(1, 9), area_labels=None):
        """Spearman-correlate this model's RDM against every Triple-N area label.

        For each area label, one RDM is built from all of its units (pooled across
        macaques) and correlated with the model RDM. Returns a DataFrame indexed by
        area label with columns ``spearman_rho`` and the per-area across-macaque
        ``noise_ceiling_low`` / ``noise_ceiling_high``.
        """
        _, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        if area_labels is None:
            area_labels = sorted(triple_n.units["area_label"].unique())
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        group_rdms, group_nc = self._triple_n_group_rdms(triple_n, stim_index, area_labels)
        return self._correlate(model_rdm, group_rdms, group_nc, area_labels, index_name="area_label")

    def rsa_tables(self, algonauts, triple_n, shared_ids, subjects=range(1, 9),
                   rois=None, area_labels=None):
        """Both RSA tables at once, over one shared stimulus set.

        Convenience wrapper around :meth:`rsa_algonauts` and :meth:`rsa_triple_n`
        that aligns the stimuli and computes the model RDM a single time. Returns
        ``{"algonauts": <DataFrame>, "triple_n": <DataFrame>}``.
        """
        rois = list(algonauts.ALGO_ROIS if rois is None else rois)
        if area_labels is None:
            area_labels = sorted(triple_n.units["area_label"].unique())

        nsd_ids, stim_index = self.aligned_stimuli(algonauts, triple_n, shared_ids, subjects)
        model_rdm = self.rdm(indices=np.asarray(stim_index) - 1)

        algo_rdms, algo_nc = self._algonauts_group_rdms(algonauts, nsd_ids, subjects, rois)
        tn_rdms, tn_nc = self._triple_n_group_rdms(triple_n, stim_index, area_labels)
        return {
            "algonauts": self._correlate(model_rdm, algo_rdms, algo_nc, rois, index_name="roi"),
            "triple_n": self._correlate(model_rdm, tn_rdms, tn_nc, area_labels, index_name="area_label"),
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
    def _triple_n_group_rdms(triple_n, stim_index, area_labels):
        """Per-area-label RDM (units pooled) + across-macaque noise ceiling."""
        macaques = sorted(triple_n.units["macaque"].unique())
        group_rdms, group_nc = {}, {}
        for label in area_labels:
            try:
                group_rdms[label] = triple_n.compute_rdm(area_label=label, indices=stim_index)
            except ValueError:
                continue                                      # fewer than 2 units for this label
            per_macaque = []
            for m in macaques:
                try:
                    r = triple_n.compute_rdm(area_label=label, macaque=m, indices=stim_index)
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
        when any region carries one. Returns the matplotlib Figure.
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 3, 0.5 * len(row_labels) + 2))
        else:
            fig = ax.figure
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
        if show_ceiling:
            ax.axhline(len(model_labels) - 0.5, color="k", lw=1.5)   # separate the ceiling row
        ax.set_xlabel(xlabel); ax.set_ylabel("model")
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label="Spearman rho")
        fig.tight_layout()
        return fig
