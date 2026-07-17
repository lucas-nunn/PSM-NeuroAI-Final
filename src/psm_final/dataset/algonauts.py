import os
from pathlib import Path

import numpy as np

from psm_final.analysis.correlating import correlation_rdm


class Algonauts():

    ALGO_ROIS = [
        "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",        # prf-visualrois
        "EBA", "FBA-1", "FBA-2", "mTL-bodies",                  # floc-bodies
        "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces",      # floc-faces
        "OPA", "PPA", "RSC",                                    # floc-places
        "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",   # floc-words
        "early", "midventral", "midlateral", "midparietal",     # streams
        "ventral", "lateral", "parietal",
    ]

    @classmethod
    def available_rois(cls):
        """Return a copy of the supported Algonauts ROI labels."""
        return list(cls.ALGO_ROIS)

    def __init__(self, algonauts_dir, nsd_indices, noise_ceiling_dir=None):
        self.algonauts_dir = algonauts_dir
        self.nsd_indices = nsd_indices
        self.noise_ceiling_dir = self._resolve_noise_ceiling_dir(
            noise_ceiling_dir
        )
        self._noise_ceiling_cache = {}
        # Per-subject/ROI read caches. The fMRI lives on an external SSD and the same
        # ~1000 shared stimuli are read for every model and every ROI, so without
        # these each model reloaded ~12.5 GB of fMRI (the full 9,841-stimulus arrays,
        # for a ~1000-stimulus slice) -- turning a CPU-bound sweep into a disk-bound
        # crawl once the OS page cache is under pressure. Loaded once, reused forever.
        self._train_row_cache = {}     # subject -> {nsd_id: train_row}
        self._response_cache = {}      # (subject, rows_key) -> (lh_shared, rh_shared)
        self._roi_mask_cache = {}      # (roi, subject) -> (lh_mask, rh_mask)

    def _resolve_noise_ceiling_dir(self, explicit):
        """Find the Algonauts root that contains official per-vertex ceilings.

        The public challenge releases are commonly unpacked as sibling ``train``
        and ``test`` roots. Accept an explicit root, but auto-detect that standard
        layout so existing ``ALGONAUTS_DIR=.../train`` configurations keep working.
        """
        training_root = Path(self.algonauts_dir).expanduser()
        candidates = (
            [Path(explicit).expanduser()]
            if explicit is not None
            else [training_root, training_root.parent / "test"]
        )
        for candidate in candidates:
            probe = (
                candidate
                / "subj01"
                / "test_split"
                / "noise_ceiling"
                / "lh_noise_ceiling.npy"
            )
            if probe.is_file():
                return str(candidate.resolve())
        return None

    def noise_ceiling(self, subject, roi=None):
        """Official per-vertex ceiling in squared-correlation (R²) units.

        Returns ``(lh, rh)`` in the same challenge-space order as
        :meth:`response_matrix`. When ``roi`` is given, the same ROI masks used for
        the response matrices are applied, which makes the returned vectors align
        exactly with the encoding targets.
        """
        if self.noise_ceiling_dir is None:
            raise FileNotFoundError(
                "Algonauts noise ceilings were not found; pass noise_ceiling_dir "
                "or place the official test release beside the training root"
            )
        subject = int(subject)
        if subject not in self._noise_ceiling_cache:
            base = (
                Path(self.noise_ceiling_dir)
                / f"subj0{subject}"
                / "test_split"
                / "noise_ceiling"
            )
            self._noise_ceiling_cache[subject] = (
                np.load(base / "lh_noise_ceiling.npy"),
                np.load(base / "rh_noise_ceiling.npy"),
            )
        left, right = self._noise_ceiling_cache[subject]
        if roi is None:
            return left, right
        left_mask, right_mask = self.roi_mask(roi, subject)
        return left[left_mask], right[right_mask]

    def nsd_to_train_row(self, subject):
        """Map NSD 73k id (1-based, as in sharedix / the Triple-N crosswalk) to this
        subject's 0-based row in ``*_training_fmri.npy``.

        Algonauts filenames look like ``train-0010_nsd-00110.png`` where the NSD id is
        **0-based** (``nsd-00110`` -> 1-based id 111) and the training row is **1-based**
        (``train-0010`` -> row 9). Only images in the subject's training split are present
        (~982 of the 1000 shared images for subj01), so not every NSD id will be a key.

        Cached per subject: the mapping is built from an ``os.listdir`` of ~9,841
        training-image filenames, and is otherwise rebuilt on every response load.
        """
        subject = int(subject)
        if subject in self._train_row_cache:
            return self._train_row_cache[subject]
        img_dir = f"{self.algonauts_dir}/subj0{subject}/training_split/training_images"
        mapping = {}
        for filename in os.listdir(img_dir):
            train_str, nsd_str = filename.split("_")[:2]
            train_row = int(train_str.split("-")[1]) - 1                # 1-based label -> 0-based row
            nsd_id = int(nsd_str.split("-")[1].split(".")[0]) + 1       # 0-based filename -> 1-based id
            mapping[nsd_id] = train_row
        self._train_row_cache[subject] = mapping
        return mapping

    def shared_stimuli_indices(self, subject, indices):
        """For a list of NSD 73k ids (1-based), return the matched ``(nsd_ids, train_rows)``
        present in this subject's training split, preserving the order of ``nsd_indices``.
        ``train_rows`` are 0-based indices into the ``*_training_fmri.npy`` arrays."""
        mapping = self.nsd_to_train_row(subject)
        matched_nsd, train_rows = [], []
        for nsd in indices:
            row = mapping.get(int(nsd))
            if row is not None:
                matched_nsd.append(int(nsd))
                train_rows.append(row)
        return matched_nsd, train_rows

    def roi_mask(self, roi, subject):
        cache_key = (roi, int(subject))
        if cache_key in self._roi_mask_cache:            # small .npy loads, but called
            return self._roi_mask_cache[cache_key]       # once per (roi, subject) per model
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'

        # Load the ROI maps in *challenge space*: these align 1-to-1 with the columns of
        # ``*_training_fmri.npy`` so they can index the fMRI arrays directly. (The
        # ``*_fsaverage_space.npy`` maps are the full ~164k-vertex surface, for plotting.)
        # Challenge space is subject-specific (each subject has a different vertex count),
        # so the masks live under that subject's ``roi_masks`` directory.
        roi_masks_dir = os.path.join(self.algonauts_dir, f'subj0{subject}', 'roi_masks')
        roi_class_dir_lh = os.path.join(roi_masks_dir, 'lh.'+roi_class+'_challenge_space.npy')
        roi_class_dir_rh = os.path.join(roi_masks_dir, 'rh.'+roi_class+'_challenge_space.npy')
        roi_map_dir = os.path.join(roi_masks_dir, 'mapping_'+roi_class+'.npy')
        challenge_roi_class_lh = np.load(roi_class_dir_lh)
        challenge_roi_class_rh = np.load(roi_class_dir_rh)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        # Boolean mask of the vertices belonging to the ROI of interest, per hemisphere.
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        lh_roi = challenge_roi_class_lh == roi_mapping
        rh_roi = challenge_roi_class_rh == roi_mapping

        self._roi_mask_cache[cache_key] = (lh_roi, rh_roi)
        return lh_roi, rh_roi

    def compute_rdm(self, subject, indices=None, roi=None):
        lh_shared, rh_shared = self.response_matrix(subject, indices=indices, roi=roi)
        shared = np.concat((lh_shared, rh_shared), axis=1)
        return correlation_rdm(shared, condensed=True)
 
    def response_matrix(self, subject, indices=None, roi=None):
        """Raw per-vertex fMRI responses for a stimulus set (no RDM computed).

        Returns (lh_responses, rh_responses), each shaped (n_stimuli, n_vertices).

        The full-vertex shared-stimulus slice is loaded once per (subject, stimulus
        set) and cached. The fMRI arrays are memory-mapped, so fancy-indexing the
        ~1,000 shared rows reads only ~158 MB from disk instead of the full
        ~1.56 GB/subject file; ROI masking is applied to the cached slice, so all
        ROIs (and all models) reuse that single read. Callers must treat the returned
        arrays as read-only (nothing here or in the encoding/RSA paths mutates them).
        """
        if indices is None:
            indices = self.nsd_indices

        _, train_rows = self.shared_stimuli_indices(subject, indices)
        train_rows = np.asarray(train_rows, dtype=np.int64)
        cache_key = (int(subject), train_rows.tobytes())

        cached = self._response_cache.get(cache_key)
        if cached is None:
            base = f"{self.algonauts_dir}/subj0{subject}/training_split/training_fmri"
            # mmap_mode='r' + fancy row indexing materialises only the shared rows;
            # ascontiguousarray realises them into RAM and drops the mmap handle.
            lh = np.load(f"{base}/lh_training_fmri.npy", mmap_mode="r")
            rh = np.load(f"{base}/rh_training_fmri.npy", mmap_mode="r")
            cached = (
                np.ascontiguousarray(lh[train_rows]),
                np.ascontiguousarray(rh[train_rows]),
            )
            self._response_cache[cache_key] = cached

        lh_shared, rh_shared = cached
        if roi is not None:
            lh_roi, rh_roi = self.roi_mask(roi, subject)
            lh_shared = lh_shared[:, lh_roi]      # boolean index -> fresh copy
            rh_shared = rh_shared[:, rh_roi]

        return lh_shared, rh_shared
