import os
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

    def __init__(self, algonauts_dir, nsd_indices):
        self.algonauts_dir = algonauts_dir
        self.nsd_indices = nsd_indices

    def nsd_to_train_row(self, subject):
        """Map NSD 73k id (1-based, as in sharedix / the Triple-N crosswalk) to this
        subject's 0-based row in ``*_training_fmri.npy``.

        Algonauts filenames look like ``train-0010_nsd-00110.png`` where the NSD id is
        **0-based** (``nsd-00110`` -> 1-based id 111) and the training row is **1-based**
        (``train-0010`` -> row 9). Only images in the subject's training split are present
        (~982 of the 1000 shared images for subj01), so not every NSD id will be a key.
        """
        img_dir = f"{self.algonauts_dir}/subj0{subject}/training_split/training_images"
        mapping = {}
        for filename in os.listdir(img_dir):
            train_str, nsd_str = filename.split("_")[:2]
            train_row = int(train_str.split("-")[1]) - 1                # 1-based label -> 0-based row
            nsd_id = int(nsd_str.split("-")[1].split(".")[0]) + 1       # 0-based filename -> 1-based id
            mapping[nsd_id] = train_row
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

        return lh_roi, rh_roi

    def compute_rdm(self, subject, indices=None, roi=None):
        if indices is None:
            indices = self.nsd_indices

        _, train_rows = self.shared_stimuli_indices(subject, indices)

        lh = np.load(f"{self.algonauts_dir}/subj0{subject}/training_split/training_fmri/lh_training_fmri.npy")
        rh = np.load(f"{self.algonauts_dir}/subj0{subject}/training_split/training_fmri/rh_training_fmri.npy")

        lh_shared = lh[train_rows]
        rh_shared = rh[train_rows]

        # Restrict to an ROI's vertices (challenge-space masks index the fMRI columns).
        if roi is not None:
            lh_roi, rh_roi = self.roi_mask(roi, subject)
            lh_shared = lh_shared[:, lh_roi]
            rh_shared = rh_shared[:, rh_roi]

        shared = np.concat((lh_shared, rh_shared), axis=1)

        dist = correlation_rdm(shared, condensed=True)

        return dist
 