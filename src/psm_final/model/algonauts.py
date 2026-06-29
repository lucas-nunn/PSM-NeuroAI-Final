import os
import numpy as np

from scipy.spatial.distance import pdist

from psm_final.model.base import Model


class Algonauts(Model):

    def __init__(self, algonauts_dir, nsd_indices):
        super().__init__(model_name="Algonauts")
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

    def roi_mask(self, roi):
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

        # Load the ROI brain surface maps
        roi_class_dir_lh = os.path.join(self.algonauts_dir, 'roi_masks',
            'lh.'+roi_class+'_fsaverage_space.npy')
        roi_class_dir_rh = os.path.join(self.algonauts_dir, 'roi_masks',
            'rh.'+roi_class+'_fsaverage_space.npy')
        roi_map_dir = os.path.join(self.algonauts_dir, 'roi_masks',
            'mapping_'+roi_class+'.npy')
        fsaverage_roi_class_lh = np.load(roi_class_dir_lh)
        fsaverage_roi_class_rh = np.load(roi_class_dir_rh)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        fsaverage_roi_lh = np.asarray(fsaverage_roi_class_lh == roi_mapping, dtype=int)
        fsaverage_roi_rh = np.asarray(fsaverage_roi_class_rh == roi_mapping, dtype=int)

        return fsaverage_roi_lh, fsaverage_roi_rh

    def compute_rdm(self, subject, indices=None, roi_mask=None):
        if indices is None:
            indices = self.nsd_indices        

        _, train_rows = self.shared_stimuli_indices(subject, indices)

        lh = np.load(f"{self.algonauts_dir}/subj0{subject}/training_split/training_fmri/lh_training_fmri.npy")
        rh = np.load(f"{self.algonauts_dir}/subj0{subject}/training_split/training_fmri/rh_training_fmri.npy")

        lh_shared = lh[train_rows]
        rh_shared = rh[train_rows]
        shared = np.concat((lh_shared, rh_shared), axis=1)

        dist = pdist(shared, 'correlation')

        return dist
 