import numpy as np
import pandas as pd
import scipy.io as sio

from pathlib import Path

from psm_final.analysis.correlating import correlation_rdm


class TripleN():
    def __init__(self, triple_n_dir):
        self.triple_n_dir = triple_n_dir
        self._load_responses()

    def _load_responses(self):
        area_xyz = pd.read_excel(f"{self.triple_n_dir}/others/AreaXYZ.xlsx")
        area_xyz["Label"] = area_xyz["Label"].astype(str).str.strip()
        area_excluded = pd.read_excel(f"{self.triple_n_dir}/others/exclude_area.xls")

        # look-ups from the area catalog, keyed by AreaIDX (== exclude_area RoiIndex)
        area_label = area_xyz.set_index("AreaIDX")["Label"]
        area_subject = area_xyz.set_index("AreaIDX")["Subject"]

        PREFERENCE = np.array(["F", "B", "O"])  # order matches the (F_SI, B_SI, O_SI) stack below

        # Build one big response matrix (units x 1072) plus an aligned per-unit metadata table, so
        # you can select any grouping with a boolean mask. exclude_area.xls maps each session to an
        # area (RoiIndex == AreaIDX) and the depth window [y1, y2] of the units in that area.
        resp_blocks = []
        meta_blocks = []
        for row in area_excluded.itertuples():
            files = list(Path(f"{self.triple_n_dir}/Processed").glob(f"*ses{row.SesIdx:02d}*"))
            if not files:
                continue
            session = sio.loadmat(files[0])
            response_best = session["response_best"]            # units x 1072 stimuli
            pos = np.asarray(session["pos"]).ravel()             # unit depth along the probe (microns)
            in_area = (pos >= row.y1) & (pos <= row.y2)          # units belonging to this area

            # z-score each unit across the 1072 stimuli (within session, before pooling); dead units
            # (zero std) map to all-zeros instead of NaN.
            resp = response_best[in_area].astype(float)
            mu = resp.mean(axis=1, keepdims=True)
            sd = resp.std(axis=1, keepdims=True)
            resp = np.divide(resp - mu, sd, out=np.zeros_like(resp), where=sd > 0)

            # each unit's own category tuning = argmax of its face/body/object selectivity indices
            selectivity = np.stack([session["F_SI"].ravel(), session["B_SI"].ravel(), session["O_SI"].ravel()])
            preference = PREFERENCE[np.argmax(selectivity[:, in_area], axis=0)]

            resp_blocks.append(resp)
            meta_blocks.append(pd.DataFrame({
                "session": row.SesIdx,                                  # SesIdx (1..90)
                "area_index": int(row.RoiIndex),                        # AreaXYZ AreaIDX
                "area_label": area_label.get(row.RoiIndex, "Unknown"),  # coarse area label (Face/Body/...)
                "patch": row.AREALABEL,                                 # fine patch name (MB1, MF1, V4, ...)
                "category": row.Categoty,                               # patch's stimulus category: B / F / O
                "region": row.Area,                                     # IT / EVC
                "macaque": f"M{int(area_subject.get(row.RoiIndex))}",   # M1..M5
                "preference": preference,                               # this unit's tuning: F / B / O
                "depth": pos[in_area],                                  # unit depth (microns)
            }))

        self.responses = np.vstack(resp_blocks)                  # (n_units, 1072) z-scored responses
        self.units = pd.concat(meta_blocks, ignore_index=True)   # one row per unit, row-aligned with responses

    @staticmethod
    def nsd_to_stim_index(nsd_ids, crosswalk=None, drop_missing=False):
        """Map NSD 73k ids (1-based) to Triple-N ``stim_index`` values (1-based, 1..1000).

        Inverts the crosswalk's ``nsd_id_73k`` column. Only the 1000 NSD-Shared scenes
        carry an NSD id; localizers (``stim_index`` 1001..1072) and any 73k id outside
        the shared-1000 set have no mapping -> ``None`` (input order preserved), unless
        ``drop_missing=True``. The returned list is exactly the ``indices`` argument
        expected by :meth:`compute_rdm`.

        ``crosswalk``: optional already-loaded crosswalk DataFrame; defaults to
        :func:`psm_final.helpers.stimulus.load_crosswalk`.
        """
        from psm_final.data.stimulus import load_crosswalk

        if crosswalk is None:
            crosswalk = load_crosswalk()
        scenes = crosswalk[crosswalk["kind"] == "scene"]
        mapping = dict(zip(scenes["nsd_id_73k"].astype(int), scenes["stim_index"].astype(int)))
        stim = [mapping.get(int(i)) for i in np.asarray(nsd_ids).reshape(-1)]
        return [s for s in stim if s is not None] if drop_missing else stim

    def compute_rdm(self, macaque=None, area=None, category=None,
                    region=None, preference=None, indices=None, **filters):
        """Stimulus x stimulus correlation-distance RDM over a selected set of units.

        Units are chosen by ANDing the given attribute filters (each a value or list
        of values; None = ignore): macaque, area (-> area_index), category, region,
        preference, plus any other `units` column via **filters (e.g. area_label=,
        patch=, session=). By default the RDM spans the 1000 NSD scenes; pass
        `indices` (1-based stim_index, 1..1072) to select/reorder stimuli. Returns the
        condensed upper triangle, matching Algonauts.compute_rdm.
        """
        # --- select units ---
        criteria = {"macaque": macaque, "area_index": area, "category": category,
                    "region": region, "preference": preference, **filters}
        unit_mask = np.ones(len(self.units), dtype=bool)
        for col, value in criteria.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                unit_mask &= self.units[col].isin(list(value)).to_numpy()
            else:
                unit_mask &= (self.units[col] == value).to_numpy()
        if unit_mask.sum() < 2:
            raise ValueError(f"need >=2 units for an RDM, matched {int(unit_mask.sum())}")

        # --- select stimuli (default: the 1000 NSD scenes; localizers are 1001..1072) ---
        stim_cols = np.arange(1000) if indices is None else np.asarray(indices) - 1

        # --- stimulus x stimulus RDM (transpose: stimuli are items, units are features) ---
        patterns = self.responses[unit_mask][:, stim_cols].T
        return correlation_rdm(patterns)