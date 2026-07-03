
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ========= WARNING: SLOP FOR REFERENCE PURPOSES =========
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""Triple-N <-> NSD/COCO stimulus crosswalk.

The Triple-N stimulus pool is **1072** images:

* indices ``1..1000``  -> the NSD-Shared1000 natural scenes, stored *by position*
  (``StimuliNNN/0001.bmp..1000.bmp`` / ``img_pool[0:1000]``), **not** by NSD id.
* indices ``1001..1072`` -> the 72 face/body/object localizer images
  (``MFOB001.bmp..MFOB072.bmp``).

A trial's ``img_idx`` (1-based, found in ``Raw/H5FILES/ses*_info.mat``) indexes this
pool directly, so the crosswalk table can be applied per-trial with
:func:`map_trials`.

Because the Triple-N scene order is *not* guaranteed to match NSD ``sharedix`` order,
:func:`build_crosswalk` recovers each scene's NSD 73k id by **pixel-matching** the
Triple-N scenes against the canonical NSD images (``nsd_stimuli.hdf5``) -- no ordering
assumption -- then joins to COCO ids via ``nsd_stim_info_merged.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

N_SCENES = 1000
N_LOCALIZERS = 72
N_STIM = N_SCENES + N_LOCALIZERS  # 1072

# Repo root: .../src/psm_final/helpers/stimulus.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Default data locations (machine-specific; override via arguments as needed).
DEFAULT_TRIPLE_N_ROOT = Path("/media/chuddy/Extreme SSD/data/triple-N")
DEFAULT_NSD_ROOT = Path("/media/chuddy/Extreme SSD/data/NSD")
DEFAULT_NSD_EXPDESIGN = _REPO_ROOT / "nsd_expdesign.mat"
# Cached table lives inside the package so it ships once built (no SSD needed to use it).
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "triple_n_crosswalk.csv"

_M_TOKEN = re.compile(r"_(M\d+)_")


# --------------------------------------------------------------------------- #
# Macaque identity
# --------------------------------------------------------------------------- #
def session_macaque(path: str | Path, info: dict | None = None) -> dict[str, Any]:
    """Return the macaque for a Triple-N session.

    ``path``: any session filename containing the ``_M#_`` token (the paper's
    subject index M1-M5), e.g. ``Processed_ses01_240629_M1_2.mat``.
    ``info``: optional already-loaded ``*_info.mat`` dict; if given, the animal's
    name is read from ``meta_data['exp_subject']`` (e.g. ``"ZhuangZhuang"``).
    """
    match = _M_TOKEN.search(str(path))
    subject_index = match.group(1) if match else None
    name = None
    if isinstance(info, dict):
        meta = info.get("meta_data", info)
        exp_subject = meta.get("exp_subject") if isinstance(meta, dict) else None
        if exp_subject is not None:
            arr = np.asarray(exp_subject).reshape(-1)
            name = str(arr[0]) if arr.size else str(exp_subject)
    return {"subject_index": subject_index, "name": name}


# --------------------------------------------------------------------------- #
# Image signatures + matching
# --------------------------------------------------------------------------- #
def _to_gray(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr.astype(np.float64)


def _resize_area(gray: np.ndarray, size: int) -> np.ndarray:
    """Area-average downsample to ``size x size`` (pure numpy, handles any input size)."""
    height, width = gray.shape
    row_bin = np.minimum(np.arange(height) * size // height, size - 1)
    col_bin = np.minimum(np.arange(width) * size // width, size - 1)
    rows = np.broadcast_to(row_bin[:, None], (height, width))
    cols = np.broadcast_to(col_bin[None, :], (height, width))
    acc = np.zeros((size, size), dtype=np.float64)
    cnt = np.zeros((size, size), dtype=np.float64)
    np.add.at(acc, (rows, cols), gray)
    np.add.at(cnt, (rows, cols), 1.0)
    return acc / np.maximum(cnt, 1.0)


def _signature(img: Any, size: int = 32) -> np.ndarray:
    """Brightness/contrast-invariant fingerprint: z-scored ``size x size`` thumbnail."""
    vec = _resize_area(_to_gray(img), size).ravel()
    vec = vec - vec.mean()
    std = vec.std()
    if std > 0:
        vec = vec / std
    return vec


def _match(triple_sigs: np.ndarray, nsd_sigs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour match Triple-N scene -> NSD shared scene.

    Returns ``(best_j, mutual, dmin)`` where ``best_j[i]`` is the matched NSD index,
    ``mutual[i]`` is True when the match is reciprocal, and ``dmin[i]`` the L2 distance.
    """
    dist2 = (
        (triple_sigs**2).sum(1)[:, None]
        + (nsd_sigs**2).sum(1)[None, :]
        - 2.0 * (triple_sigs @ nsd_sigs.T)
    )
    np.maximum(dist2, 0.0, out=dist2)
    best_j = dist2.argmin(axis=1)
    best_i_for_j = dist2.argmin(axis=0)
    mutual = best_i_for_j[best_j] == np.arange(best_j.size)
    dmin = np.sqrt(dist2[np.arange(best_j.size), best_j])
    return best_j, mutual, dmin


# --------------------------------------------------------------------------- #
# Crosswalk build / load / apply
# --------------------------------------------------------------------------- #
_INT_COLS = ["nsd_id_73k", "nsd_shared_pos", "coco_id"]


def _coerce_dtypes(df):
    import pandas as pd

    for col in _INT_COLS:
        if col in df:
            df[col] = pd.array(pd.to_numeric(df[col], errors="coerce"), dtype="Int64")
    if "match_mutual" in df:
        df["match_mutual"] = df["match_mutual"].astype("boolean")
    return df


def _load_nsd_shared_images(nsd_root: Path, sharedix: np.ndarray) -> np.ndarray:
    """Read the 1000 NSD shared images, aligned to ``sharedix`` order."""
    import h5py

    path = nsd_root / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    order = np.argsort(sharedix)  # h5py fancy-indexing needs increasing indices
    with h5py.File(path, "r") as handle:
        sorted_imgs = handle["imgBrick"][(sharedix[order] - 1).tolist()]
    images = np.empty_like(sorted_imgs)
    images[order] = sorted_imgs  # back to sharedix order: images[p] <-> sharedix[p]
    return images


def build_crosswalk(
    triple_n_root: str | Path = DEFAULT_TRIPLE_N_ROOT,
    nsd_root: str | Path = DEFAULT_NSD_ROOT,
    nsd_expdesign_path: str | Path = DEFAULT_NSD_EXPDESIGN,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    sig_size: int = 32,
    max_match_dist: float | None = None,
):
    """Build the 1072-row Triple-N stimulus crosswalk by pixel-matching to NSD.

    Returns a DataFrame with columns ``stim_index`` (1-based), ``kind``
    (``scene``/``localizer``), ``stim_filename``, ``nsd_id_73k``, ``nsd_shared_pos``,
    ``coco_id``, ``coco_split``, ``match_dist``, ``match_mutual``. If ``cache_path`` is
    not None the table is written there as CSV.
    """
    import pandas as pd

    triple_n_root = Path(triple_n_root)
    nsd_root = Path(nsd_root)

    pool = sio.loadmat(triple_n_root / "others" / "img_pool.mat", simplify_cells=True)["img_pool"]
    pool = np.asarray(pool, dtype=object).reshape(-1)
    if pool.shape[0] != N_STIM:
        raise ValueError(f"expected {N_STIM} images in img_pool, got {pool.shape[0]}")
    scene_imgs = pool[:N_SCENES]

    sharedix = sio.loadmat(nsd_expdesign_path)["sharedix"].reshape(-1).astype(int)  # 1-based, 73k
    if sharedix.size != N_SCENES:
        raise ValueError(f"expected {N_SCENES} sharedix entries, got {sharedix.size}")

    nsd_imgs = _load_nsd_shared_images(nsd_root, sharedix)

    triple_sigs = np.stack([_signature(im, sig_size) for im in scene_imgs])
    nsd_sigs = np.stack([_signature(im, sig_size) for im in nsd_imgs])
    best_j, mutual, dmin = _match(triple_sigs, nsd_sigs)

    nsd_id = sharedix[best_j]            # 1-based NSD 73k id per scene position
    nsd_shared_pos = best_j + 1          # 1-based position within sharedix

    stim = pd.read_csv(nsd_root / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv")
    coco_by_nsd = stim.set_index("nsdId")["cocoId"]      # nsdId is 0-based
    split_by_nsd = stim.set_index("nsdId")["cocoSplit"]
    coco_id = coco_by_nsd.reindex(nsd_id - 1).to_numpy()
    coco_split = split_by_nsd.reindex(nsd_id - 1).to_numpy()

    rows: list[dict[str, Any]] = []
    for i in range(N_SCENES):
        rows.append(
            dict(
                stim_index=i + 1,
                kind="scene",
                stim_filename=f"{i + 1:04d}.bmp",
                nsd_id_73k=int(nsd_id[i]),
                nsd_shared_pos=int(nsd_shared_pos[i]),
                coco_id=int(coco_id[i]),
                coco_split=str(coco_split[i]),
                match_dist=float(dmin[i]),
                match_mutual=bool(mutual[i]),
            )
        )
    for loc in range(1, N_LOCALIZERS + 1):
        rows.append(
            dict(
                stim_index=N_SCENES + loc,
                kind="localizer",
                stim_filename=f"MFOB{loc:03d}.bmp",
                nsd_id_73k=pd.NA,
                nsd_shared_pos=pd.NA,
                coco_id=pd.NA,
                coco_split=pd.NA,
                match_dist=pd.NA,
                match_mutual=pd.NA,
            )
        )
    df = _coerce_dtypes(pd.DataFrame(rows))

    n_bad = int((~mutual).sum())
    same_order = bool(np.all(best_j == np.arange(N_SCENES)))
    print(f"[crosswalk] scenes matched mutually: {N_SCENES - n_bad}/{N_SCENES}; max L2={dmin.max():.4f}")
    print(f"[crosswalk] matched nsd ids == sharedix set: {set(nsd_id.tolist()) == set(sharedix.tolist())}")
    print(f"[crosswalk] Triple-N scene order == sharedix order: {same_order}")
    if max_match_dist is not None and dmin.max() > max_match_dist:
        print(f"[crosswalk] WARNING: max match dist {dmin.max():.4f} exceeds {max_match_dist}")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[crosswalk] saved -> {cache_path}")
    return df


def load_crosswalk(cache_path: str | Path = DEFAULT_CACHE_PATH, **build_kwargs):
    """Load the cached crosswalk CSV, building it (from the SSD data) if missing."""
    import pandas as pd

    cache_path = Path(cache_path)
    if cache_path.exists():
        return _coerce_dtypes(pd.read_csv(cache_path))
    return build_crosswalk(cache_path=cache_path, **build_kwargs)


def map_trials(img_idx, crosswalk=None):
    """Map a session's per-trial ``img_idx`` (1-based, 1..1072) to crosswalk rows.

    Returns a per-trial DataFrame (``trial``, ``img_idx``, then the crosswalk columns:
    ``kind``, ``nsd_id_73k``, ``coco_id``, ...). Localizer trials carry NA NSD/COCO ids.
    """
    if crosswalk is None:
        crosswalk = load_crosswalk()
    crosswalk = crosswalk.sort_values("stim_index").reset_index(drop=True)
    img_idx = np.asarray(img_idx).reshape(-1).astype(int)
    out = crosswalk.iloc[img_idx - 1].reset_index(drop=True)
    out.insert(0, "img_idx", img_idx)
    out.insert(0, "trial", np.arange(img_idx.size))
    return out
