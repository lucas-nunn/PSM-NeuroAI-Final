"""Model-agnostic RSA and neural-encoding runner.

Discovers every implemented :class:`~psm_final.analysis.model.ModelAnalysisBase`
subclass and runs either representational-similarity analysis (RSA), nested-CV
encoding, or both over one aligned stimulus set. Encoding is explicit because it is
substantially slower; it checkpoints after each model and can safely resume only an
exactly matching configuration.

For RSA, brain RDMs are model-independent and computed once for reuse. For encoding,
each model extracts features once and trains separately for each ROI/area.

Every model plugs into the same base class, so adding a model to this run means
nothing more than writing its ``ModelAnalysisBase`` subclass (with an
``embedding`` and a ``discover``) -- see CONTRIBUTING.md section 4.

Run it::

    psm-rsa                                      # RSA (backward-compatible default)
    psm-rsa --method encoding --regression ridge --resume
    psm-rsa --method both --models '*VAE*'
"""
from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import pkgutil
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import psm_final.analysis as _analysis_pkg
from psm_final.analysis.encoding import DEFAULT_ALPHAS, RESULT_COLUMNS
from psm_final.analysis.model import ModelAnalysisBase
from psm_final.dataset.algonauts import Algonauts
from psm_final.dataset.triple_n import TripleN
from psm_final.dataset.util import shared_stimuli

# .../src/psm_final/analysis/runner.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NSD_MAT = _REPO_ROOT / "nsd_expdesign.mat"
DEFAULT_CHECKPOINTS_ROOT = _REPO_ROOT          # analyzers glob e.g. beta_vae/*/vae.pth under here
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "rsa"
DEFAULT_ENCODING_OUTPUT_DIR = _REPO_ROOT / "results" / "encoding"
DEFAULT_BOTH_OUTPUT_DIR = _REPO_ROOT / "results" / "analysis"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _load_dotenv(repo_root=_REPO_ROOT):
    """Populate os.environ from the repo-root ``.env`` (simple ``KEY=VALUE`` lines)
    without clobbering values already set in the real environment. Lets the runner
    "just work" from ``.env`` the way the notebooks expect, with no python-dotenv
    dependency."""
    env_path = Path(repo_root) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def discover_analyzers():
    """Import every module in ``psm_final.analysis`` and return the concrete
    :class:`ModelAnalysisBase` subclasses -- those that override ``embedding``.

    Importing the modules is what registers the subclasses (Python only knows a
    subclass exists once its ``class`` statement has run). A stub or broken analyzer
    module (e.g. one with unguarded top-level code) is skipped with a warning so it
    can't sink discovery of the healthy analyzers.
    """
    for mod in pkgutil.iter_modules(_analysis_pkg.__path__):
        full_name = f"{_analysis_pkg.__name__}.{mod.name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:                        # noqa: BLE001 -- a bad module must not abort discovery
            warnings.warn(f"skipping analyzer module {full_name!r}: {exc!r}")

    concrete, seen, stack = [], set(), list(ModelAnalysisBase.__subclasses__())
    while stack:                                        # walk nested subclasses too
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls in seen:
            continue
        seen.add(cls)
        if cls.embedding is ModelAnalysisBase.embedding:
            continue                                    # abstract: no embedding() -> can't make an RDM
        concrete.append(cls)
    concrete.sort(key=lambda c: c.__name__)
    return concrete


def collect_models(analyzers, *, triple_n_path, checkpoints_root, device=None):
    """Flatten every analyzer's :meth:`~ModelAnalysisBase.discover` into a single
    ``[(label, factory), ...]`` list. Labels colliding across classes get a numeric
    suffix so two models never overwrite each other in the output tables."""
    specs, used = [], {}
    for cls in analyzers:
        try:
            found = cls.discover(triple_n_path=triple_n_path,
                                 checkpoints_root=checkpoints_root, device=device)
        except Exception as exc:                        # noqa: BLE001 -- one analyzer's failure shouldn't stop the rest
            warnings.warn(f"{cls.__name__}.discover failed: {exc!r}")
            continue
        if not found:
            warnings.warn(f"{cls.__name__}: no runnable models found "
                          f"(no checkpoints under {checkpoints_root}?)")
        for label, factory in found:
            n = used.get(label, 0)
            used[label] = n + 1
            specs.append((f"{label} #{n + 1}" if n else label, factory))
    return specs


# --------------------------------------------------------------------------- #
# Core RSA
# --------------------------------------------------------------------------- #
def run_rsa(algonauts, triple_n, shared_ids, model_specs, *, subjects=range(1, 9),
            rois=None, area_labels=None, segmentations=None):
    """Correlate every model's RDM against every Algonauts ROI and each Triple-N
    segmentation (coarse area, area x firing-rate cluster, region).

    The brain group RDMs + noise ceilings are model-independent, so they are built
    once per segmentation (reusing the model-agnostic building blocks on
    :class:`ModelAnalysisBase`) and every model RDM is correlated against the same
    cached set.

    Returns ``{"algonauts": {label: DataFrame}, "triple_n": {segmentation:
    {label: DataFrame}}, "n_stimuli": int, "subjects": [...]}`` -- each DataFrame is
    exactly what :meth:`ModelAnalysisBase.rsa_algonauts` / ``rsa_triple_n`` produces.
    """
    subjects = list(subjects)
    rois = list(Algonauts.ALGO_ROIS if rois is None else rois)
    segmentations = (ModelAnalysisBase.TRIPLE_N_SEGMENTATIONS
                     if segmentations is None else segmentations)

    # --- one shared stimulus set (present in every subject + mapped to Triple-N) ---
    nsd_ids, stim_index = ModelAnalysisBase.aligned_stimuli(
        algonauts, triple_n, shared_ids, subjects)
    model_indices = np.asarray(stim_index) - 1          # 0-based positions into StimuliNNN
    print(f"[rsa] {len(stim_index)} shared stimuli | {len(rois)} ROIs | "
          f"segmentations={[' | '.join(g) for g in segmentations]} | subjects={subjects}")

    # --- brain RDMs: computed ONCE per segmentation, reused for every model ---
    algo_rdms, algo_nc = ModelAnalysisBase._algonauts_group_rdms(
        algonauts, nsd_ids, subjects, rois)
    tn_groups = {}                                       # segmentation name -> (rdms, ceilings)
    for groupby in segmentations:
        name = " | ".join(groupby)
        # --areas restricts the coarse-area segmentation only
        groups = area_labels if list(groupby) == ["area_label"] else None
        tn_groups[name] = ModelAnalysisBase._triple_n_group_rdms(
            triple_n, stim_index, groupby, groups)
    print(f"[rsa] brain RDMs ready: {len(algo_rdms)}/{len(rois)} ROIs; "
          + "; ".join(f"{len(rdms)} groups [{name}]"
                      for name, (rdms, _nc) in tn_groups.items()))

    algo_tables = {}
    tn_tables = {name: {} for name in tn_groups}        # segmentation -> {model: DataFrame}
    for label, factory in model_specs:
        print(f"[rsa] model: {label}")
        model = factory()                               # loads exactly one checkpoint at a time
        model_rdm = model.rdm(indices=model_indices)
        algo_tables[label] = ModelAnalysisBase._correlate(
            model_rdm, algo_rdms, algo_nc, rois, index_name="roi")
        for name, (group_rdms, group_nc) in tn_groups.items():
            tn_tables[name][label] = ModelAnalysisBase._correlate(
                model_rdm, group_rdms, group_nc, list(group_rdms), index_name=name)
        del model                                       # free the model before loading the next

    return {"algonauts": algo_tables, "triple_n": tn_tables,
            "n_stimuli": len(stim_index), "subjects": subjects}


# --------------------------------------------------------------------------- #
# Core encoding analysis
# --------------------------------------------------------------------------- #
def filter_model_specs(model_specs, patterns=None):
    """Return model specs matching any case-insensitive name pattern.

    A pattern may be an exact name, a shell-style glob, or a plain substring.
    Discovery order is preserved and a model matching several patterns is still
    returned only once.
    """
    if not patterns:
        return list(model_specs)
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = [str(pattern).casefold() for pattern in patterns]

    selected = []
    for spec in model_specs:
        label = spec[0]
        folded = label.casefold()
        if any(
            folded == pattern
            or fnmatch.fnmatchcase(folded, pattern)
            or pattern in folded
            for pattern in patterns
        ):
            selected.append(spec)
    return selected


def _model_output_ids(model_specs):
    """Stable IDs that remain collision-safe across separately filtered runs."""
    return {
        label: f"{_slug(label)}_{hashlib.sha1(label.encode('utf-8')).hexdigest()[:8]}"
        for label, _factory in model_specs
    }


def _normalise_alphas(alphas):
    values = np.asarray(DEFAULT_ALPHAS if alphas is None else alphas, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("alphas must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("alphas must contain only finite positive values")
    return [float(value) for value in np.unique(values)]


def _deduplicate(values, name):
    unique = list(dict.fromkeys(values))
    if len(unique) != len(values):
        warnings.warn(f"duplicate {name} were removed before encoding")
    return unique


def _implementation_fingerprint():
    """Hash analysis source + the packaged stimulus crosswalk.

    Resume is intentionally conservative: changing any implementation that could
    affect features, alignment, preprocessing, or scoring invalidates old results.
    """
    package_root = Path(__file__).resolve().parents[1]
    files = [*package_root.rglob("*.py")]
    crosswalk = package_root / "dataset" / "triple_n_crosswalk.csv"
    if crosswalk.exists():
        files.append(crosswalk)
    digest = hashlib.sha256()
    digest.update(f"python={os.sys.version}\n".encode("utf-8"))
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "Pillow",
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        digest.update(f"{distribution}={version}\n".encode("utf-8"))
    for path in sorted(files):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _update_file_metadata(digest, path, root):
    """Add a cheap content-change proxy (path, size, mtime) to ``digest``."""
    path = Path(path)
    if not path.is_file():
        return
    stat = path.stat()
    try:
        name = path.relative_to(root)
    except ValueError:
        name = path.resolve()
    digest.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))


def _data_fingerprint(algonauts, triple_n, subjects):
    """Fingerprint the selected dataset roots and relevant source-file metadata."""
    digest = hashlib.sha256()
    digest.update(
        f"algonauts_class={type(algonauts).__module__}.{type(algonauts).__qualname__}\n"
        .encode("utf-8")
    )
    algo_value = getattr(algonauts, "algonauts_dir", None)
    if algo_value is not None:
        algo_root = Path(algo_value).expanduser().resolve()
        digest.update(f"algonauts_root={algo_root}\n".encode("utf-8"))
        for subject in subjects:
            subject_root = algo_root / f"subj0{subject}"
            patterns = (
                "training_split/training_fmri/*.npy",
                "roi_masks/*.npy",
                "training_split/training_images/*",
            )
            for pattern in patterns:
                for path in sorted(subject_root.glob(pattern)):
                    _update_file_metadata(digest, path, algo_root)

    digest.update(
        f"triple_n_class={type(triple_n).__module__}.{type(triple_n).__qualname__}\n"
        .encode("utf-8")
    )
    triple_value = getattr(triple_n, "triple_n_dir", None)
    if triple_value is not None:
        triple_root = Path(triple_value).expanduser().resolve()
        digest.update(f"triple_n_root={triple_root}\n".encode("utf-8"))
        patterns = (
            "Processed/*",
            "others/AreaXYZ.xlsx",
            "others/exclude_area.xls",
            "others/StimuliNNN/*.bmp",
        )
        for pattern in patterns:
            for path in sorted(triple_root.glob(pattern)):
                _update_file_metadata(digest, path, triple_root)
    return digest.hexdigest()


def _factory_fingerprint(factory):
    """Fingerprint a deferred analyzer factory and captured model artifact metadata."""
    payload = {
        "factory_type": f"{type(factory).__module__}.{type(factory).__qualname__}",
    }
    code = getattr(factory, "__code__", None)
    if code is not None:
        payload["module"] = getattr(factory, "__module__", None)
        payload["qualname"] = getattr(factory, "__qualname__", None)
        payload["code"] = code.co_code.hex()
        payload["line"] = code.co_firstlineno

    captured = []
    captured.extend(getattr(factory, "__defaults__", None) or ())
    captured.extend((getattr(factory, "__kwdefaults__", None) or {}).values())
    for cell in getattr(factory, "__closure__", None) or ():
        try:
            captured.append(cell.cell_contents)
        except ValueError:
            captured.append("<empty closure>")

    def describe(value):
        if isinstance(value, Path):
            candidate = value.expanduser()
        elif isinstance(value, str):
            candidate = Path(value).expanduser()
        else:
            candidate = None
        if candidate is not None and candidate.exists():
            candidate = candidate.resolve()
            if candidate.is_file():
                stat = candidate.stat()
                return {
                    "path": str(candidate),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _file_sha256(candidate),
                }
            return {"directory": str(candidate)}
        if isinstance(value, type):
            return {"type": f"{value.__module__}.{value.__qualname__}"}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    payload["captured"] = [describe(value) for value in captured]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _encoding_config(
    *,
    regression,
    alphas,
    outer_folds,
    inner_folds,
    seed,
    subjects,
    rois,
    area_labels,
    nsd_ids,
    stim_index,
    implementation_hash,
    data_hash,
    dataset_roots,
):
    """Canonical configuration used for both provenance and resume checks."""
    regression = str(regression).lower()
    if regression not in {"ridge", "lasso"}:
        raise ValueError("regression must be 'ridge' or 'lasso'")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")
    if len(nsd_ids) != len(stim_index):
        raise ValueError("aligned NSD and Triple-N stimulus lists must have equal length")
    stimulus_pairs = [
        [int(nsd_id), int(triple_n_id)]
        for nsd_id, triple_n_id in zip(nsd_ids, stim_index)
    ]
    n_stimuli = len(stimulus_pairs)
    if n_stimuli < 2 * outer_folds:
        raise ValueError(
            f"need at least {2 * outer_folds} aligned stimuli so every outer CV "
            f"test fold can compute Pearson r; got {n_stimuli}"
        )
    smallest_outer_train = n_stimuli - int(np.ceil(n_stimuli / outer_folds))
    if smallest_outer_train < 2 * inner_folds:
        raise ValueError(
            "the smallest outer training split must contain at least "
            f"{2 * inner_folds} stimuli so every inner CV validation fold can "
            f"compute Pearson r; got {smallest_outer_train}"
        )
    stimulus_hash = hashlib.sha256(
        json.dumps(stimulus_pairs, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "regression": regression,
        "alphas": _normalise_alphas(alphas),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "seed": int(seed),
        "subjects": [int(subject) for subject in subjects],
        "rois": [str(roi) for roi in rois],
        "area_labels": [str(label) for label in area_labels],
        "n_stimuli": n_stimuli,
        "stimulus_hash": stimulus_hash,
        "implementation_hash": implementation_hash,
        "data_hash": data_hash,
        "dataset_roots": dataset_roots,
    }


def _encoding_paths(output_dir, output_id, regression):
    output_dir = Path(output_dir)
    prefix = f"{output_id}_encoding_{regression}"
    return {
        "algonauts": output_dir / f"{prefix}_algonauts.csv",
        "triple_n": output_dir / f"{prefix}_triple_n_area_label.csv",
        "done": output_dir / f"{prefix}.done.json",
        "error": output_dir / f"{prefix}.error.json",
        "skipped": output_dir / f"{prefix}.skipped.json",
    }


def _atomic_csv(frame, path, *, index=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_encoding_checkpoint(output_dir, output_id, label, config,
                              model_fingerprint):
    """Load a complete, exact-config checkpoint; return ``None`` if stale/broken."""
    paths = _encoding_paths(output_dir, output_id, config["regression"])
    try:
        manifest = json.loads(paths["done"].read_text())
        if (
            manifest.get("schema_version") != 2
            or manifest.get("label") != label
            or manifest.get("config") != config
            or manifest.get("model_fingerprint") != model_fingerprint
        ):
            return None
        files = manifest.get("files", {})
        for name in ("algonauts", "triple_n"):
            expected = files.get(name, {})
            if (
                expected.get("name") != paths[name].name
                or expected.get("sha256") != _file_sha256(paths[name])
            ):
                return None
        tables = {
            name: pd.read_csv(paths[name], index_col=0)
            for name in ("algonauts", "triple_n")
        }
        if any(not set(RESULT_COLUMNS).issubset(table.columns)
               for table in tables.values()):
            return None
        return tables
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return None


def _write_encoding_checkpoint(output_dir, output_id, label, config,
                               model_fingerprint, tables, elapsed_seconds=None):
    """Atomically write both tables, then the completion marker last."""
    paths = _encoding_paths(output_dir, output_id, config["regression"])
    _atomic_csv(tables["algonauts"], paths["algonauts"])
    _atomic_csv(tables["triple_n"], paths["triple_n"])
    manifest = {
        "schema_version": 2,
        "label": label,
        "config": config,
        "model_fingerprint": model_fingerprint,
        "files": {
            "algonauts": {
                "name": paths["algonauts"].name,
                "sha256": _file_sha256(paths["algonauts"]),
            },
            "triple_n": {
                "name": paths["triple_n"].name,
                "sha256": _file_sha256(paths["triple_n"]),
            },
        },
    }
    if elapsed_seconds is not None:
        manifest["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    _atomic_json(manifest, paths["done"])
    paths["error"].unlink(missing_ok=True)
    paths["skipped"].unlink(missing_ok=True)


def _write_encoding_error(output_dir, output_id, label, config,
                          model_fingerprint, exc):
    paths = _encoding_paths(output_dir, output_id, config["regression"])
    _atomic_json(
        {
            "schema_version": 2,
            "label": label,
            "config": config,
            "model_fingerprint": model_fingerprint,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
        paths["error"],
    )


def _write_encoding_skip(output_dir, output_id, label, config,
                         model_fingerprint, reason):
    paths = _encoding_paths(output_dir, output_id, config["regression"])
    _atomic_json(
        {
            "schema_version": 2,
            "label": label,
            "config": config,
            "model_fingerprint": model_fingerprint,
            "reason": reason,
        },
        paths["skipped"],
    )


def run_encoding(
    algonauts,
    triple_n,
    shared_ids,
    model_specs,
    *,
    subjects=range(1, 9),
    rois=None,
    area_labels=None,
    regression="ridge",
    alphas=None,
    outer_folds=5,
    inner_folds=3,
    seed=42,
    output_dir=None,
    resume=False,
    fail_fast=False,
):
    """Run nested-CV encoding once per ROI/area, checkpointing each model.

    Resume is deliberately conservative: a model is reused only when a completion
    manifest, both result tables, and every data/CV selection setting match.
    """
    subjects = _deduplicate(list(subjects), "subjects")
    rois = _deduplicate(
        list(Algonauts.ALGO_ROIS if rois is None else rois), "ROIs"
    )
    if area_labels is None:
        area_labels = sorted(triple_n.units["area_label"].unique())
    area_labels = _deduplicate(list(area_labels), "area labels")
    nsd_ids, stim_index = ModelAnalysisBase.aligned_stimuli(
        algonauts, triple_n, shared_ids, subjects
    )
    implementation_hash = _implementation_fingerprint()
    data_hash = _data_fingerprint(algonauts, triple_n, subjects)
    dataset_roots = {}
    for name, dataset, attribute in (
        ("algonauts", algonauts, "algonauts_dir"),
        ("triple_n", triple_n, "triple_n_dir"),
    ):
        value = getattr(dataset, attribute, None)
        dataset_roots[name] = (
            str(Path(value).expanduser().resolve()) if value is not None else None
        )
    config = _encoding_config(
        regression=regression,
        alphas=alphas,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        subjects=subjects,
        rois=rois,
        area_labels=area_labels,
        nsd_ids=nsd_ids,
        stim_index=stim_index,
        implementation_hash=implementation_hash,
        data_hash=data_hash,
        dataset_roots=dataset_roots,
    )
    output_ids = _model_output_ids(model_specs)
    model_fingerprints = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[encoding] {config['n_stimuli']} shared stimuli | {len(rois)} ROIs | "
        f"{len(area_labels)} areas | {config['regression']} | "
        f"{config['outer_folds']}x{config['inner_folds']} nested CV | "
        f"{len(config['alphas'])} alphas"
    )
    results = {
        "algonauts": {},
        "triple_n": {},
        "n_stimuli": config["n_stimuli"],
        "subjects": subjects,
        "config": config,
        "errors": {},
        "skipped": {},
        "output_ids": output_ids,
        "model_fingerprints": model_fingerprints,
    }
    if output_dir is not None:
        # A run-level summary is valid only after this invocation reaches the
        # finalizer. Start with an empty combined table so stale rows from an older
        # invocation cannot look current while the first expensive model is running.
        (output_dir / f"encoding_{config['regression']}_run.json").unlink(
            missing_ok=True
        )
        for stem in ("algonauts", "triple_n_area_label"):
            (output_dir / f"encoding_{config['regression']}_{stem}.png").unlink(
                missing_ok=True
            )
        _write_encoding_combined(results, output_dir)
        if not resume:
            # ``resume=False`` means recompute every selected model. Invalidate all
            # old completion markers up front, including models not yet reached if
            # this process is interrupted.
            for label, _factory in model_specs:
                stale = _encoding_paths(
                    output_dir, output_ids[label], config["regression"]
                )
                for key in ("algonauts", "triple_n", "done", "error", "skipped"):
                    stale[key].unlink(missing_ok=True)

    for model_index, (label, factory) in enumerate(model_specs, start=1):
        output_id = output_ids[label]
        model_fingerprint = _factory_fingerprint(factory)
        model_fingerprints[label] = model_fingerprint
        if not getattr(factory, "encoding_cv_safe", True):
            reason = (
                "feature preprocessing is fitted over all stimuli before CV "
                "(data leakage)"
            )
            results["skipped"][label] = reason
            warnings.warn(f"encoding skipped for {label!r}: {reason}")
            if output_dir is not None:
                paths = _encoding_paths(
                    output_dir, output_id, config["regression"]
                )
                for key in ("algonauts", "triple_n", "done", "error", "skipped"):
                    paths[key].unlink(missing_ok=True)
                _write_encoding_skip(
                    output_dir,
                    output_id,
                    label,
                    config,
                    model_fingerprint,
                    reason,
                )
            continue
        if resume and output_dir is not None:
            tables = _read_encoding_checkpoint(
                output_dir, output_id, label, config, model_fingerprint
            )
            if tables is not None:
                results["algonauts"][label] = tables["algonauts"]
                results["triple_n"][label] = tables["triple_n"]
                print(f"[encoding] [{model_index}/{len(model_specs)}] resumed {label}")
                try:
                    _write_encoding_combined(results, output_dir)
                except Exception as exc:  # noqa: BLE001 -- combined CSV is recoverable
                    warnings.warn(f"could not refresh partial encoding CSV: {exc!r}")
                continue

        paths = (
            _encoding_paths(output_dir, output_id, config["regression"])
            if output_dir is not None
            else None
        )
        if paths is not None:
            for key in ("algonauts", "triple_n", "done", "error", "skipped"):
                paths[key].unlink(missing_ok=True)

        started = time.monotonic()
        model = None
        print(f"[encoding] [{model_index}/{len(model_specs)}] model: {label}")
        try:
            model = factory()
            if not getattr(model, "encoding_cv_safe", True):
                reason = (
                    "feature preprocessing is fitted over all stimuli before CV "
                    "(data leakage)"
                )
                results["skipped"][label] = reason
                warnings.warn(f"encoding skipped for {label!r}: {reason}")
                if output_dir is not None:
                    _write_encoding_skip(
                        output_dir,
                        output_id,
                        label,
                        config,
                        model_fingerprint,
                        reason,
                    )
                continue

            def report_progress(modality, group, current, total):
                print(
                    f"[encoding] {label}: {modality} {group} "
                    f"[{current}/{total}]"
                )

            tables = model.encoding_tables(
                algonauts,
                triple_n,
                shared_ids,
                subjects=subjects,
                rois=rois,
                area_labels=area_labels,
                regression=config["regression"],
                alphas=config["alphas"],
                outer_folds=config["outer_folds"],
                inner_folds=config["inner_folds"],
                seed=config["seed"],
                progress=report_progress,
            )
            elapsed = time.monotonic() - started
            if output_dir is not None:
                _write_encoding_checkpoint(
                    output_dir,
                    output_id,
                    label,
                    config,
                    model_fingerprint,
                    tables,
                    elapsed,
                )
            results["algonauts"][label] = tables["algonauts"]
            results["triple_n"][label] = tables["triple_n"]
            print(f"[encoding] completed {label} in {elapsed / 60:.1f} min")
            if output_dir is not None:
                try:
                    _write_encoding_combined(results, output_dir)
                except Exception as exc:  # noqa: BLE001 -- combined CSV is recoverable
                    warnings.warn(f"could not refresh partial encoding CSV: {exc!r}")
        except Exception as exc:  # noqa: BLE001 -- isolate expensive model runs
            elapsed = time.monotonic() - started
            results["errors"][label] = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"encoding failed for {label!r} after {elapsed / 60:.1f} min: {exc!r}"
            )
            if output_dir is not None:
                try:
                    _write_encoding_error(
                        output_dir,
                        output_id,
                        label,
                        config,
                        model_fingerprint,
                        exc,
                    )
                except OSError as write_exc:
                    warnings.warn(
                        f"could not write encoding error checkpoint: {write_exc!r}"
                    )
            if fail_fast:
                raise
        finally:
            if model is not None:
                del model
            gc.collect()

    return results


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _slug(text):
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower() or "model"


def _long_form(tables, modality, segmentation):
    """Stack {label: table} into tidy rows: model, modality, segmentation, region,
    rho, ceiling. ``segmentation`` names the grouping (e.g. ``"roi"`` or
    ``"area_label | category"``) so the different Triple-N views stay distinguishable."""
    frames = []
    for label, df in tables.items():
        flat = df.reset_index().rename(columns={df.index.name: "region"})
        flat.insert(0, "segmentation", segmentation)
        flat.insert(0, "modality", modality)
        flat.insert(0, "model", label)
        frames.append(flat)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_results(results, output_dir, make_plots=True):
    """Write per-model CSVs, one combined long-form CSV, and (optionally) the stacked
    heatmaps (one for Algonauts + one per Triple-N segmentation). Returns the path to
    the combined CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_tables, tn_tables = results["algonauts"], results["triple_n"]

    for label, df in algo_tables.items():
        df.to_csv(output_dir / f"{_slug(label)}_algonauts.csv")
    for name, tables in tn_tables.items():
        for label, df in tables.items():
            df.to_csv(output_dir / f"{_slug(label)}_triple_n_{_slug(name)}.csv")

    long_frames = [_long_form(algo_tables, "algonauts", "roi")]
    for name, tables in tn_tables.items():
        long_frames.append(_long_form(tables, "triple_n", name))
    long = pd.concat(long_frames, ignore_index=True)
    combined = output_dir / "rsa_all.csv"
    long.to_csv(combined, index=False)
    print(f"[rsa] wrote {combined} ({len(long)} rows)")

    if make_plots:
        plots = [(algo_tables, "Algonauts fMRI ROI", "rsa_algonauts",
                  "RSA: model RDM x Algonauts ROI (+ noise ceiling)")]
        for name, tables in tn_tables.items():
            plots.append((tables, f"Triple-N {name}", f"rsa_triple_n_{_slug(name)}",
                          f"RSA: model RDM x Triple-N {name} (+ noise ceiling)"))
        for tables, xlabel, stem, title in plots:
            if not tables:
                continue
            fig = ModelAnalysisBase.plot_corr_table(tables, xlabel=xlabel, title=title)
            path = output_dir / f"{stem}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"[rsa] wrote {path}")
            import matplotlib.pyplot as plt
            plt.close(fig)
    return combined


def _encoding_long_form(results):
    """Stack encoding tables into one review-friendly tidy table."""
    frames = []
    groups = (
        ("algonauts", "roi", results.get("algonauts", {})),
        ("triple_n", "area_label", results.get("triple_n", {})),
    )
    for modality, segmentation, tables in groups:
        for label, table in tables.items():
            flat = table.rename_axis("region").reset_index()
            flat.insert(0, "segmentation", segmentation)
            flat.insert(0, "modality", modality)
            flat.insert(0, "model", label)
            frames.append(flat)

    base_columns = [
        "model",
        "modality",
        "segmentation",
        "region",
        *RESULT_COLUMNS,
        "regression",
        "outer_folds",
        "inner_folds",
        "seed",
        "alpha_grid",
    ]
    if not frames:
        return pd.DataFrame(columns=base_columns)

    combined = pd.concat(frames, ignore_index=True)
    config = results["config"]
    combined["regression"] = config["regression"]
    combined["outer_folds"] = config["outer_folds"]
    combined["inner_folds"] = config["inner_folds"]
    combined["seed"] = config["seed"]
    combined["alpha_grid"] = ",".join(f"{alpha:g}" for alpha in config["alphas"])
    return combined.reindex(columns=base_columns)


def _write_encoding_combined(results, output_dir):
    output_dir = Path(output_dir)
    regression = results["config"]["regression"]
    combined = output_dir / f"encoding_{regression}_all.csv"
    long = _encoding_long_form(results)
    _atomic_csv(long, combined, index=False)
    return combined


def _encoding_vmax(results):
    finite = []
    for modality in ("algonauts", "triple_n"):
        for table in results.get(modality, {}).values():
            if "mean_encoding_score" not in table:
                continue
            values = pd.to_numeric(
                table["mean_encoding_score"], errors="coerce"
            ).to_numpy(dtype=float)
            finite.extend(np.abs(values[np.isfinite(values)]))
    return max(finite) if finite and max(finite) > 0 else 1.0


def save_encoding_results(results, output_dir, make_plots=True):
    """Finalize encoding checkpoints, combined CSV, heatmaps, and run manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = results["config"]
    labels = list(dict.fromkeys([
        *results.get("algonauts", {}),
        *results.get("triple_n", {}),
    ]))
    output_ids = results.get("output_ids") or _model_output_ids(
        [(label, None) for label in labels]
    )
    results["output_ids"] = output_ids
    model_fingerprints = results.get("model_fingerprints") or {
        label: hashlib.sha256(f"unspecified:{label}".encode("utf-8")).hexdigest()
        for label in labels
    }
    results["model_fingerprints"] = model_fingerprints

    completed_labels = [
        label for label in labels
        if label in results["algonauts"] and label in results["triple_n"]
    ]
    for label in completed_labels:
        if _read_encoding_checkpoint(
            output_dir,
            output_ids[label],
            label,
            config,
            model_fingerprints[label],
        ) is None:
            _write_encoding_checkpoint(
                output_dir,
                output_ids[label],
                label,
                config,
                model_fingerprints[label],
                {
                    "algonauts": results["algonauts"][label],
                    "triple_n": results["triple_n"][label],
                },
            )

    combined = _write_encoding_combined(results, output_dir)
    print(f"[encoding] wrote {combined} ({len(_encoding_long_form(results))} rows)")

    if make_plots:
        import matplotlib.pyplot as plt

        vmax = _encoding_vmax(results)
        plots = (
            (
                results.get("algonauts", {}),
                "Algonauts fMRI ROI",
                "algonauts",
                "Encoding: model features → Algonauts ROI",
            ),
            (
                results.get("triple_n", {}),
                "Triple-N area",
                "triple_n_area_label",
                "Encoding: model features → Triple-N area",
            ),
        )
        for tables, xlabel, stem, title in plots:
            figure = ModelAnalysisBase.plot_encoding_table(
                tables,
                xlabel=xlabel,
                title=f"{title} ({config['regression']})",
                vmax=vmax,
            )
            if figure is None:
                continue
            path = output_dir / f"encoding_{config['regression']}_{stem}.png"
            figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            print(f"[encoding] wrote {path}")

    run_manifest = output_dir / f"encoding_{config['regression']}_run.json"
    _atomic_json(
        {
            "schema_version": 2,
            "config": config,
            "combined_csv": combined.name,
            "completed_models": completed_labels,
            "output_ids": output_ids,
            "model_fingerprints": model_fingerprints,
            "errors": results.get("errors", {}),
            "skipped": results.get("skipped", {}),
        },
        run_manifest,
    )
    return combined


def _best(df):
    """(region, rho) for the highest finite correlation in a table, or None."""
    if df is None or not len(df):
        return None
    rho = df["spearman_rho"].dropna()
    return (rho.idxmax(), float(rho.max())) if len(rho) else None


def print_summary(results):
    print("[rsa] best region per model:")
    for label in results["algonauts"]:
        parts = []
        best = _best(results["algonauts"][label])
        if best is not None:
            parts.append(f"Algonauts {best[0]}={best[1]:.3f}")
        for name, tables in results["triple_n"].items():
            best = _best(tables.get(label))
            if best is not None:
                parts.append(f"Triple-N[{name}] {best[0]}={best[1]:.3f}")
        print(f"    {label}: " + ("; ".join(parts) if parts else "(no comparable regions)"))


def _best_encoding(table):
    """(region, score, alpha) for the best finite held-out encoding score."""
    if table is None or not len(table):
        return None
    scores = pd.to_numeric(table["mean_encoding_score"], errors="coerce").dropna()
    if not len(scores):
        return None
    region = scores.idxmax()
    alpha = pd.to_numeric(
        pd.Series([table.loc[region, "best_alpha"]]), errors="coerce"
    ).iloc[0]
    return region, float(scores.loc[region]), float(alpha)


def print_encoding_summary(results):
    print(
        "[encoding] best neural subset per model "
        "(nested-CV Pearson r; final alpha from full-data inner CV):"
    )
    labels = list(dict.fromkeys([
        *results.get("algonauts", {}),
        *results.get("triple_n", {}),
    ]))
    for label in labels:
        parts = []
        for modality, display in (("algonauts", "Algonauts"),
                                  ("triple_n", "Triple-N")):
            best = _best_encoding(results.get(modality, {}).get(label))
            if best is not None:
                region, score, alpha = best
                alpha_text = f", final alpha={alpha:g}" if np.isfinite(alpha) else ""
                parts.append(f"{display} {region}={score:.3f}{alpha_text}")
        print(f"    {label}: " + ("; ".join(parts) if parts else "(no scorable groups)"))
    for label, error in results.get("errors", {}).items():
        print(f"    {label}: ERROR — {error}")
    for label, reason in results.get("skipped", {}).items():
        print(f"    {label}: SKIPPED — {reason}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_dir(cli_value, env_key):
    value = cli_value or os.environ.get(env_key)
    if not value:
        flag = "--" + env_key.lower().replace("_", "-")
        raise SystemExit(f"error: pass {flag} or set {env_key} (env or repo .env)")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run RSA and/or cross-validated neural encoding for every "
                    "discovered model.")
    parser.add_argument(
        "--method",
        choices=("rsa", "encoding", "both"),
        default="rsa",
        help="analysis to run (default: rsa; encoding is opt-in because it is slow)",
    )
    parser.add_argument("--algonauts-dir", help="Algonauts root (default: $ALGONAUTS_DIR)")
    parser.add_argument("--triple-n-dir", help="Triple-N root (default: $TRIPLE_N_DIR)")
    parser.add_argument("--nsd-mat", default=str(DEFAULT_NSD_MAT),
                        help="path to nsd_expdesign.mat")
    parser.add_argument("--checkpoints-root", default=str(DEFAULT_CHECKPOINTS_ROOT),
                        help="root analyzers search for checkpoints (e.g. beta_vae/*/vae.pth)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where CSVs and PNGs are written (default depends on --method)",
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)),
                        help="Algonauts subjects to average ROI RDMs over")
    parser.add_argument("--rois", nargs="+", default=None,
                        help="restrict to these Algonauts ROIs (default: all)")
    parser.add_argument("--areas", nargs="+", default=None,
                        help="restrict the coarse-area segmentation to these Triple-N "
                             "area labels (default: all); other segmentations unaffected")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=("run model names matching any case-insensitive glob or substring; "
              "the memory-heavy Pixel encoding baseline is opt-in here"),
    )
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument(
        "--regression",
        choices=("ridge", "lasso"),
        default="ridge",
        help="encoding regressor (default: ridge)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="encoding alpha sweep (default: 1e-4 ... 1e4, log-spaced)",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help="outer encoding CV folds (default: 5)",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="inner alpha-selection CV folds (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="encoding CV seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exact-config completed encoding models",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first encoding model failure (default: continue)",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip the heatmap PNGs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _load_dotenv()

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")                           # render figures without a display

    algonauts_dir = _resolve_dir(args.algonauts_dir, "ALGONAUTS_DIR")
    triple_n_dir = _resolve_dir(args.triple_n_dir, "TRIPLE_N_DIR")

    shared_ids = shared_stimuli(args.nsd_mat)
    algonauts = Algonauts(algonauts_dir, shared_ids)
    triple_n = TripleN(triple_n_dir)

    analyzers = discover_analyzers()
    print(f"[runner] analyzers: {', '.join(c.__name__ for c in analyzers) or '(none)'}")
    discovered_specs = collect_models(
        analyzers, triple_n_path=triple_n_dir,
        checkpoints_root=args.checkpoints_root, device=args.device)
    model_specs = filter_model_specs(discovered_specs, args.models)
    if not model_specs:
        if discovered_specs and args.models:
            available = ", ".join(label for label, _factory in discovered_specs)
            raise SystemExit(
                f"error: --models matched nothing; available models: {available}"
            )
        raise SystemExit("error: no runnable models discovered; nothing to compare")
    print(f"[runner] {len(model_specs)} model(s) to run: "
          + ", ".join(label for label, _factory in model_specs))

    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    elif args.method == "rsa":
        output_root = DEFAULT_OUTPUT_DIR
    elif args.method == "encoding":
        output_root = DEFAULT_ENCODING_OUTPUT_DIR
    else:
        output_root = DEFAULT_BOTH_OUTPUT_DIR

    rsa_results = None
    encoding_results = None
    if args.method in {"rsa", "both"}:
        rsa_output = output_root / "rsa" if args.method == "both" else output_root
        rsa_results = run_rsa(
            algonauts,
            triple_n,
            shared_ids,
            model_specs,
            subjects=args.subjects,
            rois=args.rois,
            area_labels=args.areas,
        )
        save_results(rsa_results, rsa_output, make_plots=not args.no_plots)
        print_summary(rsa_results)

    if args.method in {"encoding", "both"}:
        encoding_output = (
            output_root / "encoding" if args.method == "both" else output_root
        )
        encoding_specs = model_specs
        if args.models is None:
            excluded = [
                label for label, factory in encoding_specs
                if not getattr(factory, "encoding_default", True)
            ]
            encoding_specs = [
                spec for spec in encoding_specs
                if getattr(spec[1], "encoding_default", True)
            ]
            if excluded:
                print(
                    "[encoding] excluded from the default batch for memory safety: "
                    + ", ".join(excluded)
                    + " (select explicitly with --models to run)"
                )
        if not encoding_specs:
            raise SystemExit(
                "error: no models are eligible for the default encoding batch; "
                "select an opt-in model explicitly with --models"
            )
        encoding_results = run_encoding(
            algonauts,
            triple_n,
            shared_ids,
            encoding_specs,
            subjects=args.subjects,
            rois=args.rois,
            area_labels=args.areas,
            regression=args.regression,
            alphas=args.alphas,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            output_dir=encoding_output,
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
        save_encoding_results(
            encoding_results,
            encoding_output,
            make_plots=not args.no_plots,
        )
        print_encoding_summary(encoding_results)
        if encoding_results["errors"]:
            raise SystemExit(
                f"error: {len(encoding_results['errors'])} encoding model(s) failed; "
                "completed outputs were preserved"
            )
        if not encoding_results["algonauts"] and encoding_results.get("skipped"):
            raise SystemExit(
                "error: no encoding model completed; all selected models were skipped"
            )

    if args.method == "both":
        return {"rsa": rsa_results, "encoding": encoding_results}
    return rsa_results if args.method == "rsa" else encoding_results


if __name__ == "__main__":
    main()
