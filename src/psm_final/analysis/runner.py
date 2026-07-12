"""Model-agnostic RSA runner.

Discovers every implemented :class:`~psm_final.analysis.model.ModelAnalysisBase`
subclass, builds each model's RDM over one shared stimulus set, and Spearman-
correlates it against **every** Algonauts fMRI ROI and **every** Triple-N area
label. Writes per-model CSV tables, a combined long-form CSV, and two stacked
heatmaps (models x brain regions).

Why one runner instead of per-model notebooks: the brain (Algonauts / Triple-N)
group RDMs depend only on the shared stimuli, NOT on any model, so they are
computed ONCE here and reused across every model -- the expensive per-(subject,
ROI) fMRI reloads happen a single time rather than once per model.

Every model plugs into the same base class, so adding a model to this run means
nothing more than writing its ``ModelAnalysisBase`` subclass (with an
``embedding`` and a ``discover``) -- see CONTRIBUTING.md section 4.

Run it::

    python -m psm_final.analysis.runner            # uses $ALGONAUTS_DIR / $TRIPLE_N_DIR
    psm-rsa --subjects 1 2 5 7 --output-dir results/rsa
"""
from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import psm_final.analysis as _analysis_pkg
from psm_final.analysis.model import ModelAnalysisBase
from psm_final.dataset.algonauts import Algonauts
from psm_final.dataset.triple_n import TripleN
from psm_final.dataset.util import shared_stimuli

# .../src/psm_final/analysis/runner.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NSD_MAT = _REPO_ROOT / "nsd_expdesign.mat"
DEFAULT_CHECKPOINTS_ROOT = _REPO_ROOT          # analyzers glob e.g. beta_vae/*/vae.pth under here
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "rsa"


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
            rois=None, area_labels=None):
    """Correlate every model's RDM against every Algonauts ROI and Triple-N area.

    The brain group RDMs + noise ceilings are model-independent, so they are built
    once (reusing the model-agnostic building blocks on :class:`ModelAnalysisBase`)
    and every model RDM is correlated against the same cached set.

    Returns ``{"algonauts": {label: DataFrame}, "triple_n": {label: DataFrame},
    "n_stimuli": int, "subjects": [...]}`` -- each DataFrame is exactly what
    :meth:`ModelAnalysisBase.rsa_algonauts` / ``rsa_triple_n`` produces.
    """
    subjects = list(subjects)
    rois = list(Algonauts.ALGO_ROIS if rois is None else rois)
    area_labels = (sorted(triple_n.units["area_label"].unique())
                   if area_labels is None else list(area_labels))

    # --- one shared stimulus set (present in every subject + mapped to Triple-N) ---
    nsd_ids, stim_index = ModelAnalysisBase.aligned_stimuli(
        algonauts, triple_n, shared_ids, subjects)
    model_indices = np.asarray(stim_index) - 1          # 0-based positions into StimuliNNN
    print(f"[rsa] {len(stim_index)} shared stimuli | {len(rois)} ROIs, "
          f"{len(area_labels)} areas | subjects={subjects}")

    # --- brain RDMs: computed ONCE, reused for every model ---
    algo_rdms, algo_nc = ModelAnalysisBase._algonauts_group_rdms(
        algonauts, nsd_ids, subjects, rois)
    tn_rdms, tn_nc = ModelAnalysisBase._triple_n_group_rdms(
        triple_n, stim_index, area_labels)
    print(f"[rsa] brain RDMs ready: {len(algo_rdms)}/{len(rois)} ROIs and "
          f"{len(tn_rdms)}/{len(area_labels)} areas had enough data")

    algo_tables, tn_tables = {}, {}
    for label, factory in model_specs:
        print(f"[rsa] model: {label}")
        model = factory()                               # loads exactly one checkpoint at a time
        model_rdm = model.rdm(indices=model_indices)
        algo_tables[label] = ModelAnalysisBase._correlate(
            model_rdm, algo_rdms, algo_nc, rois, index_name="roi")
        tn_tables[label] = ModelAnalysisBase._correlate(
            model_rdm, tn_rdms, tn_nc, area_labels, index_name="area_label")
        del model                                       # free the model before loading the next

    return {"algonauts": algo_tables, "triple_n": tn_tables,
            "n_stimuli": len(stim_index), "subjects": subjects}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _slug(text):
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower() or "model"


def _long_form(tables, modality):
    """Stack {label: table} into tidy rows: model, modality, region, rho, ceiling."""
    frames = []
    for label, df in tables.items():
        flat = df.reset_index().rename(columns={df.index.name: "region"})
        flat.insert(0, "modality", modality)
        flat.insert(0, "model", label)
        frames.append(flat)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_results(results, output_dir, make_plots=True):
    """Write per-model CSVs, one combined long-form CSV, and (optionally) the two
    stacked heatmaps. Returns the path to the combined CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_tables, tn_tables = results["algonauts"], results["triple_n"]

    for label, df in algo_tables.items():
        df.to_csv(output_dir / f"{_slug(label)}_algonauts.csv")
    for label, df in tn_tables.items():
        df.to_csv(output_dir / f"{_slug(label)}_triple_n.csv")

    long = pd.concat(
        [_long_form(algo_tables, "algonauts"), _long_form(tn_tables, "triple_n")],
        ignore_index=True)
    combined = output_dir / "rsa_all.csv"
    long.to_csv(combined, index=False)
    print(f"[rsa] wrote {combined} ({len(long)} rows)")

    if make_plots:
        for tables, xlabel, stem, title in [
            (algo_tables, "Algonauts fMRI ROI", "rsa_algonauts",
             "RSA: model RDM x Algonauts ROI (+ noise ceiling)"),
            (tn_tables, "Triple-N area label", "rsa_triple_n",
             "RSA: model RDM x Triple-N area (+ noise ceiling)"),
        ]:
            if not tables:
                continue
            fig = ModelAnalysisBase.plot_corr_table(tables, xlabel=xlabel, title=title)
            path = output_dir / f"{stem}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"[rsa] wrote {path}")
    return combined


def _best(df):
    """(region, rho) for the highest finite correlation in a table, or None."""
    if not len(df):
        return None
    rho = df["spearman_rho"].dropna()
    return (rho.idxmax(), float(rho.max())) if len(rho) else None


def print_summary(results):
    print("[rsa] best region per model:")
    for label in results["algonauts"]:
        parts = []
        for modality, table in [("Algonauts", results["algonauts"][label]),
                                ("Triple-N", results["triple_n"][label])]:
            best = _best(table)
            if best is not None:
                parts.append(f"{modality} {best[0]}={best[1]:.3f}")
        print(f"    {label}: " + ("; ".join(parts) if parts else "(no comparable regions)"))


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
        description="Compare every implemented model RDM against every Algonauts "
                    "ROI and Triple-N area.")
    parser.add_argument("--algonauts-dir", help="Algonauts root (default: $ALGONAUTS_DIR)")
    parser.add_argument("--triple-n-dir", help="Triple-N root (default: $TRIPLE_N_DIR)")
    parser.add_argument("--nsd-mat", default=str(DEFAULT_NSD_MAT),
                        help="path to nsd_expdesign.mat")
    parser.add_argument("--checkpoints-root", default=str(DEFAULT_CHECKPOINTS_ROOT),
                        help="root analyzers search for checkpoints (e.g. beta_vae/*/vae.pth)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="where CSVs and PNGs are written")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)),
                        help="Algonauts subjects to average ROI RDMs over")
    parser.add_argument("--rois", nargs="+", default=None,
                        help="restrict to these Algonauts ROIs (default: all)")
    parser.add_argument("--areas", nargs="+", default=None,
                        help="restrict to these Triple-N area labels (default: all)")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
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
    print(f"[rsa] analyzers: {', '.join(c.__name__ for c in analyzers) or '(none)'}")
    model_specs = collect_models(
        analyzers, triple_n_path=triple_n_dir,
        checkpoints_root=args.checkpoints_root, device=args.device)
    if not model_specs:
        raise SystemExit("error: no runnable models discovered; nothing to compare")
    print(f"[rsa] {len(model_specs)} model(s) to run")

    results = run_rsa(algonauts, triple_n, shared_ids, model_specs,
                      subjects=args.subjects, rois=args.rois, area_labels=args.areas)
    save_results(results, args.output_dir, make_plots=not args.no_plots)
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
