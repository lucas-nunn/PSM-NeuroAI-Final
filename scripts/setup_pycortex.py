"""
One-time setup: imports fsaverage from FreeSurfer into the pycortex filestore.
Run this once per machine before using any brain plotting functions.

Usage:
    python scripts/setup_pycortex.py
    python scripts/setup_pycortex.py --fs-home /path/to/freesurfer
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_FS_HOME = os.environ.get("FREESURFER_HOME")


def setup_pycortex(fs_home: str) -> None:
    if not shutil.which("mri_convert", path=f"{fs_home}/bin"):
        print(f"ERROR: mri_convert not found in {fs_home}/bin")
        print("Check that FREESURFER_HOME is correct.")
        sys.exit(1)

    os.environ["FREESURFER_HOME"] = fs_home
    os.environ["FS_LICENSE"] = f"{fs_home}/license.txt"
    os.environ["SUBJECTS_DIR"] = f"{fs_home}/subjects"
    os.environ["PATH"] = f"{fs_home}/bin:" + os.environ["PATH"]

    import cortex

    filestore = Path(cortex.db.filestore)
    surface = filestore / "fsaverage" / "surfaces" / "wm_lh.gii"
    if surface.exists():
        print(f"fsaverage already imported at {filestore}. Nothing to do.")
        return

    print(f"Importing fsaverage into pycortex filestore at {filestore} ...")
    cortex.freesurfer.import_subj(
        "fsaverage",
        pycortex_subject="fsaverage",
        freesurfer_subject_dir=f"{fs_home}/subjects",
        whitematter_surf="white",
    )
    cortex.freesurfer.import_flat(
        "fsaverage",
        "cortex.patch",
        cx_subject="fsaverage",
        freesurfer_subject_dir=f"{fs_home}/subjects",
        auto_overwrite=True,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fs-home", default=DEFAULT_FS_HOME, help="Path to FreeSurfer installation")
    args = parser.parse_args()
    if not args.fs_home:
        print("ERROR: missing FreeSurfer path.")
        print("Set FREESURFER_HOME or pass --fs-home /path/to/freesurfer")
        sys.exit(1)
    setup_pycortex(args.fs_home)
