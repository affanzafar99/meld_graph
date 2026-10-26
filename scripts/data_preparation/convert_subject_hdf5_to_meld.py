"""
Convert per-subject HDF5 files into the per-site layout that meld_graph training reads.

Input  (one file per subject, e.g. MELD_H5_3T_C_0001_control_featurematrix_combat.hdf5):
    /lh/<feature>   /rh/<feature>          datasets of shape (163842,)

Output (one file per site and group, in <meld_data_path>/output/preprocessed_surf_data/MELD_<site>/):
    <site>_<group>_featurematrix_combat.hdf5
        /<site>/<scanner>/<group>/<subject_id>/<lh|rh>/<feature>

Site ("Harmo code"), group ("Group") and scanner ("Scanner") come from the demographics CSV, the subject ID
from the file name (the demographics ID the file name starts with). Your input files are only read.

Run from the repository root with the meld_graph environment active:
    python scripts/data_preparation/convert_subject_hdf5_to_meld.py \
        --input /home/affan/MSCS/data/my_data/meld_combats \
        --demographics_csv /home/affan/MSCS/data/my_data/demographics_testbatch_meld.csv

    --features standard   (default) the 34 features of the released MELD Graph models, plus lesion masks
    --features all        every feature in the files (larger output)
    --dry_run             only report what would be written
    --overwrite           replace existing output files
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from meld_graph.paths import BASE_PATH, DEFAULT_HDF5_FILE_ROOT, NVERT

LESION = ".on_lh.lesion.mgh"
BOUNDARY = ".on_lh.boundary_zone.mgh"
STANDARD_FEATURES = [
    f"{prefix}.on_lh.{name}.mgh"
    for prefix in (".combat", ".inter_z.intra_z.combat", ".inter_z.asym.intra_z.combat")
    for name in (
        "pial.K_filtered.sm20", "thickness.sm3", "thickness_regression.sm3", "w-g.pct.sm3", "sulc.sm3", "curv.sm3",
        "gm_FLAIR_0.75.sm3", "gm_FLAIR_0.5.sm3", "gm_FLAIR_0.25.sm3", "gm_FLAIR_0.sm3", "wm_FLAIR_0.5.sm3", "wm_FLAIR_1.sm3",
    )
    if not (prefix != ".combat" and name == "thickness.sm3")
]


def meld_scanner(value):
    """Same mapping as MeldSubject.scanner."""
    value = str(value)
    if value in ("15T", "1.5T", "15t", "1.5t"):
        return "15T"
    if value in ("3T", "3t"):
        return "3T"
    return "XT"


def single_column(df, name):
    cols = [c for c in df.columns if name in c]
    if len(cols) != 1:
        sys.exit(f"Demographics: need exactly one column containing '{name}', found {cols}")
    return cols[0]


def main():
    parser = argparse.ArgumentParser(description="Convert per-subject HDF5 files to meld_graph site files")
    parser.add_argument("--input", nargs="+", required=True, help="per-subject HDF5 files and/or folders containing them")
    parser.add_argument("--demographics_csv", required=True, help="CSV with ID, Harmo code, Group (, Scanner)")
    parser.add_argument("--features", choices=["standard", "all"], default="standard")
    parser.add_argument("--output_dir", default=BASE_PATH, help=f"default: BASE_PATH = {BASE_PATH}")
    parser.add_argument("--compression", type=int, default=4, help="gzip level 0-9 (default 4)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # ---- demographics
    demo = pd.read_csv(args.demographics_csv, header=0, encoding="latin")
    id_col = [c for c in demo.columns if "ID" in c][-1]
    harmo_col = single_column(demo, "Harmo code")
    group_col = single_column(demo, "Group")
    scanner_cols = [c for c in demo.columns if "Scanner" in c]
    demo[id_col] = demo[id_col].astype(str)
    demo = demo.drop_duplicates(id_col).set_index(id_col)
    ids_longest_first = sorted(demo.index, key=len, reverse=True)

    # ---- input files
    files = []
    for item in args.input:
        if os.path.isdir(item):
            files += sorted(glob.glob(os.path.join(item, "*.hdf5")) + glob.glob(os.path.join(item, "*.h5")))
        else:
            files.append(item)
    if not files:
        sys.exit("No HDF5 files found in --input")

    plan = []  # dicts with subject, file, site, scanner, group, features per hemi, has lesion
    unmatched, problems = [], []
    for path in files:
        base = os.path.basename(path)
        subj = next((i for i in ids_longest_first if base.startswith(i + "_") or os.path.splitext(base)[0] == i), None)
        if subj is None:
            unmatched.append(base)
            continue
        row = demo.loc[subj]
        site, group = str(row[harmo_col]), str(row[group_col])
        scanner = meld_scanner(row[scanner_cols[0]]) if len(scanner_cols) == 1 else "XT"
        if group not in ("patient", "control"):
            problems.append(f"{subj}: Group '{group}' must be 'patient' or 'control'")
            continue
        if f"_{group}_" not in base and ("patient" in base or "control" in base):
            problems.append(f"{base}: file name group does not match demographics Group '{group}'")
        with h5py.File(path, "r") as f:
            feats = {}
            for hemi in ("lh", "rh"):
                if hemi not in f or not isinstance(f[hemi], h5py.Group):
                    problems.append(f"{base}: no /{hemi} group")
                    feats[hemi] = set()
                    continue
                feats[hemi] = {k for k, v in f[hemi].items() if isinstance(v, h5py.Dataset) and v.shape == (NVERT,)}
                wrong = [k for k, v in f[hemi].items() if isinstance(v, h5py.Dataset) and v.shape != (NVERT,)]
                if wrong:
                    problems.append(f"{base}: /{hemi} datasets with wrong shape (expected ({NVERT},)): {wrong[:3]}")
            lesion_vertices = sum(
                int(np.count_nonzero(f[h][LESION][:] > 0)) for h in ("lh", "rh") if h in f and LESION in f[h]
            )
        plan.append(dict(subject=subj, file=path, site=site, scanner=scanner, group=group,
                         features=feats, lesion_vertices=lesion_vertices))

    # ---- features to copy
    available = set().union(*[p["features"]["lh"] | p["features"]["rh"] for p in plan]) if plan else set()
    if args.features == "standard":
        keep = [f for f in STANDARD_FEATURES if f in available]
        missing_std = [f for f in STANDARD_FEATURES if f not in available]
    else:
        keep = sorted(available - {LESION, BOUNDARY})
        missing_std = [f for f in STANDARD_FEATURES if f not in available]
    keep_with_masks = keep + [m for m in (LESION, BOUNDARY) if m in available]

    # ---- outputs
    outputs = defaultdict(list)
    for p in plan:
        out = os.path.join(args.output_dir, f"MELD_{p['site']}", DEFAULT_HDF5_FILE_ROOT.format(site_code=p["site"], group=p["group"]))
        outputs[out].append(p)
    for out in outputs:
        if os.path.exists(out) and not args.overwrite:
            problems.append(f"{out} already exists (use --overwrite to replace it)")

    # ---- report
    n_pat = sum(p["group"] == "patient" for p in plan)
    print(f"{len(files)} input files, {len(plan)} matched to demographics ({n_pat} patients, {len(plan) - n_pat} controls)")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files match no demographics ID and are skipped, e.g. {unmatched[:5]}")
    no_lesion = [p["subject"] for p in plan if p["group"] == "patient" and p["lesion_vertices"] == 0]
    print(f"patients with a lesion mask ({LESION}): {n_pat - len(no_lesion)} of {n_pat}")
    if no_lesion:
        print(f"WARNING: patients without a lesion mask (left out of training by meld_graph): {no_lesion}")
    print(f"features to copy ({args.features}): {len(keep)}")
    if missing_std:
        print(f"WARNING: {len(missing_std)} of the 34 standard MELD Graph features are not in your files: {missing_std}")
    incomplete = [p["subject"] for p in plan if any(f not in p["features"][h] for h in ("lh", "rh") for f in keep)]
    if incomplete:
        print(f"WARNING: {len(incomplete)} subjects lack some of these features (MELD loads them as zeros), e.g. {incomplete[:5]}")
    for out, subs in outputs.items():
        print(f"  -> {out}: {len(subs)} subjects")
    for prob in problems:
        print(f"ERROR: {prob}")
    if problems:
        sys.exit(f"\n{len(problems)} problem(s) - nothing written.")
    if not keep:
        sys.exit("No features to copy.")
    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    # ---- write
    for out, subs in outputs.items():
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with h5py.File(out, "w") as dst:
            for k, p in enumerate(subs, 1):
                with h5py.File(p["file"], "r") as src:
                    for hemi in ("lh", "rh"):
                        grp = dst.require_group(f"{p['site']}/{p['scanner']}/{p['group']}/{p['subject']}/{hemi}")
                        for feat in keep_with_masks:
                            if hemi in src and feat in src[hemi]:
                                grp.create_dataset(
                                    feat, data=src[hemi][feat][:].astype(np.float32),
                                    compression="gzip", compression_opts=args.compression,
                                )
                print(f"\r  {os.path.basename(out)}: {k}/{len(subs)} subjects", end="", flush=True)
        print(f"\n  wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")

    sites = sorted({p["site"] for p in plan})
    print("\nDone. For meld_graph use:")
    print(f'    "hdf5_file_root": "{DEFAULT_HDF5_FILE_ROOT}",')
    print(f'    "site_codes": {sites},')


if __name__ == "__main__":
    main()
