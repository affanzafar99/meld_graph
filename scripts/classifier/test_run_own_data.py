"""
Check your own MELD data and run a short test training on a small subset of it.

Run from the repository root with the meld_graph environment active:

    python scripts/classifier/test_run_own_data.py \
        --hdf5 /path/to/H1_patient_featurematrix.hdf5 /path/to/H1_control_featurematrix.hdf5 \
        --split_csv /path/to/split.csv \
        --demographics_csv /path/to/demographics.csv

    --check_only        only check the files and print the report, do not train
    --n_subjects 20     number of train/val subjects used for the test run (about half lesional patients)
    --epochs 2          training epochs
    --batch_size 2
    --features common   "common": only features every selected subject has; "all": every feature found
                        (missing features are loaded as zeros by MELD)
    --write_zscore F    after the checks, compute z-score parameters from ALL usable trainval subjects,
                        write them to F (JSON) and stop (for full training runs)

What it does:
  1. Reads every HDF5 file (layout /<site>/<scanner>/<group>/<subject>/<lh|rh>/<feature>) and checks
     shapes (163842 vertices), hemispheres, lesion masks and features.
  2. Checks the demographics CSV (ID, Harmo code, Group, optional Scanner) and the split CSV
     (subject_id, split) against the HDF5 contents, the way meld_graph reads them.
  3. Makes the HDF5 files visible where meld_graph looks for them
     (<meld_data_path>/output/preprocessed_surf_data/MELD_<site>/) using symlinks, if they are elsewhere.
  4. Computes z-score parameters (mean/std per feature, cortex vertices only) from the selected subjects.
  5. Trains the standard MELD Graph U-Net for a few epochs on the subset and checks the outputs.

Nothing in the meld_graph code is changed. Your demographics file is copied to the temporary path
meld_graph reads it from (meld_graph.paths.DEMOGRAPHIC_FEATURES_FILE) for this run.
Output: <meld_data_path>/models/test_run_own_data/s_0/fold_00/
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

from meld_graph.paths import BASE_PATH, DEMOGRAPHIC_FEATURES_FILE, EXPERIMENT_PATH, NVERT, load_config

LESION = ".on_lh.lesion.mgh"
BOUNDARY = ".on_lh.boundary_zone.mgh"
REFERENCE_CONFIG = "scripts/config_files/24-08-01_MRIN_dcp/fold_00/s_0.py"
EXP_NAME = "test_run_own_data/s_0"
SPLIT_SUBSET_CSV = "/tmp/meld_test_run_split.csv"

errors = []
warnings = []


def error(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


# ----------------------------------------------------------------------------------------------
# HDF5
# ----------------------------------------------------------------------------------------------
def describe_layout(f, max_lines=15):
    """First few object paths in an HDF5 file, to show its actual layout."""
    lines = []

    def visit(name, obj):
        if len(lines) >= max_lines:
            return True
        kind = f"dataset {obj.shape}" if isinstance(obj, h5py.Dataset) else "group"
        lines.append(f"      /{name}  ({kind})")

    f.visititems(visit)
    return "\n".join(lines)


def layout_is_meld(f):
    """True if the file has the depth meld_graph reads: /<site>/<scanner>/<group>/<subject>/<hemi>/<feature>."""
    def is_group(path):
        return isinstance(f.get(path, getlink=False), h5py.Group)

    for site in f.keys():
        for scanner in (f[site].keys() if is_group(site) else []):
            for group in (f[site][scanner].keys() if is_group(f"{site}/{scanner}") else []):
                grp = f"{site}/{scanner}/{group}"
                for subj in (f[grp].keys() if is_group(grp) else []):
                    for hemi in (f[f"{grp}/{subj}"].keys() if is_group(f"{grp}/{subj}") else []):
                        return is_group(f"{grp}/{subj}/{hemi}")
                    return False
                return False
            return False
        return False
    return False


def inspect_hdf5(path):
    """Return {site: {scanner: set(groups)}} and {subject_id: record} for one HDF5 file."""
    layout = {}
    subjects = {}
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        error(f"{path}: cannot open ({e})")
        return layout, subjects
    with f:
        if not layout_is_meld(f):
            error(f"{path}: layout is not /<site>/<scanner>/<patient|control>/<subject>/<lh|rh>/<feature>. "
                  f"It starts like this:\n{describe_layout(f)}")
            return layout, subjects
        for site in f.keys():
            layout[site] = {}
            if not isinstance(f[site], h5py.Group):
                error(f"{path}: top level '{site}' is not a group; expected /<site>/<scanner>/<group>/<subject>/<hemi>")
                continue
            for scanner in f[site].keys():
                layout[site][scanner] = set()
                for group in f[site][scanner].keys():
                    layout[site][scanner].add(group)
                    if group not in ("patient", "control"):
                        error(f"{path}: /{site}/{scanner}/{group}: group must be 'patient' or 'control'")
                    grp = f[site][scanner][group]
                    if not isinstance(grp, h5py.Group):
                        error(f"{path}: /{site}/{scanner}/{group} is not a group")
                        continue
                    for subj in grp.keys():
                        rec = {
                            "file": path, "site": site, "scanner": scanner, "group": group,
                            "features": {}, "lesion_hemis": [], "lesion_vertices": 0, "bad": [],
                        }
                        for hemi in ("lh", "rh"):
                            rec["features"][hemi] = set()
                            if hemi not in grp[subj]:
                                rec["bad"].append(f"no '{hemi}' group")
                                continue
                            for name, ds in grp[subj][hemi].items():
                                if not isinstance(ds, h5py.Dataset):
                                    continue
                                if ds.shape != (NVERT,):
                                    rec["bad"].append(f"{hemi}/{name} has shape {ds.shape}, expected ({NVERT},)")
                                    continue
                                if name == LESION:
                                    n = int(np.count_nonzero(ds[:] > 0))
                                    rec["lesion_hemis"].append(hemi)
                                    rec["lesion_vertices"] += n
                                elif name != BOUNDARY:
                                    rec["features"][hemi].add(name)
                        if subj in subjects:
                            error(f"{path}: subject {subj} appears more than once")
                        subjects[subj] = rec
    return layout, subjects


def derive_file_root(path, site, groups):
    """File name pattern meld_graph uses: '{site_code}_{group}_....hdf5' (or without {group})."""
    base = os.path.basename(path)
    if site not in base:
        return None
    root = base.replace(site, "{site_code}", 1)
    if len(groups) == 1:
        group = next(iter(groups))
        if group in root:
            root = root.replace(group, "{group}", 1)
    return root


def check_hdf5_files(hdf5_paths):
    all_subjects = {}
    roots = set()
    placements = []  # (source, expected)
    for path in hdf5_paths:
        path = os.path.abspath(path)
        layout, subjects = inspect_hdf5(path)
        if not layout:
            continue
        if len(layout) != 1:
            error(f"{path}: contains several sites {sorted(layout)}; meld_graph needs one file per site (and group)")
            continue
        site = next(iter(layout))
        groups = set().union(*layout[site].values()) if layout[site] else set()
        root = derive_file_root(path, site, groups)
        if root is None:
            error(f"{path}: the file name must contain the site code '{site}' "
                  "(meld_graph finds files by the pattern '{site_code}_{group}_<name>.hdf5')")
            continue
        if "{group}" not in root:
            # one file holds both groups: every scanner must then contain both groups
            for scanner, scanner_groups in layout[site].items():
                if scanner_groups != {"patient", "control"}:
                    error(f"{path}: the file name has no group ('patient'/'control') in it, so meld_graph reads it for "
                          f"both groups, but /{site}/{scanner} only has {sorted(scanner_groups)}. "
                          "Rename to '<site>_<group>_....hdf5' with one group per file, or add the missing group.")
        roots.add(root)
        for group in groups:
            expected = os.path.join(BASE_PATH, f"MELD_{site}", root.format(site_code=site, group=group))
            placements.append((path, expected))
        for subj, rec in subjects.items():
            if subj in all_subjects:
                error(f"subject {subj} is in two files: {all_subjects[subj]['file']} and {path}")
            all_subjects[subj] = rec

    if len(roots) > 1:
        error(f"HDF5 file names follow different patterns {sorted(roots)}; meld_graph needs one pattern "
              "(hdf5_file_root), e.g. H1_patient_featurematrix.hdf5, H2_control_featurematrix.hdf5")

    links = []
    for src, expected in sorted(set(placements)):
        if os.path.lexists(expected):
            if os.path.realpath(expected) != os.path.realpath(src):
                error(f"{expected} already exists and is not {src}")
        else:
            links.append((src, expected))
    for src, expected in links:
        print(f"  needs link {expected} -> {src}")

    root = next(iter(roots)) if len(roots) == 1 else None
    return all_subjects, root, links


# ----------------------------------------------------------------------------------------------
# CSVs
# ----------------------------------------------------------------------------------------------
def columns_matching(df, name):
    return [c for c in df.columns if name in c]


def check_demographics(path, subjects):
    """Mirror MeldSubject.get_demographic_features: substring column match, must be unique."""
    df = pd.read_csv(path, header=0, encoding="latin")
    id_cols = columns_matching(df, "ID")
    if not id_cols:
        error(f"{path}: no column containing 'ID'")
        return None
    if len(id_cols) > 1:
        warn(f"{path}: several columns contain 'ID' {id_cols}; meld_graph uses the last one, '{id_cols[-1]}'")
    id_col = id_cols[-1]
    if df[id_col].dtype != object:
        error(f"{path}: the ID column '{id_col}' is read as numbers; meld_graph compares IDs as text, "
              "so use text IDs that match the HDF5 subject names exactly")
    cols = {}
    for name, required in (("Harmo code", True), ("Group", True), ("Scanner", False)):
        matches = columns_matching(df, name)
        if len(matches) > 1:
            (error if required else warn)(f"{path}: several columns contain '{name}' {matches}; meld_graph needs exactly one")
        elif not matches:
            if required:
                error(f"{path}: no column containing '{name}'")
            else:
                warn(f"{path}: no 'Scanner' column; every subject counts as scanner 'XT' (fine, config includes XT)")
        else:
            cols[name] = matches[0]
    if "Harmo code" not in cols or "Group" not in cols:
        return None

    ids = df[id_col].astype(str)
    dups = sorted(set(ids[ids.duplicated()]))
    if dups:
        error(f"{path}: duplicate IDs (meld_graph cannot read these subjects): {dups[:10]}")
    df = df.assign(_id=ids).drop_duplicates("_id").set_index("_id")

    ok = set()
    not_in_demos = []
    for subj, rec in subjects.items():
        if subj not in df.index:
            not_in_demos.append(subj)
            continue
        row = df.loc[subj]
        group = str(row[cols["Group"]])
        harmo = str(row[cols["Harmo code"]])
        problems = []
        if group not in ("patient", "control"):
            problems.append(f"Group '{group}' (must be exactly 'patient' or 'control')")
        elif group != rec["group"]:
            problems.append(f"Group '{group}' but HDF5 group '{rec['group']}'")
        if harmo != rec["site"]:
            problems.append(f"Harmo code '{harmo}' but HDF5 site '{rec['site']}'")
        if problems:
            rec["demo_problems"] = problems
        else:
            ok.add(subj)
    bad = {s: subjects[s]["demo_problems"] for s in subjects if "demo_problems" in subjects[s]}
    if bad:
        examples = "; ".join(f"{s}: {', '.join(p)}" for s, p in list(bad.items())[:5])
        error(f"{len(bad)} HDF5 subjects have demographics that do not match, e.g. {examples}")
    if not_in_demos:
        warn(f"{len(not_in_demos)} HDF5 subjects are not in the demographics CSV (e.g. {not_in_demos[:5]}). "
             "If they are also in the split CSV, full training will crash on them.")
    return ok


def read_split(path):
    df = pd.read_csv(path)
    if "subject_id" in df.columns and df["subject_id"].dtype != object:
        error(f"{path}: subject_id is read as numbers; meld_graph compares IDs as text, "
              "so use text IDs that match the HDF5 subject names exactly")
    if "subject_id" not in df.columns or "split" not in df.columns:
        error(f"{path}: needs columns 'subject_id' and 'split' (found {list(df.columns)})")
        return None
    values = set(df["split"].astype(str))
    unknown = values - {"trainval", "test"}
    if unknown:
        error(f"{path}: 'split' values must be 'trainval' or 'test'; found {sorted(unknown)}. "
              "Subjects with other values are ignored by meld_graph (e.g. rename 'train'/'val' to 'trainval').")
    return df.assign(subject_id=df["subject_id"].astype(str))


# ----------------------------------------------------------------------------------------------
# subset, features, z-score
# ----------------------------------------------------------------------------------------------
def select_subjects(subjects, demo_ok, split_df, n_subjects, seed=0):
    trainval = set(split_df.loc[split_df["split"].astype(str) == "trainval", "subject_id"])
    usable = [s for s in subjects if s in demo_ok and s in trainval and not subjects[s]["bad"]]
    lesional = sorted(s for s in usable if subjects[s]["group"] == "patient" and subjects[s]["lesion_vertices"] > 0)
    controls = sorted(s for s in usable if subjects[s]["group"] == "control")
    no_lesion = [s for s in usable if subjects[s]["group"] == "patient" and subjects[s]["lesion_vertices"] == 0]
    if no_lesion:
        warn(f"{len(no_lesion)} patients have no (or an empty) lesion mask '{LESION}'; meld_graph leaves them out "
             f"of training (lesional_only), e.g. {no_lesion[:5]}")
    rng = np.random.default_rng(seed)
    n_les = min(len(lesional), max(2, n_subjects // 2))
    picked = list(rng.choice(lesional, n_les, replace=False)) if n_les else []
    n_con = min(len(controls), n_subjects - n_les)
    picked += list(rng.choice(controls, n_con, replace=False)) if n_con else []
    return picked, len(usable), len(lesional), len(controls)


def choose_features(subjects, selected, mode):
    per_subject = [subjects[s]["features"]["lh"] for s in selected]
    union = set().union(*per_subject)
    common = set.intersection(*per_subject) if per_subject else set()
    for s in selected:
        lh, rh = subjects[s]["features"]["lh"], subjects[s]["features"]["rh"]
        if lh != rh:
            warn(f"{s}: lh and rh have different features (only in lh: {sorted(lh - rh)[:3]}, only in rh: {sorted(rh - lh)[:3]})")
    partial = sorted(union - common)
    if partial:
        warn(f"{len(partial)} features exist only in some subjects (MELD loads them as zeros where missing): {partial[:8]}")
    return sorted(common if mode == "common" else union)


# the 34 input features of the released MELD Graph models (scripts/config_files/*_dcp/fold_00/s_0.py)
MELD_STANDARD_FEATURES = [
    f"{prefix}.on_lh.{name}.mgh"
    for prefix in (".combat", ".inter_z.intra_z.combat", ".inter_z.asym.intra_z.combat")
    for name in (
        "pial.K_filtered.sm20", "thickness.sm3", "thickness_regression.sm3", "w-g.pct.sm3", "sulc.sm3", "curv.sm3",
        "gm_FLAIR_0.75.sm3", "gm_FLAIR_0.5.sm3", "gm_FLAIR_0.25.sm3", "gm_FLAIR_0.sm3", "wm_FLAIR_0.5.sm3", "wm_FLAIR_1.sm3",
    )
    if not (prefix != ".combat" and name == "thickness.sm3")
]


def preprocessing_stage(feature):
    """MELD preprocessing stage a feature name belongs to (see data_preprocessing.Feature)."""
    if ".asym." in feature:
        return "asymmetry (inter_z.asym.intra_z)"
    if ".inter_z." in feature or ".intra_z." in feature:
        return "normalised (inter_z.intra_z)"
    if feature.startswith(".combat."):
        return "harmonised (combat)"
    if ".sm" in feature:
        return "smoothed only"
    return "raw (not preprocessed)"


def report_preprocessing(features):
    stages = {}
    for feat in features:
        stages.setdefault(preprocessing_stage(feat), []).append(feat)
    print("  preprocessing stage of the features:")
    for stage, feats in stages.items():
        print(f"    {stage}: {len(feats)}")
        for feat in feats:
            print(f"        {feat}")
    missing_std = [f for f in MELD_STANDARD_FEATURES if f not in features]
    if not missing_std:
        print("  all 34 standard MELD Graph features are present")
    else:
        print(f"  {34 - len(missing_std)} of the 34 standard MELD Graph features are present; missing: {missing_std}")
    if "raw (not preprocessed)" in stages or "smoothed only" in stages:
        warn("some features are not harmonised/normalised (no '.combat.' / '.inter_z.' prefix) - "
             "check that these are the files from the MELD preprocessing step")
    if not any(s.startswith(("harmonised", "normalised", "asymmetry")) for s in stages):
        warn("no preprocessed features (.combat. / .inter_z. / .asym.) found: these HDF5s look like raw feature files")


def compute_zscore(subjects, selected, features, out_file):
    from meld_graph.meld_cohort import MeldCohort

    cortex = MeldCohort().cortex_mask  # needs <meld_params>/fsaverage_sym/label/lh.cortex.label
    acc = {feat: [0, 0.0, 0.0] for feat in features}
    for subj in selected:
        rec = subjects[subj]
        with h5py.File(rec["file"], "r") as f:
            base = f[rec["site"]][rec["scanner"]][rec["group"]][subj]
            for hemi in ("lh", "rh"):
                if hemi not in base:
                    continue
                for feat in features:
                    if feat not in base[hemi]:
                        continue
                    vals = base[hemi][feat][:][cortex].astype(np.float64)
                    if not np.any(vals):  # all zeros = feature missing for this subject (e.g. no FLAIR)
                        continue
                    acc[feat][0] += vals.size
                    acc[feat][1] += vals.sum()
                    acc[feat][2] += np.square(vals).sum()
    params = {}
    for feat, (n, s, ss) in acc.items():
        if n == 0:
            warn(f"feature {feat} is all zeros in the selected subjects; z-scoring is skipped for it")
            params[feat] = {"mean": 0.0, "std": 1.0}
            continue
        mean = s / n
        std = float(np.sqrt(max(ss / n - mean ** 2, 0.0)))
        params[feat] = {"mean": float(mean), "std": std if std > 0 else 1.0}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(params, fp, indent=1)
    return params


# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Check your MELD data and run a short test training")
    parser.add_argument("--hdf5", nargs="+", required=True, help="your HDF5 files")
    parser.add_argument("--split_csv", required=True, help="CSV with columns subject_id, split (trainval/test)")
    parser.add_argument("--demographics_csv", required=True, help="CSV with ID, Harmo code, Group (, Scanner)")
    parser.add_argument("--check_only", action="store_true", help="only check, do not train")
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--features", choices=["common", "all"], default="common")
    parser.add_argument("--write_zscore", default=None, help="write z-score JSON from all usable trainval subjects and stop")
    args = parser.parse_args()

    if not os.path.isdir("data/icospheres"):
        sys.exit("Run this from the meld_graph repository root (data/icospheres not found).")

    print(f"meld_graph looks for data in BASE_PATH = {BASE_PATH}\n")
    print("== HDF5 files")
    subjects, file_root, links = check_hdf5_files(args.hdf5)
    n_pat = sum(r["group"] == "patient" for r in subjects.values())
    print(f"  {len(subjects)} subjects ({n_pat} patients, {len(subjects) - n_pat} controls) "
          f"in sites {sorted({r['site'] for r in subjects.values()})}; file pattern: {file_root}")
    bad = {s: r["bad"] for s, r in subjects.items() if r["bad"]}
    if bad:
        error(f"{len(bad)} subjects have structural problems, e.g. " +
              "; ".join(f"{s}: {', '.join(p[:2])}" for s, p in list(bad.items())[:5]))

    print("== demographics CSV")
    demo_ok = check_demographics(args.demographics_csv, subjects) or set()
    print(f"  {len(demo_ok)} HDF5 subjects have matching demographics")

    print("== split CSV")
    split_df = read_split(args.split_csv)
    if split_df is not None:
        counts = split_df["split"].astype(str).value_counts().to_dict()
        in_hdf5 = split_df["subject_id"].isin(list(subjects)).sum()
        print(f"  {len(split_df)} rows {counts}; {in_hdf5} of them are in the HDF5 files")
        missing = split_df.loc[~split_df["subject_id"].isin(list(subjects)), "subject_id"].tolist()
        if missing:
            warn(f"{len(missing)} split CSV subjects are not in any HDF5 file (ignored), e.g. {missing[:5]}")

    selected, features = [], []
    if split_df is not None and subjects:
        print("== test-run subset")
        selected, n_usable, n_les, n_con = select_subjects(subjects, demo_ok, split_df, args.n_subjects)
        n_sel_les = sum(subjects[s]["group"] == "patient" for s in selected)
        print(f"  usable trainval subjects: {n_usable} ({n_les} lesional patients, {n_con} controls); "
              f"selected {len(selected)} ({n_sel_les} lesional, {len(selected) - n_sel_les} controls)")
        if n_sel_les < 2:
            error("need at least 2 lesional patients (with a non-empty lesion mask) in the trainval split to train")
        if selected:
            features = choose_features(subjects, selected, args.features)
            print(f"  {len(features)} features ({args.features})")
            report_preprocessing(features)
            if not features:
                error("no features found")

    print()
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        sys.exit(f"\n{len(errors)} problem(s) found - fix them and run again.")
    print("\nAll checks passed.")
    if args.write_zscore:
        all_trainval, n_usable, _, _ = select_subjects(subjects, demo_ok, split_df, n_subjects=10 ** 9)
        all_features = choose_features(subjects, all_trainval, args.features)
        print(f"Computing z-score parameters for {len(all_features)} features from all {len(all_trainval)} "
              f"usable trainval subjects -> {args.write_zscore}")
        compute_zscore(subjects, all_trainval, all_features, os.path.abspath(args.write_zscore))
        print("Features (for the config):")
        for feat in all_features:
            print(f'    "{feat}",')
        return
    if args.check_only:
        return

    # ---------------- prepare and train ----------------
    for src, expected in links:
        os.makedirs(os.path.dirname(expected), exist_ok=True)
        os.symlink(src, expected)
        print(f"Linked {expected} -> {src}")
    shutil.copy(args.demographics_csv, DEMOGRAPHIC_FEATURES_FILE)
    pd.DataFrame({"subject_id": selected, "split": "trainval"}).to_csv(SPLIT_SUBSET_CSV, index=False)
    zscore_file = os.path.join(EXPERIMENT_PATH, os.path.dirname(EXP_NAME), "feature_means_test_run.json")
    print(f"Computing z-score parameters from {len(selected)} subjects -> {zscore_file}")
    compute_zscore(subjects, selected, features, zscore_file)

    import meld_graph.experiment

    ref = load_config(REFERENCE_CONFIG)
    network_parameters = copy.deepcopy(ref.network_parameters)
    data_parameters = copy.deepcopy(ref.data_parameters)
    network_parameters["name"] = EXP_NAME
    tp = network_parameters["training_parameters"]
    tp["num_epochs"] = args.epochs + 1  # the training loop starts at epoch 1
    tp["max_epochs_lr_decay"] = args.epochs + 1
    tp["batch_size"] = args.batch_size
    tp["init_weights"] = None
    sites = sorted({subjects[s]["site"] for s in selected})
    data_parameters.update(
        {
            "hdf5_file_root": file_root,
            "site_codes": sites,
            "scanners": ["15T", "3T", "XT"],
            "group": "both",
            "dataset": SPLIT_SUBSET_CSV,
            "number_of_folds": 2,
            "fold_n": 0,
            "features": features,
            "features_to_replace_with_0": [],
            "subject_features_to_exclude": [],
        }
    )
    data_parameters["preprocessing_parameters"]["zscore"] = zscore_file

    exp = meld_graph.experiment.Experiment(network_parameters, data_parameters, verbose=logging.INFO)
    train_ids, val_ids, _ = exp.get_train_val_test_ids()
    print(f"Train subjects: {len(train_ids)}, val subjects: {len(val_ids)}")
    if len(train_ids) == 0 or len(val_ids) == 0:
        sys.exit("meld_graph found no train or val subjects - see the log above.")
    exp.train()

    out_dir = exp.experiment_path
    expected = ["best_model.pt", "final_model.pt", "train_scores.csv", "val_scores.csv"]
    missing = [f for f in expected if not os.path.isfile(os.path.join(out_dir, f))]
    if missing:
        sys.exit(f"Training ran but these outputs are missing in {out_dir}: {missing}")
    print(f"\nTEST RUN PASSED. Outputs in {out_dir}")
    print(pd.read_csv(os.path.join(out_dir, "val_scores.csv")).tail().to_string())
    print("\nSettings for a full run (copy into scripts/config_files/my_train_config.py):")
    print(f'    "hdf5_file_root": "{file_root}",')
    print(f'    "site_codes": {sites},')
    print(f'    "features": <the {len(features)} features listed in {os.path.join(out_dir, "data_parameters.json")}>')


if __name__ == "__main__":
    main()
