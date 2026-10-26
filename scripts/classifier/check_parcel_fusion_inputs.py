"""
Pre-flight checks for parcel-fusion training (MoNetUnetParcelFusion). Changes nothing.

Run from the repository root with the meld_graph environment active:
    python scripts/classifier/check_parcel_fusion_inputs.py --config_file scripts/config_files/parcel_fusion_config.py

Checks:
  1. the released model files and its z-score file (loading the config already does this)
  2. the Desikan-Killiany atlas: 36 parcels, their order (= the order your vectors must follow), vertices per parcel
  3. one parcel-vector .npy per subject used in training: shape (2, 36[, D]), no NaN/inf, value ranges
  4. the pretrained checkpoint: readable, ensemble members, the chosen fold model exists
  5. overlap between your subjects and the subjects the released MELD model was trained / tested on
"""

import argparse
import os
import sys

import h5py
import nibabel as nb
import numpy as np
import pandas as pd

from meld_graph.paths import BASE_PATH, DK_ATLAS_FILE, EXPERIMENT_PATH, MELD_PARAMS_PATH, load_config


def hdf5_subjects(dp):
    """Subjects in your HDF5 files, as meld_graph will find them."""
    found = {}
    for site in dp["site_codes"]:
        for group in ("patient", "control"):
            path = os.path.join(BASE_PATH, f"MELD_{site}", dp["hdf5_file_root"].format(site_code=site, group=group))
            if not os.path.isfile(path):
                continue
            with h5py.File(path, "r") as f:
                if site not in f:
                    continue
                for scanner in f[site]:
                    if group in f[site][scanner]:
                        for subj in f[site][scanner][group]:
                            found[subj] = group
    return found


def main():
    parser = argparse.ArgumentParser(description="Check inputs for parcel-fusion training")
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args()

    errors, warnings = [], []
    print(f"== config {args.config_file}")
    cfg = load_config(args.config_file)  # raises if the released model files or its z-score file are missing
    dp, net = cfg.data_parameters, cfg.network_parameters
    tp = net["training_parameters"]
    pf = dp["parcel_fusion"]
    print(f"  model {net['network_type']}, start from {tp['init_weights']} (fold model {tp['init_weights_ensemble_member']}), "
          f"freeze_encoder={tp['freeze_encoder']}")
    print(f"  {len(dp['features'])} input features, z-score file {dp['preprocessing_parameters']['zscore']}")
    print(f"  parcel vectors: {pf['parcel_dir']}, n_parcels={pf['n_parcels']}, parcel_dim={pf['parcel_dim']}")

    # ---- 1. z-score file covers every feature
    import json

    with open(dp["preprocessing_parameters"]["zscore"]) as fp:
        z = json.load(fp)
    missing_z = [f for f in dp["features"] if f not in z]
    if missing_z:
        errors.append(f"z-score file has no entry for {missing_z}")

    # ---- 2. DK atlas
    print("\n== Desikan-Killiany atlas (your vectors must use this parcel order)")
    annot = pf.get("annot_file") or os.path.join(MELD_PARAMS_PATH, DK_ATLAS_FILE)
    if not os.path.isfile(annot):
        errors.append(f"DK annotation not found: {annot}")
    else:
        labels, _, names = nb.freesurfer.read_annot(annot)
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        counts = np.bincount(labels[labels >= 0], minlength=len(names))
        if len(names) != pf["n_parcels"]:
            errors.append(f"{annot} has {len(names)} parcels, config says n_parcels={pf['n_parcels']}")
        for i, name in enumerate(names):
            print(f"  {i:2d} {name:28s} {counts[i]:6d} vertices" + ("   (no vertices: value is unused)" if counts[i] == 0 else ""))
        n_none = int((labels < 0).sum())
        if n_none:
            print(f"  -- {n_none} vertices have no parcel and get a zero vector")

    # ---- 3. parcel vector files for the subjects meld_graph will use
    print("\n== parcel vector files")
    split = pd.read_csv(os.path.join(BASE_PATH, dp["dataset"]))
    split["subject_id"] = split["subject_id"].astype(str)
    in_hdf5 = hdf5_subjects(dp)
    used = split[split["subject_id"].isin(list(in_hdf5))]
    print(f"  {len(used)} subjects in both the split CSV and your HDF5 files "
          f"({(used['split'] == 'trainval').sum()} trainval, {(used['split'] == 'test').sum()} test)")
    expected_shape = (2, pf["n_parcels"], pf["parcel_dim"])
    missing, bad, arrays = [], [], []
    for subj in used["subject_id"]:
        path = os.path.join(pf["parcel_dir"], f"{subj}.npy")
        if not os.path.isfile(path):
            missing.append(subj)
            continue
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape != expected_shape:
            bad.append(f"{subj}: shape {arr.shape}, expected {expected_shape}")
        elif not np.all(np.isfinite(arr)):
            bad.append(f"{subj}: contains NaN or inf")
        else:
            arrays.append(arr)
    if missing:
        msg = f"{len(missing)} subjects have no parcel vector file, e.g. {missing[:5]}"
        (warnings if pf.get("missing_subject") == "zeros" else errors).append(msg)
    if bad:
        errors.append(f"{len(bad)} parcel vector files are invalid, e.g. {bad[:5]}")
    if arrays:
        stack = np.stack(arrays)  # (subjects, 2, parcels, D)
        print(f"  {len(arrays)} valid files, shape {expected_shape}")
        for d in range(stack.shape[-1]):
            v = stack[..., d]
            print(f"  dim {d}: min {v.min():.3g}, max {v.max():.3g}, mean {v.mean():.3g}, std {v.std():.3g}")
        if np.abs(stack).max() > 10:
            warnings.append("parcel vector values go above 10 in magnitude; consider scaling them (e.g. z-scoring) "
                            "so they are on a similar scale to the network's features")
        if np.all(stack[:, 0] == stack[:, 1]):
            warnings.append("left and right hemisphere vectors are identical in every file - check the (2, ...) axis")

    # ---- 4. pretrained checkpoint
    print("\n== pretrained checkpoint")
    import torch

    ckpt = os.path.join(EXPERIMENT_PATH, tp["init_weights"])
    state = torch.load(ckpt, map_location="cpu")
    members = sorted({int(k.split(".")[1]) for k in state if k.startswith("models.")})
    if members:
        print(f"  {ckpt}: ensemble of {len(members)} fold models")
        if tp["init_weights_ensemble_member"] not in members:
            errors.append(f"init_weights_ensemble_member={tp['init_weights_ensemble_member']} not in {members}")
    else:
        print(f"  {ckpt}: single model, {len(state)} tensors")
    first_layer = [k for k in state if k.endswith("encoder_conv_layers.0.0.layer.weight")]
    if first_layer:
        n_in = state[first_layer[0]].shape[1] // net["model_parameters"]["spiral_len"]
        print(f"  first layer expects {n_in} input features (config gives {len(dp['features'])})")
        if n_in != len(dp["features"]):
            errors.append(f"pretrained model expects {n_in} features, config has {len(dp['features'])}")

    # ---- 5. overlap with the released model's subjects
    print("\n== overlap with the released MELD model's subjects")
    pre = cfg.PRETRAINED_DATA_PARAMETERS
    pre_trained_on = set(map(str, pre.get("train_ids", []))) | set(map(str, pre.get("val_ids", [])))
    pre_test = set(map(str, pre.get("test_ids", [])))
    if not pre_trained_on and not pre_test:
        print("  the released model's data_parameters.json lists no subject IDs - overlap unknown")
    for split_name in ("trainval", "test"):
        ids = set(used.loc[used["split"] == split_name, "subject_id"])
        n_tr, n_te = len(ids & pre_trained_on), len(ids & pre_test)
        print(f"  your {split_name} subjects ({len(ids)}): {n_tr} were in the released model's training set, "
              f"{n_te} in its test set")
        if split_name == "test" and n_tr:
            warnings.append(f"{n_tr} of your test subjects were used to train the released MELD model: results on them "
                            f"with pretrained weights are not an independent test, e.g. {sorted(ids & pre_trained_on)[:5]}")

    print()
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        sys.exit(f"\n{len(errors)} problem(s) found.")
    print("All checks passed.")


if __name__ == "__main__":
    main()
