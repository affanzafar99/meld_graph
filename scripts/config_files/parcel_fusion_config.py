# Parcel-fusion fine-tuning on your H5 data, starting from the released MELD Graph model.
#
#   - model: MoNetUnetParcelFusion = MoNetUnet + gated fusion of a per-subject Desikan-Killiany parcel vector
#     after every decoder block (see meld_graph/models.py)
#   - weights: one fold model of the released ensemble (23-10-30_LVHZ_dcp/fold_all), encoder frozen
#   - architecture, losses, deep supervision, input features and z-score file are read from the released
#     model's own JSON files below, so the pretrained weights always fit
#
# Run from the repository root:
#     python scripts/classifier/check_parcel_fusion_inputs.py --config_file scripts/config_files/parcel_fusion_config.py
#     python scripts/classifier/train_own_data.py --config_file scripts/config_files/parcel_fusion_config.py \
#         --demographics_csv /home/affan/MSCS/data/my_data/demographics_testbatch_meld.csv

import glob
import json
import os

import numpy as np

from meld_graph.paths import EXPERIMENT_PATH, MELD_PARAMS_PATH

# ---------------------------------------------------------------------------------------------
# settings you are most likely to change
# ---------------------------------------------------------------------------------------------
EXPERIMENT_NAME = "H5_parcel_fusion/s_0"  # saved to EXPERIMENT_PATH/<name>/fold_<fold_n>
PRETRAINED = "23-10-30_LVHZ_dcp/fold_all"  # released MELD model, relative to EXPERIMENT_PATH
PRETRAINED_FOLD_MODEL = 0  # which of the 5 fold models in the ensemble to start from (0-4)
PARCEL_DIR = "/home/affan/MSCS/data/parcel_vectors"  # one <subject_id>.npy per subject, shape (2, 36[, D]), order lh, rh
N_EPOCHS = 2
BATCH_SIZE = 2  # 2 fits in RAM on CPU; MELD's standard is 8 on GPU
FUSION_LEVELS = None  # decoder levels to fuse at; None = all (2-7). e.g. [2, 3, 4, 5, 6] skips the 163842-vertex level


# ---------------------------------------------------------------------------------------------
# read the released model's parameters
# ---------------------------------------------------------------------------------------------
_pretrained_dir = os.path.join(EXPERIMENT_PATH, PRETRAINED)
for _name in ("data_parameters.json", "network_parameters.json", "best_model.pt"):
    if not os.path.isfile(os.path.join(_pretrained_dir, _name)):
        raise FileNotFoundError(f"Released MELD model file not found: {os.path.join(_pretrained_dir, _name)}")
with open(os.path.join(_pretrained_dir, "data_parameters.json")) as _f:
    PRETRAINED_DATA_PARAMETERS = json.load(_f)
with open(os.path.join(_pretrained_dir, "network_parameters.json")) as _f:
    PRETRAINED_NETWORK_PARAMETERS = json.load(_f)

# z-score file the released model was trained with (relative paths are relative to MELD_PARAMS_PATH)
_zscore = os.path.join(MELD_PARAMS_PATH, PRETRAINED_DATA_PARAMETERS["preprocessing_parameters"]["zscore"])
if not os.path.isfile(_zscore):
    raise FileNotFoundError(
        f"z-score file of the released model not found: {os.path.normpath(_zscore)}. "
        f"It is listed in {os.path.join(_pretrained_dir, 'data_parameters.json')}."
    )


def _detect_parcel_dim(parcel_dir):
    files = sorted(glob.glob(os.path.join(parcel_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy parcel vector files in {parcel_dir}")
    arr = np.load(files[0], mmap_mode="r")
    return 1 if arr.ndim == 2 else int(arr.shape[-1])


_pre_training = PRETRAINED_NETWORK_PARAMETERS["training_parameters"]

network_parameters = {
    "name": EXPERIMENT_NAME,
    "network_type": "MoNetUnetParcelFusion",
    # identical architecture to the released model
    "model_parameters": PRETRAINED_NETWORK_PARAMETERS["model_parameters"],
    "training_parameters": {
        # same losses / heads / deep supervision as the released model, so every pretrained weight is used
        "loss_dictionary": _pre_training["loss_dictionary"],
        "deep_supervision": _pre_training.get("deep_supervision", {"levels": [], "weight": []}),
        "metrics": _pre_training.get("metrics", ["dice_lesion", "dice_nonlesion", "precision", "recall", "tp", "fp", "fn", "auroc"]),
        "optimiser": _pre_training.get("optimiser", "sgd"),
        "optimiser_parameters": _pre_training.get("optimiser_parameters", {"lr": 0.0001, "momentum": 0.99, "nesterov": True}),
        "lr_decay": _pre_training.get("lr_decay", 0.9),
        "stopping_metric": _pre_training.get("stopping_metric", {"name": "loss", "sign": 1}),
        "metric_smoothing": False,
        "oversampling": True,
        "shuffle_each_epoch": True,
        # the training loop starts at epoch 1, so num_epochs = N_EPOCHS + 1 trains N_EPOCHS epochs
        "num_epochs": N_EPOCHS + 1,
        "max_epochs_lr_decay": N_EPOCHS + 1,
        "max_patience": 1000,
        "batch_size": BATCH_SIZE,
        "start_epoch": 0,
        # start from one fold model of the released ensemble, freeze its encoder
        "init_weights": f"{PRETRAINED}/best_model.pt",
        "init_weights_ensemble_member": PRETRAINED_FOLD_MODEL,
        "freeze_encoder": True,
    },
}

data_parameters = {
    # ---- your data (same as my_train_config.py)
    "hdf5_file_root": "{site_code}_{group}_featurematrix_combat.hdf5",
    "dataset": "split_testbatch.csv",  # in BASE_PATH
    "site_codes": ["H5"],
    "scanners": ["15T", "3T", "XT"],
    "group": "both",
    "subject_features_to_exclude": [],
    "number_of_folds": 5,
    "fold_n": 0,
    "subsample_cohort_fraction": False,
    # ---- inputs exactly as the released model expects: its 34 features in its order, its z-score file.
    # Your files have no FLAIR features; MELD loads missing features as zeros, as it does for patients without FLAIR.
    "features": PRETRAINED_DATA_PARAMETERS["features"],
    "features_to_exclude": [],
    "features_to_replace_with_0": [],
    "preprocessing_parameters": {"scaling": None, "zscore": os.path.normpath(_zscore)},
    "combine_hemis": None,
    "distance_mask_medial_wall": True,
    "smooth_labels": False,
    "icosphere_parameters": {"distance_type": "exact"},
    "augment_data": {
        "spinning": {"p": 0.2, "file": "data/spinning/spinning_ico7_10.npy"},
        "warping": {"p": 0.2, "file": "data/warping/warping_ico7_10.npy"},
        "flipping": {"p": 0.5, "file": "data/flipping/flipping_ico7_3.npy"},
        "noise": {"p": 0.15},
        "blur": {"p": 0.2},
        "brightness": {"p": 0.15},
        "contrast": {"p": 0.15},
        "low_res": {"p": 0.25},
        "gamma": {"p": 0.15},
        "augment_lesion": {"p": 0.0},
        "extend_lesion": {"p": 0.0},
    },
    # ---- per-subject DK parcel vectors (see meld_graph/parcels.py)
    "parcel_fusion": {
        "parcel_dir": PARCEL_DIR,
        "n_parcels": 36,  # FreeSurfer aparc colour-table order of fsaverage_sym/label/lh.aparc.annot
        "parcel_dim": _detect_parcel_dim(PARCEL_DIR),  # numbers per parcel, read from your first .npy file
        "fusion_levels": FUSION_LEVELS,
        "missing_subject": "error",  # or "zeros"
        "annot_file": None,  # default: MELD_PARAMS_PATH/fsaverage_sym/label/lh.aparc.annot
    },
    # ---- off
    "lobes": False,
    "lesion_bias": 0,
    "object_detection": False,
    "synth_on_the_fly": False,
    "synthetic_data": {"run_synthetic": False},
}
