# Config for training the base MELD Graph U-Net (MoNetUnet) on your own data.
# Based on scripts/config_files/24-08-01_MRIN_dcp/fold_00/s_0.py.
#
# Run from the repository root:
#     python scripts/classifier/train.py --config_file scripts/config_files/my_train_config.py
#
# Paths below that are not absolute are resolved against meld_config.ini:
#     BASE_PATH        = <meld_data_path>/output/preprocessed_surf_data   (HDF5 site folders + split CSV)
#     EXPERIMENT_PATH  = <meld_data_path>/models                          (trained models are saved here)
#     MELD_PARAMS_PATH = <meld_data_path>/meld_params                     (fsaverage_sym, z-score file via ../)
#
# Lines marked  # <-- SET  depend on your data.

network_parameters = {
    # experiment folder: EXPERIMENT_PATH/<name>/fold_<fold_n>
    "name": "H5_full_run_2epochs/s_0",
    # model class from meld_graph/models.py: "MoNetUnet" (graph U-Net) or "MoNet" (convolutions only)
    "network_type": "MoNetUnet",
    "model_parameters": {
        # U-Net: one list per resolution level (7 = 163842 vertices ... 1 = 42 vertices),
        # each list = output channels of the convolutions in that block. Decoder mirrors this.
        "layer_sizes": [
            [32, 32, 32],
            [32, 32, 32],
            [64, 64, 64],
            [64, 64, 64],
            [128, 128, 128],
            [128, 128, 128],
            [256, 256, 256],
        ],
        "activation_fn": "leaky_relu",  # "relu" or "leaky_relu"
        "conv_type": "SpiralConv",  # "SpiralConv" or "GMMConv"
        "spiral_len": 7,  # vertices per spiral kernel (centre + first ring)
        "dim": 2,  # GMMConv only
        "kernel_size": 3,  # GMMConv only
        "norm": None,  # None or "instance"
        "distance_head": False,  # separate linear head for distance regression
    },
    "training_parameters": {
        # NOTE: the training loop starts at epoch 1, so num_epochs = N trains N-1 epochs. 3 -> 2 epochs.
        "num_epochs": 3,
        "max_patience": 1000,  # stop if stopping_metric does not improve for this many epochs
        "stopping_metric": {"name": "loss", "sign": 1},  # best model = lowest val loss
        "metric_smoothing": False,
        "optimiser": "sgd",  # "sgd" or "adam"
        "optimiser_parameters": {"lr": 0.0001, "momentum": 0.99, "nesterov": True},
        "lr_decay": 0.9,  # lr * (1 - epoch / max_epochs_lr_decay) ** lr_decay ; 0 = off
        "max_epochs_lr_decay": 3,
        # hemispheres per batch. MELD's standard is 8 (GPU); 2 fits in RAM on CPU (8 was killed out of memory)
        "batch_size": 2,
        "shuffle_each_epoch": True,
        # oversampling: every epoch = all lesional hemispheres + 2x as many random ones
        "oversampling": True,
        # losses and their weights
        "loss_dictionary": {
            "cross_entropy": {"weight": 1},
            "dice": {"class_weights": [0.0, 1.0], "weight": 1},
            # predict geodesic distance to the lesion (needs lesion masks)
            "distance_regression": {"loss": "mae", "weigh_by_gt": True, "weight": 1},
            # lesional vs non-lesional hemisphere, from the bottleneck
            "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
        },
        # extra losses on coarser decoder levels, with these weights
        "deep_supervision": {
            "levels": [6, 5, 4, 3, 2, 1],
            "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
        },
        "metrics": [
            "dice_lesion", "dice_nonlesion", "precision", "recall",
            "tp", "fp", "fn", "auroc", "cl_precision", "cl_recall", "sub_auroc",
        ],
        # None = train from scratch. Use None, not False.
        # Or a checkpoint relative to EXPERIMENT_PATH, e.g. "23-10-30_LVHZ_dcp/fold_00/best_model.pt".
        # Note: on main, a path containing "fold_all" (the 5-fold ensemble) is loaded as an ensemble.
        "init_weights": None,
        "start_epoch": 0,  # >0 resumes from EXPERIMENT_PATH/<name>/fold_XX/epoch_<start_epoch>.pt
    },
}

data_parameters = {
    # ---- where the data is ----
    # HDF5 file name pattern inside BASE_PATH/MELD_<site_code>/
    "hdf5_file_root": "{site_code}_{group}_featurematrix_combat.hdf5",
    # split CSV in BASE_PATH, columns: subject_id, split ("trainval" or "test")
    # copy your split CSV to BASE_PATH under this name (or put an absolute path here)
    "dataset": "split_testbatch.csv",
    # sites to use: must match MELD_<site> folders, the HDF5 top-level group, and "Harmo code" in demographics
    "site_codes": ["H5"],
    # scanners to keep, from the demographics "Scanner" column. Values other than 3T/1.5T (or no column) become "XT".
    "scanners": ["15T", "3T", "XT"],
    "group": "both",  # "patient", "control" or "both"

    # ---- subject filters / splits ----
    "subject_features_to_exclude": [],  # e.g. ["FLAIR"] = only subjects that have FLAIR features
    "number_of_folds": 5,  # trainval subjects are split into this many folds
    "fold_n": 0,  # which fold is the validation set (0 .. number_of_folds-1)
    "subsample_cohort_fraction": False,

    # ---- input features ----
    # exact dataset names inside your HDF5s (lh and rh both use ".on_lh." names).
    # A name that is missing in a subject's HDF5 is silently loaded as zeros.
    # the 16 T1 features in your data (no FLAIR); same order as the standard MELD list
    "features": [
        ".combat.on_lh.pial.K_filtered.sm20.mgh",
        ".combat.on_lh.thickness.sm3.mgh",
        ".combat.on_lh.thickness_regression.sm3.mgh",
        ".combat.on_lh.w-g.pct.sm3.mgh",
        ".combat.on_lh.sulc.sm3.mgh",
        ".combat.on_lh.curv.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.curv.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.curv.sm3.mgh",
    ],
    "features_to_exclude": [],  # only used if "features" is not given
    "features_to_replace_with_0": [],  # features loaded as zeros on purpose

    # ---- preprocessing ----
    "preprocessing_parameters": {
        "scaling": None,  # not implemented anymore, keep None
        # JSON with {"<feature>": {"mean": .., "std": ..}} for every feature above.
        # Relative paths are joined to MELD_PARAMS_PATH; an absolute path is used as is.
        # computed from all your trainval subjects with test_run_own_data.py --write_zscore
        "zscore": "/home/affan/MSCS/data/models/feature_means_H5_trainval.json",
    },
    "combine_hemis": None,  # None = each hemisphere is one sample
    "distance_mask_medial_wall": True,  # medial wall = max distance (300) in distance maps
    "smooth_labels": False,
    "icosphere_parameters": {"distance_type": "exact"},

    # ---- augmentation (training only); files are relative to the repo root ----
    "augment_data": {
        "spinning": {"p": 0.2, "file": "data/spinning/spinning_ico7_10.npy"},
        "warping": {"p": 0.2, "file": "data/warping/warping_ico7_10.npy"},
        "flipping": {"p": 0.5, "file": "data/flipping/flipping_ico7_3.npy"},
        "noise": {"p": 0.15},
        "blur": {"p": 0.2},  # currently does nothing
        "brightness": {"p": 0.15},
        "contrast": {"p": 0.15},
        "low_res": {"p": 0.25},  # currently does nothing
        "gamma": {"p": 0.15},
        "augment_lesion": {"p": 0.0},
        "extend_lesion": {"p": 0.0},
    },

    # ---- alternative tasks / synthetic data: leave off for normal training ----
    "lobes": False,  # True = predict frontal lobe instead of lesions (sanity-check task)
    "lesion_bias": 0,
    "object_detection": False,
    "synth_on_the_fly": False,
    "synthetic_data": {
        "run_synthetic": False,
        "n_subs": 1000,
        "use_controls": True,
        "radius": 2,
        "n_subtypes": 25,
        "jitter_factor": 2,
        "bias": 1,
        "proportion_features_abnormal": 0.2,
        "proportion_hemispheres_lesional": 0.9,
        "smooth_lesion": False,
    },
}
