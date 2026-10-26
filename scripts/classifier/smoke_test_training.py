"""
Quick end-to-end training test on the MELD test data (site "TEST", downloaded by prepare_classifier.py).

Checks that the environment, data loading, model, training loop and checkpoint saving all work,
without touching your own data. Does not change any MELD code.

Run from the repository root, with the meld_graph environment active:
    python scripts/classifier/smoke_test_training.py            # 2 training epochs
    python scripts/classifier/smoke_test_training.py --epochs 1

Output: <meld_data_path>/models/smoke_test/s_0/fold_00/ (best_model.pt, train_scores.csv, val_scores.csv, train.log)
"""

import argparse
import copy
import json
import logging
import os
import sys

import pandas as pd

from meld_graph.paths import (
    BASE_PATH,
    DEFAULT_HDF5_FILE_ROOT,
    DEMOGRAPHIC_FEATURES_FILE,
    EXPERIMENT_PATH,
    MELD_PARAMS_PATH,
    load_config,
)
from meld_graph.meld_cohort import MeldCohort
from meld_graph.test.utils import create_test_demos
import meld_graph.experiment

REFERENCE_CONFIG = "scripts/config_files/24-08-01_MRIN_dcp/fold_00/s_0.py"
ZSCORE_FILE = "feature_means_nocombat.json"  # in meld_params, also used by meld_graph/test/test_dataset.py
DATASET_CSV = "/tmp/meld_smoke_test_dataset.csv"


def main():
    parser = argparse.ArgumentParser(description="Smoke-test MELD Graph training on the MELD_TEST data")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs (default 2)")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    if not os.path.isdir("data/icospheres"):
        sys.exit("Run this from the meld_graph repository root (data/icospheres not found).")
    test_dir = os.path.join(BASE_PATH, "MELD_TEST")
    if not os.path.isdir(test_dir):
        sys.exit(f"Test data not found at {test_dir}. Run ./meldgraph.sh prepare_classifier.py first.")

    # 1. demographics for the test subjects, written to the path MeldSubject reads (DEMOGRAPHIC_FEATURES_FILE)
    create_test_demos()
    demos = pd.read_csv(DEMOGRAPHIC_FEATURES_FILE)
    print(f"Demographics for {len(demos)} test subjects written to {DEMOGRAPHIC_FEATURES_FILE}")

    # 2. split file: all test subjects are train/val (none held out as test)
    pd.DataFrame({"subject_id": demos["ID"], "split": "trainval"}).to_csv(DATASET_CSV, index=False)

    # 3. features present in the test HDF5s that also have z-score parameters
    cohort = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=DATASET_CSV)
    available = cohort.get_features()
    with open(os.path.join(MELD_PARAMS_PATH, ZSCORE_FILE)) as f:
        z_params = json.load(f)
    features = [feat for feat in available if feat in z_params]
    skipped = [feat for feat in available if feat not in z_params]
    print(f"Using {len(features)} features" + (f" (skipping {len(skipped)} without z-score params)" if skipped else ""))
    if not features:
        sys.exit(f"No features in the test data match {ZSCORE_FILE}")

    # 4. config: the standard MELD config, pointed at the test data and shortened
    ref = load_config(REFERENCE_CONFIG)
    network_parameters = copy.deepcopy(ref.network_parameters)
    data_parameters = copy.deepcopy(ref.data_parameters)

    network_parameters["name"] = "smoke_test/s_0"
    tp = network_parameters["training_parameters"]
    # the training loop skips epoch 0 (it starts at start_epoch + 1), so add one
    tp["num_epochs"] = args.epochs + 1
    tp["max_epochs_lr_decay"] = args.epochs + 1
    tp["batch_size"] = args.batch_size
    tp["init_weights"] = None

    data_parameters.update(
        {
            "hdf5_file_root": DEFAULT_HDF5_FILE_ROOT,
            "site_codes": ["TEST"],
            "scanners": ["15T", "3T", "XT"],
            "dataset": DATASET_CSV,
            "number_of_folds": 2,
            "fold_n": 0,
            "features": features,
            "features_to_replace_with_0": [],
        }
    )
    data_parameters["preprocessing_parameters"]["zscore"] = ZSCORE_FILE

    # 5. train
    exp = meld_graph.experiment.Experiment(network_parameters, data_parameters, verbose=logging.INFO)
    train_ids, val_ids, _ = exp.get_train_val_test_ids()
    print(f"Train subjects: {len(train_ids)}, val subjects: {len(val_ids)}")
    if len(train_ids) == 0 or len(val_ids) == 0:
        sys.exit("No train or val subjects found - check the test data and demographics.")
    exp.train()

    # 6. check outputs
    out_dir = exp.experiment_path
    expected = ["best_model.pt", "train_scores.csv", "val_scores.csv", "data_parameters.json", "network_parameters.json"]
    missing = [f for f in expected if not os.path.isfile(os.path.join(out_dir, f))]
    print()
    if missing:
        print(f"Training ran but these outputs are missing in {out_dir}: {missing}")
        sys.exit(1)
    print(f"SMOKE TEST PASSED. Outputs in {out_dir}")
    print(pd.read_csv(os.path.join(out_dir, "val_scores.csv")).tail().to_string())


if __name__ == "__main__":
    main()
