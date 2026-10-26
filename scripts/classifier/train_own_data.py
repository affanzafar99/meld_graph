"""
Same as scripts/classifier/train.py, but first puts your demographics CSV where meld_graph reads it.

meld_graph reads demographics from meld_graph.paths.DEMOGRAPHIC_FEATURES_FILE, a temporary path
(/tmp/demographics_file_<process id>.csv) that only the prediction pipeline fills in. This script copies your
file there for this training run, so no meld_graph code has to change.

Run from the repository root with the meld_graph environment active:
    python scripts/classifier/train_own_data.py \
        --config_file scripts/config_files/my_train_config.py \
        --demographics_csv /home/affan/MSCS/data/my_data/demographics_testbatch_meld.csv
"""

import argparse
import logging
import os
import shutil
import sys

import meld_graph.experiment
from meld_graph.paths import DEMOGRAPHIC_FEATURES_FILE, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using config_file, with your demographics CSV")
    parser.add_argument("--config_file", required=True, help="path to experiment config .py")
    parser.add_argument("--demographics_csv", required=True, help="CSV with ID, Harmo code, Group (, Scanner)")
    parser.add_argument("--wandb_logging", action="store_true", help="enable wandb logging.")
    args = parser.parse_args()

    if not os.path.isdir("data/icospheres"):
        sys.exit("Run this from the meld_graph repository root (data/icospheres not found).")
    shutil.copy(args.demographics_csv, DEMOGRAPHIC_FEATURES_FILE)
    print(f"Demographics: {args.demographics_csv} -> {DEMOGRAPHIC_FEATURES_FILE}")

    config = load_config(args.config_file)
    exp = meld_graph.experiment.Experiment(config.network_parameters, config.data_parameters, verbose=logging.INFO)
    train_ids, val_ids, test_ids = exp.get_train_val_test_ids()
    print(f"Subjects - train: {len(train_ids)}, val: {len(val_ids)}, test (held out): {len(test_ids)}")
    if len(train_ids) == 0 or len(val_ids) == 0:
        sys.exit("No train or val subjects - check dataset, site_codes, scanners and demographics in the config.")
    exp.train(wandb_logging=args.wandb_logging)
    print(f"Finished. Outputs in {exp.experiment_path}")
