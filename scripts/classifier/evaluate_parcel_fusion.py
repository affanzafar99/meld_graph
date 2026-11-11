"""
Evaluate a MELD Graph ensemble on test subjects the way MELD evaluates its released model, so models can be
compared on exactly the same subjects, input data and scoring.

Two modes:

  A) --experiment EXP                  your trained 5-fold experiment, e.g. parcel_fusion/run1
     The fold models are combined into EXP/fold_all (MELD's create_ensemble.py) and evaluated on the
     experiment's test subjects. A parcel fusion model gets each subject's parcel vectors, as in training.

  B) --ensemble DIR --reference EXP    an existing ensemble, e.g. the released MELD model
                                       23-10-30_LVHZ_dcp/fold_all (no fusion layers)
     Evaluated with MELD's own code on the subjects of the reference experiment, reading the same files
     with the same features and normalisation (the reference's data settings, without parcel vectors).
     The ensemble folder is only read; results go to EXPERIMENT_PATH/<--output>/fold_all/.

Steps (finished steps are reused on a rerun; --overwrite redoes everything):
  1. Ensemble: built (mode A; rebuilt automatically if a fold model is newer) or checked (mode B).
  2. Thresholds (--threshold two_threshold, MELD's default):
       --thresholds trainval (default): the ensemble predicts the trainval subjects and the lower threshold
         is chosen from their subject-level ROC curve with MELD's rule (calculate_thresholds.py: first peak
         of sensitivity + specificity after the largest jump); the upper threshold is 0.5.
       --thresholds released (mode B only): the ensemble's own results_<model>/two_thresholds.csv.
     With --threshold slope_threshold this step is skipped (MELD's per-subject slope rule).
  3. Test: the ensemble predicts the test subjects; MELD's Evaluator thresholds the predictions, removes
     clusters smaller than 100 vertices and scores each subject. In <output>/fold_all/results_<model>/:
       predictions.hdf5   vertex-wise predictions and clusters
       test_results.csv   per subject: detected, TP/FP clusters, Dice, ...
       roc_auc.pickle     subject-level ROC over thresholds 0..1 (sensitivity / specificity counts, AUC)
       sub_aucs.pickle    per patient: vertex-level AUC of the prediction vs. the lesion border zone
  4. Summary: printed and written to summary.json in the same folder.

Definitions (as in MELD Graph):
  patient detected   a predicted cluster overlaps the lesion or the 20 mm border zone around it
  sensitivity        detected patients / patients
  specificity        controls without any predicted cluster / controls

Run from the repository root with the meld_graph environment active (a GPU makes it much faster):
    python scripts/classifier/evaluate_parcel_fusion.py --experiment parcel_fusion/run1 \
        --demographics_csv /path/to/demographics_all_meld.csv
    python scripts/classifier/evaluate_parcel_fusion.py --ensemble 23-10-30_LVHZ_dcp/fold_all \
        --reference parcel_fusion/run1 --output meld_released \
        --demographics_csv /path/to/demographics_all_meld.csv
Compare the results with scripts/classifier/compare_parcel_fusion.py.
"""

import argparse
import copy
import importlib.util
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

ROC_THRESHOLDS = np.linspace(0, 1, 101)
UPPER_THRESHOLD = 0.5  # MELD's two-threshold rule: the high-confidence threshold is fixed at 0.5


# --------------------------------------------------------------------------------------------- helpers
def choose_lower_threshold(roc, thresholds):
    """MELD's rule (calculate_thresholds.py): first peak of normalised sensitivity + specificity after the largest jump."""
    from scipy.signal import find_peaks

    if roc["specificity"][-1] == 0 or roc["sensitivity_plus"][0] == 0:
        sys.exit("Cannot choose a threshold: the trainval subjects need both patients and controls.")
    norm_spec = roc["specificity"] / roc["specificity"][-1]
    norm_sens = roc["sensitivity_plus"] / roc["sensitivity_plus"][0]
    youden = norm_spec + norm_sens
    big_jump = int(np.argmax(np.diff(youden)))
    peaks, _ = find_peaks(youden)
    peaks = peaks[peaks > big_jump]
    index = int(peaks.min()) if len(peaks) else int(np.argmax(youden))
    return float(thresholds[index])


def subject_level_auc(roc):
    """Subject-level ROC AUC as MELD's Evaluator.calculate_aucs ('auc_plus'); NaN without patients or controls."""
    import sklearn.metrics

    roc = dict(roc)
    if roc["specificity"][-1] > 0 and roc["sensitivity_plus"][0] > 0:
        x = 1 - roc["specificity"] / roc["specificity"][-1]
        roc["auc_plus"] = float(sklearn.metrics.auc(x, roc["sensitivity_plus"] / roc["sensitivity_plus"][0]))
        roc["auc"] = float(sklearn.metrics.auc(x, roc["sensitivity"] / roc["sensitivity"][0]))
    else:
        roc["auc_plus"] = roc["auc"] = float("nan")
    return roc


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def summarize(results_dir):
    """Summary metrics from test_results.csv, roc_auc.pickle and sub_aucs.pickle in results_dir."""
    df = pd.read_csv(os.path.join(results_dir, "test_results.csv"))
    patient = as_bool(df["group"])
    detected = as_bool(df["detected"])
    fp = pd.to_numeric(df["number FP clusters"])
    tp = pd.to_numeric(df["number TP clusters"])
    dice = pd.to_numeric(df["dice lesional"])
    n_pat, n_con = int(patient.sum()), int((~patient).sum())
    summary = {
        "n_patients": n_pat,
        "n_controls": n_con,
        "sensitivity": float(detected[patient].mean()) if n_pat else None,
        "specificity": float((fp[~patient] == 0).mean()) if n_con else None,
        "cluster_precision_patients": float(tp[patient].sum() / (tp[patient].sum() + fp[patient].sum()))
        if n_pat and (tp[patient].sum() + fp[patient].sum()) > 0 else None,
        "fp_clusters_per_patient": float(fp[patient].mean()) if n_pat else None,
        "fp_clusters_per_control": float(fp[~patient].mean()) if n_con else None,
        "dice_lesion_median": float(dice[patient].median()) if n_pat else None,
        "dice_lesion_q25_q75": [float(dice[patient].quantile(0.25)), float(dice[patient].quantile(0.75))] if n_pat else None,
    }
    roc_file = os.path.join(results_dir, "roc_auc.pickle")
    if os.path.isfile(roc_file):
        with open(roc_file, "rb") as f:
            summary["subject_roc_auc"] = float(pickle.load(f).get("auc_plus", float("nan")))
    auc_file = os.path.join(results_dir, "sub_aucs.pickle")
    if os.path.isfile(auc_file):
        with open(auc_file, "rb") as f:
            sub_aucs = pickle.load(f)
        summary["vertex_auc_median_patients"] = float(np.median(list(sub_aucs.values()))) if sub_aucs else None
    thr_file = os.path.join(results_dir, "two_thresholds.csv")
    if os.path.isfile(thr_file):
        summary["two_thresholds"] = pd.read_csv(thr_file)[["ymin", "ymax"]].values[0].tolist()
    return summary


def print_summary(name, s):
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    num = lambda v, d=3: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    print(f"\n== {name}: {s['n_patients']} patients, {s['n_controls']} controls")
    print(f"  sensitivity                 {pct(s['sensitivity'])}")
    print(f"  specificity                 {pct(s['specificity'])}")
    print(f"  cluster precision (pat.)    {pct(s['cluster_precision_patients'])}")
    print(f"  FP clusters per patient     {num(s['fp_clusters_per_patient'], 2)}")
    print(f"  FP clusters per control     {num(s['fp_clusters_per_control'], 2)}")
    q = s.get("dice_lesion_q25_q75") or [None, None]
    print(f"  Dice lesion, median [IQR]   {num(s['dice_lesion_median'])} [{num(q[0])}, {num(q[1])}]")
    print(f"  subject-level ROC AUC       {num(s.get('subject_roc_auc'))}")
    print(f"  vertex AUC, median (pat.)   {num(s.get('vertex_auc_median_patients'))}")
    if "two_thresholds" in s:
        print(f"  thresholds (low, high)      {s['two_thresholds'][0]:.2f}, {s['two_thresholds'][1]:.2f}")


# --------------------------------------------------------------------------------------------- MELD steps
def load_create_ensemble():
    """MELD's create_ensemble() from scripts/classifier/create_ensemble.py (next to this script)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_ensemble.py")
    spec = importlib.util.spec_from_file_location("meld_create_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_ensemble


def check_ensemble_weights(ens_dir, model_name):
    """MELD loads checkpoints with strict=False; make sure every weight of the ensemble really matches."""
    import torch

    from meld_graph.experiment import Experiment

    exp = Experiment.from_folder(ens_dir)
    ckpt = os.path.join(ens_dir, f"{model_name}.pt")
    exp.load_model(ckpt)
    result = exp.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if result.missing_keys or result.unexpected_keys:
        sys.exit(f"Ensemble weights do not match the model: missing {result.missing_keys[:5]}, "
                 f"unexpected {result.unexpected_keys[:5]}")
    print(f"ensemble weights OK ({type(exp.model.models[0]).__name__} x {len(exp.model.models)})")


def predict(ckpt_dir, out_dir, model_name, subjects, save_dir, threshold, thresh_and_clust, save_prediction,
            chunk_size, data_parameters=None):
    """
    Run MELD's Evaluator with the ensemble in ckpt_dir on `subjects`, in chunks (GraphDataset keeps a chunk's
    data in memory). data_parameters (optional) replaces the ensemble's own data settings.
    out_dir is where the Evaluator looks for results_<model>/two_thresholds.csv.
    Returns (roc dictionary with AUCs, per-patient vertex AUCs).
    """
    from meld_graph.dataset import GraphDataset
    from meld_graph.evaluation import Evaluator
    from meld_graph.experiment import Experiment
    from meld_graph.meld_cohort import MeldCohort

    exp = Experiment.from_folder(ckpt_dir)
    if data_parameters is not None:
        exp.data_parameters = copy.deepcopy(data_parameters)
    exp.experiment_path = out_dir
    exp.data_parameters["augment_data"] = {}
    cohort = MeldCohort(hdf5_file_root=exp.data_parameters["hdf5_file_root"], dataset=exp.data_parameters["dataset"])
    eva, sub_aucs = None, {}
    for start in range(0, len(subjects), chunk_size):
        chunk = list(subjects[start:start + chunk_size])
        print(f"  subjects {start + 1}-{start + len(chunk)} of {len(subjects)}", flush=True)
        dataset = GraphDataset(chunk, cohort, exp.data_parameters, mode="test")  # test mode: no augmentation
        if eva is None:
            # mode="train" only stops the Evaluator from keeping every subject's input features in memory;
            # the data come from the test-mode dataset above
            eva = Evaluator(experiment=exp, checkpoint_path=ckpt_dir, save_dir=save_dir, make_images=False,
                            dataset=dataset, cohort=cohort, subject_ids=chunk, mode="train",
                            threshold=threshold, thresh_and_clust=thresh_and_clust, model_name=model_name)
            # ROC AUCs are computed once at the end (a chunk may contain no controls)
            eva.calculate_aucs = lambda: None
            eva.save_roc_scores = lambda suffix="": None
        else:
            eva.dataset, eva.subject_ids, eva.cohort = dataset, dataset.subject_ids, dataset.cohort
        eva.data_dictionary = None
        eva.load_predict_data(store_predictions=thresh_and_clust, roc_curves_thresholds=ROC_THRESHOLDS,
                              save_prediction=save_prediction)
        sub_aucs.update(eva.subject_aucs)
        if thresh_and_clust:
            eva.threshold_and_cluster()
            eva.stat_subjects()
        eva.data_dictionary = None
        del dataset
    roc = subject_level_auc(eva.roc_dictionary)
    roc["thresholds"] = ROC_THRESHOLDS
    os.makedirs(eva.results_dir, exist_ok=True)
    with open(os.path.join(eva.results_dir, "roc_auc.pickle"), "wb") as f:
        pickle.dump(roc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(eva.results_dir, "sub_aucs.pickle"), "wb") as f:
        pickle.dump(sub_aucs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return roc, sub_aucs


# --------------------------------------------------------------------------------------------- setup per mode
def resolve(path, experiment_path):
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(experiment_path, path))


def require_files(folder, model_name, hint):
    for name in (f"{model_name}.pt", "data_parameters.json", "network_parameters.json"):
        if not os.path.isfile(os.path.join(folder, name)):
            sys.exit(f"Missing {os.path.join(folder, name)} - {hint}")


def clear(results_dir, *names):
    for name in names:
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def setup_experiment(args, experiment_path):
    """Mode A: build (or reuse) the ensemble of a trained 5-fold experiment."""
    exp_dir = resolve(args.experiment, experiment_path)
    folds = [os.path.join(exp_dir, f"fold_{k:02d}") for k in range(5)]
    for d in folds:
        require_files(d, args.model_name, "are all 5 folds trained?")
    # MELD derives an experiment's folder from its name, so the experiment must live at EXPERIMENT_PATH/<name>
    name = json.load(open(os.path.join(folds[0], "network_parameters.json")))["name"]
    if os.path.realpath(exp_dir) != os.path.realpath(os.path.join(experiment_path, name)):
        sys.exit(f"{exp_dir} does not match the experiment name '{name}' (expected {os.path.join(experiment_path, name)})")
    test_sets = [sorted(json.load(open(os.path.join(d, "data_parameters.json")))["test_ids"]) for d in folds]
    if any(t != test_sets[0] for t in test_sets):
        sys.exit("The folds have different test subjects; they cannot be ensembled.")

    ens_dir = os.path.join(exp_dir, "fold_all")
    results_dir = os.path.join(ens_dir, f"results_{args.model_name}")
    ens_ckpt = os.path.join(ens_dir, f"{args.model_name}.pt")
    stale = os.path.isfile(ens_ckpt) and any(
        os.path.getmtime(os.path.join(d, f"{args.model_name}.pt")) > os.path.getmtime(ens_ckpt) for d in folds)
    redo = args.overwrite or stale or not os.path.isfile(ens_ckpt)
    print(f"\n== 1. ensemble of the 5 folds -> {ens_ckpt}")
    if redo:
        if stale:
            print("  a fold model is newer than the ensemble: rebuilding it and redoing all results")
        clear(results_dir, "two_thresholds.csv", "test_results.csv", "predictions.hdf5")
        load_create_ensemble()(ens_dir, folds, model_name=args.model_name)
    else:
        print("  exists, reused")
    check_ensemble_weights(ens_dir, args.model_name)
    params = json.load(open(os.path.join(ens_dir, "data_parameters.json")))
    parcel = (params.get("parcel_fusion") or {}).get("parcel_dir")
    return dict(label=os.path.relpath(exp_dir, experiment_path), ckpt_dir=ens_dir, out_dir=ens_dir,
                data_parameters=None, trainval=list(params["train_ids"]) + list(params["val_ids"]),
                test=list(params["test_ids"]), redo=redo, parcel_dir=parcel)


def setup_ensemble(args, experiment_path):
    """Mode B: an existing ensemble (e.g. the released MELD model) on the reference experiment's subjects and data."""
    if not args.reference:
        sys.exit("--ensemble needs --reference: the experiment whose test subjects and data settings to use")
    ens_dir = resolve(args.ensemble, experiment_path)
    if os.path.basename(ens_dir) != "fold_all":
        sys.exit(f"{ens_dir}: MELD loads an ensemble only from a folder named fold_all")
    require_files(ens_dir, args.model_name, "is this an ensemble folder?")
    ref_dir = resolve(args.reference, experiment_path)
    ref_file = next((os.path.join(ref_dir, f, "data_parameters.json") for f in ("fold_all", "fold_00")
                     if os.path.isfile(os.path.join(ref_dir, f, "data_parameters.json"))), None)
    if ref_file is None:
        sys.exit(f"No fold_all or fold_00/data_parameters.json in the reference experiment {ref_dir}")
    ref = json.load(open(ref_file))
    own = json.load(open(os.path.join(ens_dir, "data_parameters.json")))
    network_type = json.load(open(os.path.join(ens_dir, "network_parameters.json")))["network_type"]

    # same input data as the reference: its files, features and normalisation, without parcel vectors
    if list(own["features"]) != list(ref["features"]):
        sys.exit(f"{ens_dir} expects different input features than {ref_dir}; the models cannot be fed the same data")
    z_own = os.path.basename(own["preprocessing_parameters"].get("zscore") or "")
    z_ref = os.path.basename(ref["preprocessing_parameters"].get("zscore") or "")
    if z_own != z_ref:
        print(f"WARNING: different z-score files: ensemble {z_own}, reference {z_ref} (the reference's is used)")
    data_parameters = copy.deepcopy(ref)
    data_parameters.pop("parcel_fusion", None)
    if network_type == "MoNetUnetParcelFusion":
        sys.exit("--ensemble is for models without parcel fusion; evaluate fusion experiments with --experiment")

    out_root = resolve(args.output or f"{os.path.basename(os.path.dirname(ens_dir))}_eval", experiment_path)
    out_dir = os.path.join(out_root, "fold_all")
    os.makedirs(os.path.join(out_dir, f"results_{args.model_name}"), exist_ok=True)
    info = {"ensemble": ens_dir, "reference": ref_dir, "network_type": network_type,
            "data_settings_from": ref_file, "parcel_vectors": None}
    with open(os.path.join(out_dir, "evaluated_model.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n== 1. existing ensemble {ens_dir} ({network_type})")
    check_ensemble_weights(ens_dir, args.model_name)
    test = list(ref["test_ids"])
    seen = sorted(set(test) & set(own.get("train_ids", [])) | set(test) & set(own.get("val_ids", [])))
    if seen:
        print(f"WARNING: {len(seen)} of the {len(test)} test subjects were in this model's own training data, "
              f"e.g. {seen[:3]}; its results on them are optimistic")
    print(f"  results -> {out_dir}")
    return dict(label=os.path.relpath(out_root, experiment_path), ckpt_dir=ens_dir, out_dir=out_dir,
                data_parameters=data_parameters, trainval=list(ref["train_ids"]) + list(ref.get("val_ids", [])),
                test=test, redo=args.overwrite, parcel_dir=None)


# --------------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Evaluate an ensemble on test subjects the way MELD does")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--experiment", help="trained 5-fold experiment (fold_00..fold_04), e.g. parcel_fusion/run1")
    mode.add_argument("--ensemble", help="existing ensemble folder, e.g. 23-10-30_LVHZ_dcp/fold_all (the released MELD model)")
    parser.add_argument("--reference", help="with --ensemble: experiment whose test subjects and data settings to use")
    parser.add_argument("--output", help="with --ensemble: results folder under EXPERIMENT_PATH (default <ensemble name>_eval)")
    parser.add_argument("--demographics_csv", required=True, help="the demographics CSV used for training")
    parser.add_argument("--threshold", choices=["two_threshold", "slope_threshold"], default="two_threshold")
    parser.add_argument("--thresholds", choices=["trainval", "released"], default="trainval",
                        help="two_threshold values: chosen on the trainval subjects (default), or the ensemble's own file (--ensemble only)")
    parser.add_argument("--model_name", default="best_model")
    parser.add_argument("--chunk_size", type=int, default=100, help="subjects loaded into memory at a time")
    parser.add_argument("--overwrite", action="store_true", help="redo all steps")
    args = parser.parse_args()

    from meld_graph.paths import DEMOGRAPHIC_FEATURES_FILE, EXPERIMENT_PATH

    if not os.path.isdir("data/icospheres"):
        sys.exit("Run this from the meld_graph repository root (data/icospheres not found).")
    if args.experiment and (args.reference or args.output or args.thresholds == "released"):
        sys.exit("--reference, --output and --thresholds released only go with --ensemble")
    shutil.copy(args.demographics_csv, DEMOGRAPHIC_FEATURES_FILE)
    print(f"Demographics: {args.demographics_csv} -> {DEMOGRAPHIC_FEATURES_FILE}")

    run = setup_experiment(args, EXPERIMENT_PATH) if args.experiment else setup_ensemble(args, EXPERIMENT_PATH)
    print(f"  {len(run['trainval'])} trainval, {len(run['test'])} test subjects; parcel vectors: {run['parcel_dir'] or 'none'}")
    results_dir = os.path.join(run["out_dir"], f"results_{args.model_name}")
    os.makedirs(results_dir, exist_ok=True)
    thr_file = os.path.join(results_dir, "two_thresholds.csv")
    test_csv = os.path.join(results_dir, "test_results.csv")
    summary_file = os.path.join(results_dir, "summary.json")
    previous = json.load(open(summary_file)) if os.path.isfile(summary_file) else {}
    source = args.thresholds if args.threshold == "two_threshold" else None

    # ---- 2. thresholds
    thresholds_changed = False
    if args.threshold == "slope_threshold":
        print("\n== 2. thresholds: slope_threshold, nothing to optimise")
    elif source == "released":
        released = os.path.join(run["ckpt_dir"], f"results_{args.model_name}", "two_thresholds.csv")
        print(f"\n== 2. thresholds: the model's own {released}")
        if not os.path.isfile(released):
            sys.exit(f"{released} not found; use --thresholds trainval")
        old = open(thr_file).read() if os.path.isfile(thr_file) else None
        shutil.copy(released, thr_file)
        thresholds_changed = open(thr_file).read() != old
    else:
        print(f"\n== 2. thresholds from the trainval subjects -> {thr_file}")
        if os.path.isfile(thr_file) and not run["redo"] and previous.get("threshold_source", "trainval") == "trainval":
            print("  exists, reused")
        else:
            roc, _ = predict(run["ckpt_dir"], run["out_dir"], args.model_name, run["trainval"],
                             os.path.join(run["out_dir"], "trainval_eval"), threshold=args.threshold,
                             thresh_and_clust=False, save_prediction=False, chunk_size=args.chunk_size,
                             data_parameters=run["data_parameters"])
            low = choose_lower_threshold(roc, ROC_THRESHOLDS)
            pd.DataFrame([[low, UPPER_THRESHOLD]], columns=["ymin", "ymax"]).to_csv(thr_file)
            thresholds_changed = True
            print(f"  trainval subject-level AUC {roc['auc_plus']:.3f}; thresholds low {low:.2f}, high {UPPER_THRESHOLD:.2f}")
            if low >= UPPER_THRESHOLD:
                print(f"  NOTE: the low threshold is not below {UPPER_THRESHOLD}: MELD's rule found no useful low-confidence "
                      "threshold, so effectively one threshold is used")
    if os.path.isfile(thr_file) and args.threshold == "two_threshold":
        print("  thresholds (low, high): " + ", ".join(f"{v:.2f}" for v in pd.read_csv(thr_file)[["ymin", "ymax"]].values[0]))

    # ---- 3. test subjects
    print(f"\n== 3. test subjects -> {results_dir}")
    n_done = len(pd.read_csv(test_csv)) if os.path.isfile(test_csv) else 0
    same_setup = previous.get("threshold_mode", args.threshold) == args.threshold and previous.get("threshold_source", source) == source
    if n_done == len(run["test"]) and not run["redo"] and not thresholds_changed and same_setup:
        print("  test_results.csv complete, reused")
    else:
        clear(results_dir, "test_results.csv", "predictions.hdf5")
        predict(run["ckpt_dir"], run["out_dir"], args.model_name, run["test"], run["out_dir"], threshold=args.threshold,
                thresh_and_clust=True, save_prediction=True, chunk_size=args.chunk_size,
                data_parameters=run["data_parameters"])

    # ---- 4. summary
    summary = summarize(results_dir)
    summary.update({"evaluated": run["label"], "model_checkpoint": os.path.join(run["ckpt_dir"], f"{args.model_name}.pt"),
                    "parcel_vectors": run["parcel_dir"], "threshold_mode": args.threshold, "threshold_source": source,
                    "model_name": args.model_name})
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(run["label"], summary)
    print(f"\nWritten: {summary_file}")


if __name__ == "__main__":
    main()
