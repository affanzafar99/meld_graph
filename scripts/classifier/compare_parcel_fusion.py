"""
Compare experiments evaluated with scripts/classifier/evaluate_parcel_fusion.py, subject by subject.

Typical use: parcel fusion vs. the zero-vector baseline (same folds, same training, parcel vectors all 0):
    python scripts/classifier/compare_parcel_fusion.py \
        --run fusion=parcel_fusion/run1 --run zeros=parcel_fusion_zeros/run1

Reads <EXPERIMENT_PATH>/<experiment>/fold_all/results_best_model/{test_results.csv, summary.json, sub_aucs.pickle}
(experiment paths may also be absolute) and prints:
  - the summary metrics of every run side by side
  - for every pair of runs, paired tests on the test subjects both runs scored:
      detection of patients           exact McNemar test (patients detected by one run but not the other)
      controls without clusters       exact McNemar test
      Dice (patients)                 Wilcoxon signed-rank test
      vertex AUC (patients)           Wilcoxon signed-rank test
      sensitivity / specificity diff. 95% bootstrap interval (subjects resampled, --n_bootstrap times)
Per-subject results of all runs are written to --out (CSV).

Needs only numpy, pandas and scipy, so it can run on the login node.
"""

import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    from meld_graph.paths import EXPERIMENT_PATH
except Exception:  # absolute experiment paths still work without meld_graph
    EXPERIMENT_PATH = None


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def results_dir(experiment, model_name):
    exp_dir = experiment if os.path.isabs(experiment) or EXPERIMENT_PATH is None else os.path.join(EXPERIMENT_PATH, experiment)
    return os.path.join(exp_dir, "fold_all", f"results_{model_name}")


def load_run(label, experiment, model_name):
    rdir = results_dir(experiment, model_name)
    csv = os.path.join(rdir, "test_results.csv")
    if not os.path.isfile(csv):
        sys.exit(f"{label}: {csv} not found - run evaluate_parcel_fusion.py for {experiment} first")
    df = pd.read_csv(csv)
    out = pd.DataFrame({
        "ID": df["ID"].astype(str),
        "patient": as_bool(df["group"]),
        "detected": as_bool(df["detected"]),
        "fp_clusters": pd.to_numeric(df["number FP clusters"]),
        "tp_clusters": pd.to_numeric(df["number TP clusters"]),
        "dice": pd.to_numeric(df["dice lesional"]),
    })
    if out["ID"].duplicated().any():
        sys.exit(f"{label}: {csv} lists some subjects twice - rerun evaluate_parcel_fusion.py with --overwrite")
    out["clean"] = out["fp_clusters"] + out["tp_clusters"] == 0  # no cluster at all (relevant for controls)
    auc_file = os.path.join(rdir, "sub_aucs.pickle")
    sub_aucs = pickle.load(open(auc_file, "rb")) if os.path.isfile(auc_file) else {}
    out["vertex_auc"] = out["ID"].map(sub_aucs)
    summary_file = os.path.join(rdir, "summary.json")
    summary = json.load(open(summary_file)) if os.path.isfile(summary_file) else {}
    return out.set_index("ID"), summary


def mcnemar_exact(a, b):
    """Exact McNemar test for paired booleans a, b. Returns (only a, only b, p)."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    try:
        p = stats.binomtest(k, n, 0.5).pvalue
    except AttributeError:  # scipy < 1.7
        p = stats.binom_test(k, n, 0.5)
    return only_a, only_b, float(min(1.0, p))


def wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return len(d), float("nan"), 1.0
    return len(d), float(np.median(d)), float(stats.wilcoxon(d).pvalue)


def bootstrap_diff(a, b, n_boot, rng):
    """95% interval of mean(a) - mean(b) for paired booleans, resampling subjects."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt_p(p):
    return "n/a" if p is None or np.isnan(p) else (f"{p:.3f}" if p >= 0.001 else "<0.001")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluated experiments subject by subject")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=EXPERIMENT",
                        help="e.g. fusion=parcel_fusion/run1; give at least two")
    parser.add_argument("--model_name", default="best_model")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--out", default=None, help="per-subject CSV (default: comparison_<labels>.csv)")
    args = parser.parse_args()

    runs = []
    for item in args.run:
        if "=" not in item:
            sys.exit(f"--run {item}: use LABEL=EXPERIMENT")
        label, experiment = item.split("=", 1)
        runs.append((label, experiment))
    if len(runs) < 2:
        sys.exit("Give at least two --run")
    data = {label: load_run(label, exp, args.model_name) for label, exp in runs}
    labels = [label for label, _ in runs]

    # ---- summaries side by side
    rows = [
        ("patients / controls", lambda s: f"{s.get('n_patients')} / {s.get('n_controls')}"),
        ("sensitivity", lambda s: f"{100 * s['sensitivity']:.1f}%" if s.get("sensitivity") is not None else "n/a"),
        ("specificity", lambda s: f"{100 * s['specificity']:.1f}%" if s.get("specificity") is not None else "n/a"),
        ("cluster precision (pat.)", lambda s: f"{100 * s['cluster_precision_patients']:.1f}%" if s.get("cluster_precision_patients") is not None else "n/a"),
        ("FP clusters / patient", lambda s: f"{s['fp_clusters_per_patient']:.2f}" if s.get("fp_clusters_per_patient") is not None else "n/a"),
        ("FP clusters / control", lambda s: f"{s['fp_clusters_per_control']:.2f}" if s.get("fp_clusters_per_control") is not None else "n/a"),
        ("Dice lesion, median", lambda s: f"{s['dice_lesion_median']:.3f}" if s.get("dice_lesion_median") is not None else "n/a"),
        ("subject-level ROC AUC", lambda s: f"{s['subject_roc_auc']:.3f}" if s.get("subject_roc_auc") is not None else "n/a"),
        ("vertex AUC, median (pat.)", lambda s: f"{s['vertex_auc_median_patients']:.3f}" if s.get("vertex_auc_median_patients") is not None else "n/a"),
        ("thresholds (low, high)", lambda s: ", ".join(f"{t:.2f}" for t in s["two_thresholds"]) if s.get("two_thresholds") else s.get("threshold_mode", "n/a")),
    ]
    width = max(14, *(len(l) + 2 for l in labels))
    print("\n" + " " * 28 + "".join(f"{l:>{width}}" for l in labels))
    for name, fn in rows:
        print(f"{name:<28}" + "".join(f"{fn(data[l][1]):>{width}}" for l in labels))

    # ---- paired comparisons
    rng = np.random.default_rng(0)
    for la, lb in itertools.combinations(labels, 2):
        a, b = data[la][0], data[lb][0]
        common = a.index.intersection(b.index)
        if len(common) < max(len(a), len(b)):
            print(f"\nWARNING: {la} and {lb} share only {len(common)} of {len(a)} / {len(b)} test subjects; comparing those")
        a, b = a.loc[common], b.loc[common]
        if (a["patient"] != b["patient"]).any():
            sys.exit(f"{la} and {lb} disagree on which subjects are patients - were they evaluated on the same data?")
        pat, con = a["patient"].values, ~a["patient"].values
        print(f"\n== {la} vs {lb}  ({int(pat.sum())} patients, {int(con.sum())} controls in both)")
        oa, ob, p = mcnemar_exact(a["detected"].values[pat], b["detected"].values[pat])
        lo, hi = bootstrap_diff(a["detected"].values[pat], b["detected"].values[pat], args.n_bootstrap, rng)
        diff = a["detected"].values[pat].mean() - b["detected"].values[pat].mean() if pat.any() else float("nan")
        print(f"  sensitivity difference      {100 * diff:+.1f} points  (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
        print(f"    detected only by {la}: {oa}, only by {lb}: {ob}  ->  McNemar p = {fmt_p(p)}")
        if con.any():
            oa, ob, p = mcnemar_exact(a["clean"].values[con], b["clean"].values[con])
            lo, hi = bootstrap_diff(a["clean"].values[con], b["clean"].values[con], args.n_bootstrap, rng)
            diff = a["clean"].values[con].mean() - b["clean"].values[con].mean()
            print(f"  specificity difference      {100 * diff:+.1f} points  (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
            print(f"    controls clean only with {la}: {oa}, only with {lb}: {ob}  ->  McNemar p = {fmt_p(p)}")
        n, med, p = wilcoxon(a["dice"].values[pat], b["dice"].values[pat])
        print(f"  Dice (patients)             median difference {med:+.3f} over {n} patients  ->  Wilcoxon p = {fmt_p(p)}")
        n, med, p = wilcoxon(a["vertex_auc"].values[pat], b["vertex_auc"].values[pat])
        print(f"  vertex AUC (patients)       median difference {med:+.3f} over {n} patients  ->  Wilcoxon p = {fmt_p(p)}")

    # ---- per-subject table
    table = None
    for label in labels:
        df = data[label][0][["patient", "detected", "clean", "fp_clusters", "tp_clusters", "dice", "vertex_auc"]]
        df = df.rename(columns={c: f"{c}_{label}" for c in df.columns if c != "patient"})
        table = df if table is None else table.join(df.drop(columns="patient"), how="outer")
    out = args.out or f"comparison_{'_vs_'.join(labels)}.csv"
    table.reset_index().to_csv(out, index=False)
    print(f"\nPer-subject results: {out}")
    print("Positive differences favour the first run of each pair.")


if __name__ == "__main__":
    main()
