#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVI Detection with Explainable Boosting Machine (EBM) — End‑to‑End, Fully Explainable

What this script does:
- Loads your feature table (feature_table.csv) with selected physical features
- Trains an Explainable Boosting Classifier (InterpretML) with early stopping
- Evaluates on validation + test sets (ROC/PR, calibration, confusion matrix, Brier score, metrics)
- Produces fully explainable outputs:
  * Global: term importances, univariate shapes, pairwise interaction surfaces
  * Local: per-sample contribution breakdown (log‑odds) for selected cases
  * Uncertainty: bagging variance via predict_with_uncertainty (if available)
- Saves artifacts: trained model (.joblib), JSON/Excel scorecards, plots, CSVs (importances, local explanations)

Usage:
    python ovi_ebm_pipeline.py

Dependencies (install once):
    pip install interpret scikit-learn pandas numpy matplotlib joblib xlsxwriter

IMPORTANT: Update BASE_PATH / OUTPUT_FILE to your local path if needed.
"""

import os
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight

# InterpretML
from interpret.glassbox import ExplainableBoostingClassifier

# --------------------- 0) Paths & columns ---------------------
BASE_PATH = '/Users/wavefunction/ASU Dropbox/Tanmay Singh/'
OUTPUT_FILE = os.path.join(BASE_PATH, 'Synthetic_IGrM_Sightlines/TNG50_fitting_results/feature_table.csv')

# Allow fallback to /mnt/data for convenience
if not os.path.exists(OUTPUT_FILE):
    alt_path = '/mnt/data/feature_table.csv'
    if os.path.exists(alt_path):
        OUTPUT_FILE = alt_path
    else:
        raise FileNotFoundError(f'Cannot find feature_table.csv at {OUTPUT_FILE} or {alt_path}. Please update OUTPUT_FILE.')

FEATURES = [
    'log_M_halo',
    'log_M_star_group',
    'impact_param_group',
    'impact_param_galaxy',
    'log_M_star_galaxy',
    'log_sSFR_galaxy',
    'is_central',
    'is_star_forming',
    'is_bound'
]
TARGET = 'has_OVI_absorber'

ARTIFACTS_DIR = Path('./ebm_artifacts')
PLOTS_DIR = ARTIFACTS_DIR / 'plots'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------- 1) Load data ---------------------
usecols = FEATURES + [TARGET]
dtypes = {
    'log_M_halo': 'float32',
    'log_M_star_group': 'float32',
    'impact_param_group': 'float32',
    'impact_param_galaxy': 'float32',
    'log_M_star_galaxy': 'float32',
    'log_sSFR_galaxy': 'float32',
    'is_central': 'int8',
    'is_star_forming': 'int8',
    'is_bound': 'int8',
    TARGET: 'int8'
}

print('Loading:', OUTPUT_FILE)
data = pd.read_csv(OUTPUT_FILE, usecols=usecols, dtype=dtypes, low_memory=True)
print('Raw shape:', data.shape)

# Basic checks
assert set(FEATURES).issubset(data.columns), "Missing expected features."
assert TARGET in data.columns, "Missing target column."
data = data.dropna(subset=[TARGET])

# Class balance
pos_rate = data[TARGET].mean()
print(f"Positive rate (has_OVI_absorber==1): {pos_rate:.4f}")

X = data[FEATURES]
y = data[TARGET].astype(int)

# Split: train/test, then validation from train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
)
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

# Balanced sample weights (train only)
sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

# --------------------- 2) Train EBM ---------------------
ebm = ExplainableBoostingClassifier(
    interactions=8,
    max_bins=256,
    max_leaves=3,
    learning_rate=0.01,
    outer_bags=8,
    validation_size=0.1,
    early_stopping_rounds=100,
    random_state=42,
    n_jobs=-1
)
ebm.fit(X_train, y_train, sample_weight=sample_weight_train)

# --------------------- 3) Evaluation helpers ---------------------
def plot_roc_pr(y_true, p, split_name):
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    roc_auc = roc_auc_score(y_true, p)
    ap = average_precision_score(y_true, p)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {split_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"roc_{split_name.lower()}.png"); plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {split_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"pr_{split_name.lower()}.png"); plt.close()
    return roc_auc, ap

def plot_calibration(y_true, p, split_name):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=20, strategy='quantile')
    bs = brier_score_loss(y_true, p)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker='o', linestyle='-')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Calibration — {split_name} (Brier={bs:.3f})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"calibration_{split_name.lower()}.png"); plt.close()
    return bs

def print_metrics(y_true, p, split_name, threshold):
    y_pred = (p >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, p)
    ap = average_precision_score(y_true, p)
    print(f"\n[{split_name}]  Acc={acc:.3f}  BalAcc={bacc:.3f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  AUC={auc:.3f}  AP={ap:.3f}")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix — {split_name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"confusion_{split_name.lower()}.png"); plt.close()
    return {"accuracy": acc, "balanced_accuracy": bacc, "f1": f1, "precision": prec, "recall": rec, "auc": auc, "ap": ap}

# Probabilities
p_tr = ebm.predict_proba(X_train)[:, 1]
p_va = ebm.predict_proba(X_valid)[:, 1]
p_te = ebm.predict_proba(X_test)[:, 1]

# Curves
roc_tr, ap_tr = plot_roc_pr(y_train, p_tr, "Train")
roc_va, ap_va = plot_roc_pr(y_valid, p_va, "Valid")
roc_te, ap_te = plot_roc_pr(y_test, p_te, "Test")

brier_tr = plot_calibration(y_train, p_tr, "Train")
brier_va = plot_calibration(y_valid, p_va, "Valid")
brier_te = plot_calibration(y_test, p_te, "Test")

# Threshold selection on validation (maximize F1)
prec_v, rec_v, th_v = precision_recall_curve(y_valid, p_va)
f1s = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)
best_idx = np.nanargmax(f1s)
best_threshold = th_v[max(best_idx - 1, 0)] if best_idx > 0 else 0.5
print(f"Chosen threshold (from Valid, max F1): {best_threshold:.4f}")

m_tr = print_metrics(y_train, p_tr, "Train", best_threshold)
m_va = print_metrics(y_valid, p_va, "Valid", best_threshold)
m_te = print_metrics(y_test, p_te, "Test", best_threshold)

eval_summary = {
    "roc": {"train": float(roc_tr), "valid": float(roc_va), "test": float(roc_te)},
    "ap": {"train": float(ap_tr), "valid": float(ap_va), "test": float(ap_te)},
    "brier": {"train": float(brier_tr), "valid": float(brier_va), "test": float(brier_te)},
    "threshold": float(best_threshold),
    "metrics": {"train": m_tr, "valid": m_va, "test": m_te}
}
with open(ARTIFACTS_DIR / "evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
print("Saved evaluation summary to:", ARTIFACTS_DIR / "evaluation_summary.json")

# --------------------- 4) Global interpretability ---------------------
importances = ebm.term_importances()
term_names = ebm.term_names_
term_features = ebm.term_features_

term_df = pd.DataFrame({
    "term": term_names,
    "importance": importances,
    "is_interaction": [len(t) > 1 for t in term_features]
}).sort_values("importance", ascending=False).reset_index(drop=True)
term_df.to_csv(ARTIFACTS_DIR / "ebm_term_importances.csv", index=False)
print("\nTop terms by importance:\n", term_df.head(20))

# Global importances plot
top_k = 20 if len(term_df) > 20 else len(term_df)
plt.figure()
plt.barh(term_df["term"].iloc[:top_k][::-1], term_df["importance"].iloc[:top_k][::-1])
plt.xlabel("Term importance (avg abs contribution)")
plt.title("EBM — Top Term Importances")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "global_importances_top.png"); plt.close()

# Shapes via explanation object
global_exp = ebm.explain_global(name="EBM Global")

def plot_univariate_shape(term_index, topn=50):
    d = global_exp.data(term_index)
    names = d.get("names", [])
    scores = d.get("scores", [])
    # Downsample if too many bins
    if isinstance(names, (list, tuple)) and len(names) > topn:
        step = max(1, len(names)//topn)
        names = names[::step]
        scores = scores[::step]
    x = np.arange(len(names))
    plt.figure()
    plt.plot(x, scores, marker='o')
    plt.xticks(x, names, rotation=90)
    plt.ylabel("Contribution (log-odds)")
    plt.title(f"Shape — {term_names[term_index]}")
    plt.tight_layout()
    fname = f"shape_{term_index:03d}_{term_names[term_index].replace(' ', '_')}.png"
    plt.savefig(PLOTS_DIR / fname); plt.close()

# Plot top 8 main effects
main_effect_indices = [i for i, t in enumerate(term_features) if len(t) == 1]
main_effect_indices_sorted = sorted(main_effect_indices, key=lambda i: importances[i], reverse=True)
for i in main_effect_indices_sorted[:8]:
    plot_univariate_shape(i)

# Pairwise interactions
def plot_interaction(term_index, topx=40, topy=40):
    d = global_exp.data(term_index)
    names = d.get("names", [])
    scores = np.array(d.get("scores", []))
    if not isinstance(names, (list, tuple)) or len(names) != 2 or scores.ndim != 2:
        return
    names_x, names_y = names
    # Downsample
    if len(names_x) > topx:
        step_x = max(1, len(names_x)//topx)
        names_x = names_x[::step_x]
        scores = scores[::step_x, :]
    if len(names_y) > topy:
        step_y = max(1, len(names_y)//topy)
        names_y = names_y[::step_y]
        scores = scores[:, ::step_y]
    plt.figure()
    plt.imshow(scores.T, aspect='auto', origin='lower')
    plt.colorbar(label="Contribution (log-odds)")
    # Ticks
    xticks = np.arange(len(names_x))
    yticks = np.arange(len(names_y))
    plt.xticks(xticks, names_x, rotation=90)
    plt.yticks(yticks, names_y)
    title = f"Interaction — {term_names[term_index]}"
    plt.title(title)
    plt.xlabel(term_names[term_index].split(" x ")[0])
    plt.ylabel(term_names[term_index].split(" x ")[1] if " x " in term_names[term_index] else "")
    plt.tight_layout()
    fname = f"interaction_{term_index:03d}_{term_names[term_index].replace(' ', '_')}.png"
    plt.savefig(PLOTS_DIR / fname); plt.close()

inter_indices = [i for i, t in enumerate(term_features) if len(t) == 2]
inter_indices_sorted = sorted(inter_indices, key=lambda i: importances[i], reverse=True)
for i in inter_indices_sorted[:4]:
    plot_interaction(i)

# --------------------- 5) Local explanations ---------------------
p_test = p_te
y_pred_test = (p_test >= best_threshold).astype(int)

idx_tp = np.where((y_test.values == 1) & (y_pred_test == 1))[0]
idx_fn = np.where((y_test.values == 1) & (y_pred_test == 0))[0]
idx_fp = np.where((y_test.values == 0) & (y_pred_test == 1))[0]

def explain_one(ix, label="case"):
    if ix is None:
        print(f"No index for {label}."); return
    x_row = X_test.iloc[[ix]]
    # label not needed for scoring; pass any scalar
    local = ebm.explain_local(x_row, np.array([0]))
    d = local.data(0)
    names = d.get("names", [])
    scores = np.array(d.get("scores", []))
    order = np.argsort(np.abs(scores))[::-1]
    names = [names[i] for i in order]
    scores = scores[order]
    # Bar plot
    y_pos = np.arange(len(names))
    plt.figure()
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, names)
    plt.xlabel("Contribution (log-odds)")
    plt.title(f"Local explanation — {label}")
    plt.tight_layout()
    safe_label = label.replace(" ", "_")
    plt.savefig(PLOTS_DIR / f"local_{safe_label}.png"); plt.close()
    # CSV
    pd.DataFrame({"term": names, "contribution_log_odds": scores}).to_csv(
        ARTIFACTS_DIR / f"local_{safe_label}.csv", index=False
    )

explain_one(int(idx_tp[0]) if len(idx_tp) else None, label="True_Positive")
explain_one(int(idx_fn[0]) if len(idx_fn) else None, label="False_Negative")
explain_one(int(idx_fp[0]) if len(idx_fp) else None, label="False_Positive")

# --------------------- 6) Uncertainty via predict_with_uncertainty ---------------------
try:
    pu = ebm.predict_with_uncertainty(X_test.values)
    mean_pred = pu[:, 0]
    uncert = pu[:, 1]
    # Histogram
    plt.figure()
    plt.hist(uncert, bins=50)
    plt.xlabel("Uncertainty (std across bagged models)"); plt.ylabel("Count")
    plt.title("EBM — Prediction Uncertainty (Test)")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "uncertainty_hist.png"); plt.close()
    # Scatter
    plt.figure()
    plt.scatter(uncert, mean_pred, s=3, alpha=0.3)
    plt.xlabel("Uncertainty"); plt.ylabel("Predicted probability (class 1)")
    plt.title("Uncertainty vs. Predicted Probability")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "uncertainty_vs_prob.png"); plt.close()
except Exception as e:
    print("predict_with_uncertainty not available in this version of InterpretML:", e)

# --------------------- 7) Export model & scorecards ---------------------
model_path = ARTIFACTS_DIR / "ebm_model.joblib"
joblib.dump(ebm, model_path)
print("Saved model to:", model_path)

json_path = ARTIFACTS_DIR / "ebm_model.json"
ebm.to_json(json_path, detail='interpretable', indent=2)
print("Saved model JSON to:", json_path)

try:
    xlsx_path = ARTIFACTS_DIR / "ebm_model.xlsx"
    ebm.to_excel(xlsx_path)
    print("Saved model Excel to:", xlsx_path)
except Exception as e:
    print("Excel export failed (install xlsxwriter?):", e)

with open(ARTIFACTS_DIR / "chosen_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\nAll artifacts saved under:", ARTIFACTS_DIR.resolve())
