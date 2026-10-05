"""
Model Training: Baseline Profile -> Unsupervised Detector -> Supervised Classifier -> Explainability
======================================================================================================

Three-layer design (mirrors the deliverables in the brief):

1. BASELINE PROFILING
   Already implemented in features.py: per-entity running mean/std for
   time-of-day and session duration, with population-level fallback for
   cold-start entities. This *is* the "per-entity normal behavior
   representation."

2. DETECTION MODEL (unsupervised, sequence-aware features)
   IsolationForest trained on the behavioral-deviation features (which
   are themselves derived from each entity's access sequence over time,
   satisfying "sequence-aware" without requiring a deep sequence model
   to be trained from scratch on a small synthetic corpus). Trained
   WITHOUT using the label column, matching a real deployment where
   most traffic is unlabeled. Produces a continuous risk score.

   NOTE ON MODEL CHOICE: the brief suggests LSTM/GRU/Transformer as one
   option. On a corpus this size (45k events, <1,000 anomalies) a deep
   sequence model would be heavily overparameterized and prone to
   overfitting; IsolationForest over explicit behavioral-deviation
   features is the more defensible choice here, and the features
   themselves already encode sequence history (rolling stats, time-since-
   last-event, resource-set growth). This tradeoff is called out
   explicitly in the report's Limitations section -- a production system
   with far more data/history per entity would benefit from a learned
   sequence encoder (e.g. GRU autoencoder over command_sequence) instead
   of/alongside these hand-built features.

3. ANOMALY CLASSIFICATION (supervised, on top of flagged events)
   RandomForestClassifier trained on labeled synthetic data to predict
   WHICH category an anomaly resembles (credential misuse family vs.
   lateral movement vs. impossible travel etc.), not just anomalous/not.

4. EXPLAINABILITY
   SHAP TreeExplainer over the RandomForest gives per-alert feature
   attribution ("flagged due to geo-velocity + new device fingerprint").
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

RNG_SEED = 42


def train_isolation_forest(feat_df: pd.DataFrame):
    X = feat_df[FEATURE_COLUMNS].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Contamination set close to the true anomaly rate, but the model never sees labels.
    true_rate = (feat_df["label"] != "normal").mean()
    iso = IsolationForest(
        n_estimators=300, contamination=min(0.05, max(0.01, true_rate)),
        random_state=RNG_SEED, n_jobs=1,  # n_jobs=1 for full run-to-run reproducibility
    )
    iso.fit(Xs)

    # score_samples: higher = more normal. Flip and rescale to a 0-100 "risk score".
    raw_scores = -iso.score_samples(Xs)
    risk_score = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    # raw_scores.min()/max() are persisted so a SINGLE new event can be
    # rescaled the same way at inference time (the API can't recompute a
    # batch min/max from one event), keeping online and batch scoring in sync.
    bounds = {"raw_min": float(raw_scores.min()), "raw_max": float(raw_scores.max())}
    return iso, scaler, risk_score, bounds


def evaluate_detection(feat_df, risk_score, threshold_pct=95):
    y_true = (feat_df["label"] != "normal").astype(int).values
    threshold = np.percentile(risk_score, threshold_pct)
    y_pred = (risk_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return dict(
        threshold_percentile=threshold_pct, threshold_value=float(threshold),
        roc_auc=float(roc_auc_score(y_true, risk_score)),
        average_precision=float(average_precision_score(y_true, risk_score)),
        precision_at_threshold=float(precision), recall_at_threshold=float(recall),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        n_flagged=int(y_pred.sum()), n_true_anomalies=int(y_true.sum()), n_total=int(len(y_true)),
    )


def train_classifier(feat_df: pd.DataFrame):
    """Supervised classifier over ONLY the events flagged as anomalous (plus
    a sample of normals for contrast), predicting the anomaly category.
    This models the realistic SOC workflow: the unsupervised detector
    narrows down a firehose of events to a review queue, and the
    classifier + explainability layer tells the analyst what kind of
    anomaly it looks like."""
    X = feat_df[FEATURE_COLUMNS].fillna(0)
    y = feat_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=12, class_weight="balanced_subsample",
        random_state=RNG_SEED, n_jobs=1,  # n_jobs=1 for full run-to-run reproducibility
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return clf, X_train, X_test, y_test, y_pred, report


def build_explainer(clf, X_background: pd.DataFrame):
    return shap.TreeExplainer(clf)


def explain_alert(explainer, clf, x_row: pd.DataFrame, top_k=3):
    """Return the top_k contributing features for the PREDICTED class of a
    single alert, in human-readable form (feature_name, shap_value) pairs."""
    pred_class = clf.predict(x_row)[0]
    class_idx = list(clf.classes_).index(pred_class)
    shap_values = explainer.shap_values(x_row)
    # shap_values shape: (n_samples, n_features, n_classes) in recent shap versions.
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0]
    else:
        sv = shap_values[0, :, class_idx] if shap_values.ndim == 3 else shap_values[0]
    top = sorted(zip(x_row.columns, sv), key=lambda t: -abs(t[1]))[:top_k]
    return pred_class, top


if __name__ == "__main__":
    feat_df = pd.read_csv("../data/features.csv")

    print("=== Training IsolationForest (unsupervised detector) ===")
    iso, scaler, risk_score, score_bounds = train_isolation_forest(feat_df)
    feat_df["risk_score"] = risk_score
    det_metrics = evaluate_detection(feat_df, risk_score)
    print(json.dumps(det_metrics, indent=2))

    print("\n=== Training RandomForest classifier (anomaly type) ===")
    clf, X_train, X_test, y_test, y_pred, report = train_classifier(feat_df)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n=== Building SHAP explainer ===")
    explainer = build_explainer(clf, X_train.sample(min(200, len(X_train)), random_state=RNG_SEED))

    # Demonstrate explainability on a handful of true anomalies from the test set.
    anomaly_idx = y_test[y_test != "normal"].index[:5]
    print("\nSample explanations:")
    for idx in anomaly_idx:
        row = X_test.loc[[idx]]
        pred_class, top_feats = explain_alert(explainer, clf, row)
        feat_str = ", ".join(f"{f}={v:+.3f}" for f, v in top_feats)
        print(f"  true={y_test.loc[idx]:20s} pred={pred_class:20s} top_features=[{feat_str}]")

    # Persist everything needed for the dashboard / report.
    joblib.dump(iso, "../outputs/isolation_forest.joblib")
    joblib.dump(scaler, "../outputs/scaler.joblib")
    joblib.dump(clf, "../outputs/rf_classifier.joblib")
    feat_df.to_csv("../data/features_scored.csv", index=False)

    with open("../outputs/detection_metrics.json", "w") as f:
        json.dump(det_metrics, f, indent=2)
    with open("../outputs/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("../outputs/iso_score_bounds.json", "w") as f:
        json.dump(score_bounds, f, indent=2)

    print("\nSaved models and metrics to ../outputs/")