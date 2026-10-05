"""
Build the ranked alert queue that feeds the analyst dashboard.

For every event flagged by the unsupervised detector (top N by risk
score), attach: the classifier's predicted anomaly category, the
ground-truth label (kept ONLY for this demo's dashboard so a reviewer
can sanity-check the system -- in production this column would not
exist at alert time), and the top-3 SHAP feature contributions
translated into a human-readable reason string.
"""

import json

import joblib
import pandas as pd

from common import reason_labels_for
from features import FEATURE_COLUMNS
from train_models import build_explainer, explain_alert


def _load_top_alerts(top_n):
    """Join the top-N riskiest scored events with their raw log details."""
    feat_df = pd.read_csv("../data/features_scored.csv")
    top = feat_df.sort_values("risk_score", ascending=False).head(top_n).copy()

    raw_logs = pd.read_csv("../data/access_logs.csv")
    raw_logs["timestamp"] = pd.to_datetime(raw_logs["timestamp"])
    top = top.merge(
        raw_logs[["event_id", "timestamp", "source_ip", "geo_location",
                   "resource_accessed", "auth_method", "device_os"]],
        on="event_id", how="left"
    )
    return feat_df, top


def main(top_n=150):
    feat_df, top = _load_top_alerts(top_n)
    X_all = feat_df[FEATURE_COLUMNS].fillna(0)

    clf = joblib.load("../outputs/rf_classifier.joblib")
    explainer = build_explainer(clf, X_all.sample(min(200, len(X_all)), random_state=42))

    alerts = []
    for _, row in top.iterrows():
        idx = feat_df.index[feat_df["event_id"] == row["event_id"]][0]
        pred_class, top_feats = explain_alert(explainer, clf, X_all.loc[[idx]], top_k=3)
        reasons = reason_labels_for([f for f, _ in top_feats], [v for _, v in top_feats])

        alerts.append({
            "event_id": row["event_id"],
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"],
            "timestamp": str(row["timestamp"]),
            "risk_score": round(float(row["risk_score"]), 1),
            "predicted_category": pred_class,
            "ground_truth_label": row["label"],  # demo-only visibility
            "reasons": reasons,
            "source_ip": row["source_ip"],
            "geo_location": row["geo_location"],
            "resource_accessed": row["resource_accessed"],
            "auth_method": row["auth_method"],
            "device_os": row["device_os"],
            "cold_start": bool(row["cold_start"]),
        })

    with open("../outputs/alert_queue.json", "w") as f:
        json.dump(alerts, f, indent=2)

    # Summary stats for the dashboard header.
    correctly_flagged = sum(1 for a in alerts if a["ground_truth_label"] != "normal")
    summary = {
        "total_events_scored": int(len(feat_df)),
        "alerts_shown": len(alerts),
        "true_anomalies_in_queue": correctly_flagged,
        "precision_in_queue": round(correctly_flagged / len(alerts), 3),
        "category_breakdown": pd.Series([a["predicted_category"] for a in alerts]).value_counts().to_dict(),
    }
    with open("../outputs/alert_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()