"""
Near-Real-Time Scoring API
============================

Wraps the trained IsolationForest + RandomForest + SHAP explainer behind
a REST endpoint that scores ONE new access-log event at a time, updating
each entity's running behavioral profile incrementally (via
behavior_state.BehaviorState) rather than requiring a full batch re-run.

This directly addresses the brief's "detect intrusions... in near
real-time" framing, which the rest of this project's batch pipeline
(generate -> features -> train -> alert queue) does not, by itself,
satisfy -- that pipeline is for building and evaluating the models, not
for serving them.

HONEST SCOPE NOTE: this is a single-process demo API with in-memory
state (no persistence across restarts, no auth, no horizontal scaling,
approximate IP-fanout tracking). A production deployment would back
BehaviorState with a shared store (e.g. Redis) and add authentication --
called out explicitly in report.docx Section 11, not hidden.

RUN LOCALLY:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
Then see README.md / http://localhost:8000/docs for interactive testing.
"""

import json
from collections import deque
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from behavior_state import BehaviorState
from common import reason_labels_for
from features import FEATURE_COLUMNS
from train_models import build_explainer, explain_alert

app = FastAPI(
    title="Behavioral Anomaly Detection — Scoring API",
    description="Near-real-time risk scoring for access-log events, backed by the trained IsolationForest + RandomForest + SHAP pipeline.",
    version="1.0.0",
)

# ---- loaded once at process startup ----
_MODELS = {}
_STATE: Optional[BehaviorState] = None
_RECENT_ALERTS: deque = deque(maxlen=500)  # in-memory ring buffer
_events_at_warm_start = 0


class AccessEvent(BaseModel):
    entity_id: str
    entity_type: str = Field(..., pattern="^(user|service_account|edge_device)$")
    timestamp: Optional[str] = None  # ISO 8601; defaults to now if omitted
    resource_accessed: str
    session_duration_min: float
    auth_success: bool = True
    geo_lat: float
    geo_lon: float
    geo_location: Optional[str] = "unknown"
    source_ip: Optional[str] = "unknown"
    auth_method: Optional[str] = "unknown"
    device_os: str
    device_mac: str
    command_sequence: Optional[List[str]] = []


class ScoreResponse(BaseModel):
    entity_id: str
    risk_score: float
    predicted_category: str
    reasons: List[str]
    cold_start: bool
    out_of_order: bool
    timestamp: str


@app.on_event("startup")
def startup():
    global _STATE, _events_at_warm_start
    print("Loading models...")
    _MODELS["iso"] = joblib.load("../outputs/isolation_forest.joblib")
    _MODELS["scaler"] = joblib.load("../outputs/scaler.joblib")
    _MODELS["clf"] = joblib.load("../outputs/rf_classifier.joblib")
    with open("../outputs/iso_score_bounds.json") as f:
        _MODELS["bounds"] = json.load(f)

    feat_df = pd.read_csv("../data/features_scored.csv")
    background = feat_df[FEATURE_COLUMNS].fillna(0).sample(min(200, len(feat_df)), random_state=42)
    _MODELS["explainer"] = build_explainer(_MODELS["clf"], background)

    print("Warm-starting behavioral state from historical corpus (this replays the full log once)...")
    hist = pd.read_csv("../data/access_logs.csv")
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    _STATE = BehaviorState().warm_start(hist)
    _events_at_warm_start = _STATE.n_events_seen
    print(f"Warm start complete: {len(_STATE.entities)} known entities, {_STATE.n_events_seen} historical events replayed.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "entities_known": len(_STATE.entities) if _STATE else 0,
        "events_processed_since_warm_start": (_STATE.n_events_seen - _events_at_warm_start) if _STATE else 0,
    }


def _risk_score(x_scaled):
    """Rescale the IsolationForest's raw anomaly score to the 0-100 scale
    fitted during training (see train_models.train_isolation_forest)."""
    raw = -_MODELS["iso"].score_samples(x_scaled)[0]
    bounds = _MODELS["bounds"]
    scaled = 100 * (raw - bounds["raw_min"]) / max(bounds["raw_max"] - bounds["raw_min"], 1e-9)
    return float(np.clip(scaled, 0, 100))


@app.post("/score", response_model=ScoreResponse)
def score_event(event: AccessEvent):
    if _STATE is None:
        raise HTTPException(503, "Model not loaded yet")

    ts = pd.to_datetime(event.timestamp) if event.timestamp else pd.Timestamp.now()
    ev = {
        "entity_id": event.entity_id, "entity_type": event.entity_type, "timestamp": ts,
        "resource_accessed": event.resource_accessed,
        "session_duration_min": event.session_duration_min, "auth_success": event.auth_success,
        "geo_lat": event.geo_lat, "geo_lon": event.geo_lon,
        "device_os": event.device_os, "device_mac": event.device_mac,
        "n_commands": len(event.command_sequence or []),
    }
    features, cold_start, out_of_order = _STATE.score_event(ev)

    x = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)
    x_scaled = _MODELS["scaler"].transform(x.values)
    risk_score = round(_risk_score(x_scaled), 1)

    pred_class, top_feats = explain_alert(_MODELS["explainer"], _MODELS["clf"], x, top_k=3)
    if pred_class == "normal":
        # A positive SHAP value here means "pushed toward normal", not
        # "this is anomalous" -- showing it as an anomaly reason would be
        # actively misleading, so normal predictions get no reason chips.
        reasons = []
    else:
        reasons = reason_labels_for([f for f, _ in top_feats], [v for _, v in top_feats])

    result = {
        "entity_id": event.entity_id, "risk_score": risk_score,
        "predicted_category": pred_class, "reasons": reasons,
        "cold_start": cold_start, "out_of_order": out_of_order, "timestamp": str(ts),
    }
    _RECENT_ALERTS.append(result)
    return result


@app.get("/alerts/top")
def top_alerts(n: int = 20):
    return sorted(_RECENT_ALERTS, key=lambda a: -a["risk_score"])[:n]


@app.get("/entity/{entity_id}")
def entity_profile(entity_id: str):
    if _STATE is None or entity_id not in _STATE.entities:
        raise HTTPException(404, "Unknown entity (no history observed yet)")
    st = _STATE.entities[entity_id]
    return {
        "entity_id": entity_id,
        "entity_type": st["entity_type"],
        "n_events_observed": len(st["hist_hours"]),
        "mean_login_hour": round(float(np.mean(st["hist_hours"])), 2) if st["hist_hours"] else None,
        "mean_session_duration_min": round(float(np.mean(st["hist_durs"])), 2) if st["hist_durs"] else None,
        "distinct_resources_ever": len(st["hist_resources"]),
        "distinct_device_fingerprints_ever": len(st["hist_fingerprints"]),
        "cold_start": len(st["hist_hours"]) < 5,
    }