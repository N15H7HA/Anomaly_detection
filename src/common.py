"""
Shared constants and math helpers used across the batch pipeline
(features.py, sequence_model.py, train_models.py, build_alert_queue.py)
and the online serving path (behavior_state.py, api.py).

Pulling these out avoids re-implementing the same formulas twice with
subtly different code, and gives every consumer of "reason chips" or
"cold-start threshold" a single source of truth. No formula here differs
from the original inline versions -- this is a pure extraction.
"""

from math import radians, sin, cos, asin, sqrt

import numpy as np

# Number of prior events an entity needs before we trust its own running
# profile instead of the population-level prior (cold-start threshold).
MIN_HISTORY = 5

# Backoff smoothing strength toward the population transition graph in the
# sequence-transition score (see sequence_model.py / behavior_state.py).
SEQ_ALPHA = 2.0

# Human-readable labels for each model feature, used to turn a SHAP
# attribution into an analyst-facing "reason" string.
REASON_LABELS = {
    "hour_zscore": "unusual time-of-day",
    "dur_zscore": "atypical session duration",
    "is_new_resource": "first-time resource access",
    "distinct_resources_recent20": "unusually broad resource footprint",
    "geo_distance_km": "large geographic jump",
    "geo_velocity_kmh": "implausible geo-velocity",
    "fingerprint_mismatch": "device fingerprint mismatch",
    "time_since_last_s": "abnormal event timing/burst",
    "auth_success": "authentication outcome",
    "n_commands": "unusual command volume",
    "session_duration_min": "session length deviation",
    "rolling_fail_5": "recent authentication failures",
    "ip_distinct_entities_recent": "many identities from one source IP",
    "cold_start": "limited history for this entity",
    "seq_transition_score": "anomalous action sequence/order",
}


def reason_labels_for(feature_names, shap_values, top_k=3):
    """Turn (feature, shap_value) pairs into up to top_k human-readable
    reason strings, keeping only features that pushed TOWARD the predicted
    class (positive SHAP value). Falls back to the single strongest
    feature if none were positive, so a reason is always shown."""
    impacts = sorted(zip(feature_names, shap_values), key=lambda t: -abs(t[1]))
    reasons = [REASON_LABELS.get(f, f) for f, v in impacts if v > 0][:top_k]
    if not reasons:
        reasons = [REASON_LABELS.get(impacts[0][0], impacts[0][0])]
    return reasons


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def zscore_from_stats(value, mean, std, circular=False):
    """|value - mean| / std, optionally wrapped to a 24-hour clock so that
    e.g. 23:00 and 01:00 are treated as close together rather than far apart."""
    diff = abs(value - mean)
    if circular:
        diff = min(diff, 24 - diff)
    return diff / max(std, 1e-6)


def transition_probability(entity_edge_count, entity_node_total, pop_edge_count, pop_node_total,
                            alpha=SEQ_ALPHA):
    """Katz-style backoff estimate of P(next_resource | prev_resource, entity):
    blends the entity's own transition graph with the population graph for
    that entity_type, so an entity with little history relies mostly on the
    population prior and an entity with lots of history relies on its own."""
    p_pop = (pop_edge_count + 1) / (pop_node_total + 50) if pop_node_total > 0 else 1e-3
    return (entity_edge_count + alpha * p_pop) / (entity_node_total + alpha)


def marginal_probability(resource_count, total_count, n_distinct_resources):
    """Fallback distribution over resources used for an entity's very first
    event (no previous resource to condition a transition on)."""
    if total_count == 0:
        return 1.0
    return (resource_count + 1) / (total_count + n_distinct_resources + 1)