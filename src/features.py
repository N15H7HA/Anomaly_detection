"""
Feature Engineering: Behavioral Profiling + Sequence-Aware Features
=====================================================================

For every event, we compute features relative to that ENTITY's own
history up to (not including) that event -- this is what lets a
statistically rare-but-globally-common action (e.g. logging in at
2am) be flagged for a 9-5 accountant but ignored for a night-shift
service account. This directly implements the "baseline profiling
model" + "sequence-aware detection" deliverables together, since the
detection features are literally deviations from the running profile.

Cold-start handling: entities with fewer than MIN_HISTORY prior events
get a `cold_start` flag and their z-score features are computed against
a population-level prior (all entities of the same entity_type) instead
of their own empty history. This is a documented, explicit strategy for
deliverable #4 (cold-start problem) -- see report.md for discussion.
"""

import json
import numpy as np
import pandas as pd

from common import MIN_HISTORY, haversine_km, zscore_from_stats

FEATURE_COLUMNS = [
    "hour_zscore", "dur_zscore", "is_new_resource", "distinct_resources_recent20",
    "geo_distance_km", "geo_velocity_kmh", "fingerprint_mismatch", "time_since_last_s",
    "auth_success", "n_commands", "session_duration_min", "rolling_fail_5",
    "ip_distinct_entities_recent", "cold_start", "seq_transition_score",
]


def _population_priors(df):
    """Per-entity_type (hour, duration) mean/std, used as the cold-start
    fallback when an entity doesn't yet have enough history of its own."""
    priors = {}
    for etype, g in df.groupby("entity_type"):
        priors[etype] = dict(
            hour_mean=g["hour_of_day"].mean(), hour_std=max(g["hour_of_day"].std(), 1e-6),
            dur_mean=g["session_duration_min"].mean(), dur_std=max(g["session_duration_min"].std(), 1e-6),
        )
    return priors


def _entity_features(g, etype_prior):
    """Compute per-event features for a single entity's events (already
    sorted by timestamp), by replaying its history event-by-event and
    updating running state AFTER each event's features are computed
    (no leakage)."""
    hist_hours, hist_durs = [], []
    hist_resources, hist_fingerprints = set(), set()
    recent_resources_window = []  # sliding window, last 20 events
    last_ts, last_geo = None, None
    rows = []

    for _, row in g.iterrows():
        cold_start = len(hist_hours) < MIN_HISTORY

        if cold_start:
            hour_mean, hour_std = etype_prior["hour_mean"], etype_prior["hour_std"]
            dur_mean, dur_std = etype_prior["dur_mean"], etype_prior["dur_std"]
        else:
            hour_mean, hour_std = np.mean(hist_hours), max(np.std(hist_hours), 1e-6)
            dur_mean, dur_std = np.mean(hist_durs), max(np.std(hist_durs), 1e-6)

        hour_zscore = zscore_from_stats(row["hour_of_day"], hour_mean, hour_std, circular=True)
        dur_zscore = zscore_from_stats(row["session_duration_min"], dur_mean, dur_std)

        is_new_resource = int(row["resource_accessed"] not in hist_resources) if not cold_start else 0
        distinct_recent = len(set(recent_resources_window[-20:]))

        if last_geo is not None:
            dist_km = haversine_km(last_geo[0], last_geo[1], row["geo_lat"], row["geo_lon"])
            dt_hours = max((row["timestamp"] - last_ts).total_seconds() / 3600.0, 1e-4)
            geo_velocity_kmh = dist_km / dt_hours
        else:
            dist_km, geo_velocity_kmh = 0.0, 0.0

        fp = (row["device_os"], row["device_mac"])
        fingerprint_mismatch = int(bool(hist_fingerprints) and fp not in hist_fingerprints and not cold_start)
        time_since_last_s = (row["timestamp"] - last_ts).total_seconds() if last_ts is not None else np.nan

        rows.append({
            "event_id": row["event_id"], "entity_id": row["entity_id"], "entity_type": row["entity_type"],
            "cold_start": int(cold_start),
            "hour_zscore": hour_zscore, "dur_zscore": dur_zscore,
            "is_new_resource": is_new_resource, "distinct_resources_recent20": distinct_recent,
            "geo_distance_km": dist_km, "geo_velocity_kmh": geo_velocity_kmh,
            "fingerprint_mismatch": fingerprint_mismatch, "time_since_last_s": time_since_last_s,
            "auth_success": int(row["auth_success"]), "n_commands": row["n_commands"],
            "session_duration_min": row["session_duration_min"], "hour_of_day": row["hour_of_day"],
            "label": row["label"],
        })

        # Update running history AFTER computing features (no leakage).
        hist_hours.append(row["hour_of_day"])
        hist_durs.append(row["session_duration_min"])
        hist_resources.add(row["resource_accessed"])
        hist_fingerprints.add(fp)
        recent_resources_window.append(row["resource_accessed"])
        last_ts, last_geo = row["timestamp"], (row["geo_lat"], row["geo_lon"])

    return rows


def _rolling_fail_5(df):
    """Per-entity rolling count of auth failures in the last 5 events
    (brute-force / credential-stuffing signal)."""
    df = df.sort_values(["entity_id", "timestamp"]).copy()
    df["auth_fail"] = (~df["auth_success"]).astype(int)
    df["rolling_fail_5"] = (
        df.groupby("entity_id")["auth_fail"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    )
    return df


def _ip_distinct_entities_recent(df, window=20):
    """For each event, how many distinct entities hit the same source_ip
    in the preceding `window` events from that IP (credential-stuffing /
    IP-fanout signal)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ip_distinct = np.zeros(len(df))
    for _, idxs in df.groupby("source_ip").groups.items():
        idxs = list(idxs)
        eids = df.loc[idxs, "entity_id"].tolist()
        for pos, i in enumerate(idxs):
            ip_distinct[i] = len(set(eids[max(0, pos - (window - 1)):pos + 1]))
    df["ip_distinct_entities_recent"] = ip_distinct
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["command_sequence"] = df["command_sequence"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["n_commands"] = df["command_sequence"].apply(len)

    priors = _population_priors(df)
    rows = []
    for eid, g in df.groupby("entity_id", sort=False):
        rows.extend(_entity_features(g, priors[g["entity_type"].iloc[0]]))
    feat_df = pd.DataFrame(rows)

    df = _rolling_fail_5(df)
    df = _ip_distinct_entities_recent(df)
    feat_df = feat_df.merge(
        df[["event_id", "rolling_fail_5", "ip_distinct_entities_recent"]], on="event_id", how="left"
    )
    # First event per entity has no time_since_last; fill with the median.
    feat_df["time_since_last_s"] = feat_df["time_since_last_s"].fillna(feat_df["time_since_last_s"].median())

    # Graph-based sequence-transition score (see sequence_model.py).
    from sequence_model import build_transition_scores
    seq_df = build_transition_scores(df)
    feat_df = feat_df.merge(seq_df[["event_id", "seq_transition_score"]], on="event_id", how="left")
    feat_df["seq_transition_score"] = feat_df["seq_transition_score"].fillna(50.0)

    return feat_df


if __name__ == "__main__":
    df = pd.read_csv("../data/access_logs.csv")
    feats = build_features(df)
    feats.to_csv("../data/features.csv", index=False)
    print(feats.shape)
    print(feats["label"].value_counts())
    print(feats[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])