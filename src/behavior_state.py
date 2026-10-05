"""
Incremental Behavioral State — powers the near-real-time scoring API
========================================================================

features.py and sequence_model.py compute per-entity baselines and the
graph-based sequence score in a single BATCH pass over a static CSV.
That's fine for training/evaluation, but the brief asks for detection
"in near real-time" -- a real deployment needs to score ONE new event
at a time as it arrives, updating each entity's running profile
incrementally, without re-reading the whole log.

This module is the same logic as features.py / sequence_model.py,
refactored into a class that holds running state in memory and exposes
a single `score_event()` call per incoming event. `api.py` warm-starts
one of these from the historical CSV (so day-1-in-production entities
aren't all cold-start) and then calls `score_event()` for each new
event as it arrives.

Two details intentionally differ from the batch pipeline, both because
streaming can't assume a clean, pre-sorted, complete log:
  - geo velocity uses abs() around the time delta and a coarser floor
    (1e-2h vs batch's 1e-4h), so a late/out-of-order event can't produce
    a near-zero or negative denominator and blow up the velocity.
  - the sequence-transition score is rescaled with a fixed logistic
    curve instead of a percentile rank, because a percentile needs the
    full batch distribution, which isn't available online. The curve
    is calibrated to match the batch run's typical neg-log-prob range.
"""

import json
from collections import defaultdict

import numpy as np

from common import MIN_HISTORY, SEQ_ALPHA, haversine_km, marginal_probability, transition_probability, zscore_from_stats


class BehaviorState:
    """Holds every entity's running profile plus the population-level and
    per-entity transition graphs. Thread-unsafe by design (a single
    process should own this) -- fine for a demo API; a production
    deployment would back this with a shared store (Redis, etc.),
    called out explicitly in the report."""

    def __init__(self):
        self.entities = {}  # entity_id -> dict of running lists/sets/state
        self.pop_priors = defaultdict(lambda: {"hours": [], "durs": []})
        self.pop_edge = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.pop_node_total = defaultdict(lambda: defaultdict(int))
        self.pop_marginal = defaultdict(lambda: defaultdict(int))
        self.pop_marginal_total = defaultdict(int)
        self.n_events_seen = 0

    def _get_entity(self, entity_id, entity_type):
        if entity_id not in self.entities:
            self.entities[entity_id] = dict(
                entity_type=entity_type, hist_hours=[], hist_durs=[],
                hist_resources=set(), hist_fingerprints=set(),
                last_ts=None, last_geo=None, recent_resources=[], recent_fail_flags=[],
                ent_edge=defaultdict(lambda: defaultdict(int)),
                ent_node_total=defaultdict(int), prev_resource=None,
            )
        return self.entities[entity_id]

    def score_event(self, ev: dict) -> dict:
        """ev must contain: entity_id, entity_type, timestamp (datetime),
        resource_accessed, session_duration_min, auth_success (bool),
        geo_lat, geo_lon, device_os, device_mac, n_commands (int, optional).
        Returns the same feature dict schema used in FEATURE_COLUMNS,
        AND updates internal state with this event (no leakage: features
        are computed from state BEFORE this event is folded in)."""
        etype = ev["entity_type"]
        st = self._get_entity(ev["entity_id"], etype)
        hour_of_day = ev["timestamp"].hour + ev["timestamp"].minute / 60.0
        n_commands = ev.get("n_commands", len(ev.get("command_sequence", [])))

        cold_start = len(st["hist_hours"]) < MIN_HISTORY

        # --- time-of-day / session-duration deviation ---
        pop_hours, pop_durs = self.pop_priors[etype]["hours"], self.pop_priors[etype]["durs"]
        if cold_start:
            hour_mean = np.mean(pop_hours) if pop_hours else hour_of_day
            hour_std = max(np.std(pop_hours), 1e-6) if pop_hours else 1.0
            dur_mean = np.mean(pop_durs) if pop_durs else ev["session_duration_min"]
            dur_std = max(np.std(pop_durs), 1e-6) if pop_durs else 1.0
        else:
            hour_mean, hour_std = np.mean(st["hist_hours"]), max(np.std(st["hist_hours"]), 1e-6)
            dur_mean, dur_std = np.mean(st["hist_durs"]), max(np.std(st["hist_durs"]), 1e-6)
        hour_zscore = zscore_from_stats(hour_of_day, hour_mean, hour_std, circular=True)
        dur_zscore = zscore_from_stats(ev["session_duration_min"], dur_mean, dur_std)

        # --- new resource? / breadth in recent window ---
        is_new_resource = int(ev["resource_accessed"] not in st["hist_resources"]) if not cold_start else 0
        distinct_recent = len(set(st["recent_resources"][-20:]))

        # --- geo velocity ---
        # NOTE: this assumes events arrive in non-decreasing timestamp order
        # per entity, as a real streaming deployment would guarantee (or
        # would need a watermarking/buffering policy for late arrivals,
        # which is out of scope for this demo). If a caller submits an
        # out-of-order timestamp anyway, dt is floored using abs() so the
        # result stays a finite, sane number instead of blowing up from a
        # near-zero or negative denominator.
        if st["last_geo"] is not None:
            dist_km = haversine_km(st["last_geo"][0], st["last_geo"][1], ev["geo_lat"], ev["geo_lon"])
            dt_hours = max(abs((ev["timestamp"] - st["last_ts"]).total_seconds()) / 3600.0, 1e-2)
            geo_velocity_kmh = dist_km / dt_hours
        else:
            dist_km, geo_velocity_kmh = 0.0, 0.0
        out_of_order = bool(st["last_ts"] is not None and ev["timestamp"] < st["last_ts"])

        # --- fingerprint mismatch ---
        fp = (ev["device_os"], ev["device_mac"])
        fingerprint_mismatch = int(bool(st["hist_fingerprints"]) and fp not in st["hist_fingerprints"] and not cold_start)

        # --- time since last event ---
        time_since_last_s = abs((ev["timestamp"] - st["last_ts"]).total_seconds()) if st["last_ts"] is not None else 3600.0

        # --- rolling auth-failure count (last 5 events incl. this one) ---
        recent_fails = st["recent_fail_flags"]
        rolling_fail_5 = sum(recent_fails[-4:]) + (0 if ev["auth_success"] else 1)

        # --- IP fan-out: not trackable per-event without a shared IP index;
        #     the online API approximates with 1 (no other entities seen
        #     yet on this IP in this session) unless the caller supplies it ---
        ip_distinct_entities_recent = ev.get("ip_distinct_entities_recent", 1)

        # --- graph-based sequence-transition score ---
        prev_resource, cur = st["prev_resource"], ev["resource_accessed"]
        if prev_resource is None:
            p = marginal_probability(
                self.pop_marginal[etype][cur], self.pop_marginal_total[etype], len(self.pop_marginal[etype])
            )
        else:
            p = transition_probability(
                st["ent_edge"][prev_resource][cur], st["ent_node_total"][prev_resource],
                self.pop_edge[etype][prev_resource][cur], self.pop_node_total[etype][prev_resource],
            )
        seq_neg_log_prob = -np.log(max(p, 1e-9))
        # Percentile rescaling isn't available online (no full distribution);
        # a fixed logistic squashing calibrated from the batch run's typical
        # range (neg-log-prob of ~0-12) approximates the same 0-100 scale.
        seq_transition_score = float(100 / (1 + np.exp(-(seq_neg_log_prob - 5))))

        features = {
            "hour_zscore": hour_zscore, "dur_zscore": dur_zscore,
            "is_new_resource": is_new_resource,
            "distinct_resources_recent20": distinct_recent,
            "geo_distance_km": dist_km, "geo_velocity_kmh": geo_velocity_kmh,
            "fingerprint_mismatch": fingerprint_mismatch,
            "time_since_last_s": time_since_last_s,
            "auth_success": int(ev["auth_success"]), "n_commands": n_commands,
            "session_duration_min": ev["session_duration_min"],
            "rolling_fail_5": rolling_fail_5,
            "ip_distinct_entities_recent": ip_distinct_entities_recent,
            "cold_start": int(cold_start),
            "seq_transition_score": seq_transition_score,
        }

        # --- fold this event into state (AFTER feature computation) ---
        st["hist_hours"].append(hour_of_day)
        st["hist_durs"].append(ev["session_duration_min"])
        st["hist_resources"].add(cur)
        st["hist_fingerprints"].add(fp)
        st["recent_resources"].append(cur)
        recent_fails.append(0 if ev["auth_success"] else 1)
        st["recent_fail_flags"] = recent_fails[-5:]
        if prev_resource is not None:
            st["ent_edge"][prev_resource][cur] += 1
            st["ent_node_total"][prev_resource] += 1
            self.pop_edge[etype][prev_resource][cur] += 1
            self.pop_node_total[etype][prev_resource] += 1
        self.pop_marginal[etype][cur] += 1
        self.pop_marginal_total[etype] += 1
        st["prev_resource"] = cur
        st["last_ts"] = ev["timestamp"]
        st["last_geo"] = (ev["geo_lat"], ev["geo_lon"])
        self.pop_priors[etype]["hours"].append(hour_of_day)
        self.pop_priors[etype]["durs"].append(ev["session_duration_min"])
        self.n_events_seen += 1

        return features, cold_start, out_of_order

    def warm_start(self, df):
        """Replay a historical dataframe (sorted by entity, timestamp) to
        catch this state up to 'now', exactly like the batch pipeline."""
        df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        for _, row in df.iterrows():
            ev = {
                "entity_id": row["entity_id"], "entity_type": row["entity_type"],
                "timestamp": row["timestamp"], "resource_accessed": row["resource_accessed"],
                "session_duration_min": row["session_duration_min"],
                "auth_success": bool(row["auth_success"]),
                "geo_lat": row["geo_lat"], "geo_lon": row["geo_lon"],
                "device_os": row["device_os"], "device_mac": row["device_mac"],
                "n_commands": len(json.loads(row["command_sequence"])) if isinstance(row["command_sequence"], str) else 0,
            }
            self.score_event(ev)  # discard features, we only want the state update
        return self