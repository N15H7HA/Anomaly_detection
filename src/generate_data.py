"""
Synthetic Access-Log Generator for Behavioral Anomaly Detection
=================================================================

WHAT THIS GENERATES
--------------------
A table of access/connection events for a mixed population of entities
(human users, service accounts, edge/IoT devices). Each entity has a
stable "behavioral profile" (habitual login hours, home geo-location,
usual resources, typical session length, usual auth method, device
fingerprint). Normal events are sampled around that profile with noise.
A small percentage of sessions are then overwritten with one of seven
injected attack/edge-case patterns, with the ground-truth label kept
in a separate column (as it would be hidden at inference time).

DOCUMENTED BEHAVIORAL ASSUMPTIONS
----------------------------------
1. Each entity has ONE home timezone/geo and a small set (2-4) of
   "usual" secondary geos (e.g. a laptop that also connects from home
   and a coworking space). All other geos are "unusual" for that entity.
2. Human users mostly log in during a personal work-hour window
   (~8-10 hour band) with a Gaussian spread; a small fraction of
   sessions naturally fall outside this window (legitimate late work),
   which is why the model must be probabilistic, not a hard rule.
3. Service accounts have flatter, near-continuous activity (they run
   jobs around the clock) but a strongly repetitive resource set.
4. Edge/IoT devices have a fixed, narrow protocol/resource footprint
   and a fixed device_fingerprint (OS/firmware + MAC) that essentially
   never changes under normal operation.
5. "Normal" resource access follows a Zipf-like distribution per
   entity: a handful of resources account for most access events, with
   a long tail of rarely-touched resources -- this is what makes
   "an unusual breadth of resources" a meaningful lateral-movement
   signal.
6. session_duration and command_sequence length are correlated with
   resource sensitivity (privileged resources -> longer sessions).
7. Attack injection rates follow the brief's 0.5-3% guidance per
   pattern; "insider drift" is intentionally left ambiguous (labeled
   but statistically closer to normal) to stress-test false positive
   tuning, per the brief.

USAGE
-----
python generate_data.py --n-entities 300 --days 30 --out ../data/access_logs.csv

NOTE ON REPRODUCIBILITY
------------------------
All behavioral sampling uses the seeded numpy Generator `rng` passed
through every function below. Refactors in this file are careful to
never add, remove, or reorder an `rng.*` call relative to the original
implementation -- doing so would silently change every downstream
random draw and produce a different (though equally valid) dataset.
`uuid.uuid4()` and Faker calls are NOT seeded by `rng` and were never
reproducible run-to-run in the original script either, so touching
those is safe.
"""

import argparse
import json
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker 

from common import haversine_km

fake = Faker()
Faker.seed(42)

# ---------------------------------------------------------------------------
# Reference pools
# ---------------------------------------------------------------------------

ENTITY_TYPES = ["user", "service_account", "edge_device"]
ENTITY_TYPE_WEIGHTS = [0.65, 0.15, 0.20]

AUTH_METHODS = ["password", "token", "certificate", "biometric", "sso"]

RESOURCE_POOL = (
    [f"/file/dept_share/{i}" for i in range(40)]
    + [f"/api/v1/endpoint_{i}" for i in range(30)]
    + [f"port:{p}" for p in [22, 80, 443, 3389, 3306, 8080, 502, 1883]]
    + [f"device_fn/{f}" for f in
       ["telemetry_read", "firmware_update", "control_write", "config_read",
        "sensor_poll", "actuator_cmd"]]
    + ["/admin/user_mgmt", "/admin/audit_log", "/finance/ledger",
       "/hr/payroll", "/eng/source_repo", "/eng/ci_pipeline"]
)

PRIVILEGED_RESOURCES = {
    "/admin/user_mgmt", "/admin/audit_log", "/finance/ledger", "/hr/payroll",
    "/eng/source_repo", "/eng/ci_pipeline",
}

COMMANDS = (
    ["ls", "cd", "cat", "read", "list_files", "get_metadata"]
    + ["put", "post", "update_record", "write_config"]
    + ["whoami", "sudo -l", "net user", "reg query", "ps aux"]  # priv-recon flavored
    + ["scp", "curl", "wget", "export_data"]
)

# A rough world geo pool: (lat, lon, label)
GEO_POOL = [
    (17.385, 78.4867, "Hyderabad,IN"), (12.9716, 77.5946, "Bengaluru,IN"),
    (28.6139, 77.2090, "Delhi,IN"), (19.0760, 72.8777, "Mumbai,IN"),
    (13.0827, 80.2707, "Chennai,IN"), (1.3521, 103.8198, "Singapore,SG"),
    (51.5072, -0.1276, "London,GB"), (40.7128, -74.0060, "New York,US"),
    (37.7749, -122.4194, "San Francisco,US"), (35.6762, 139.6503, "Tokyo,JP"),
    (55.7558, 37.6173, "Moscow,RU"), (-23.5505, -46.6333, "Sao Paulo,BR"),
    (52.5200, 13.4050, "Berlin,DE"), (25.2048, 55.2708, "Dubai,AE"),
    (6.5244, 3.3792, "Lagos,NG"),
]


def _device_fields(profile):
    """The three device-fingerprint columns, read the same way in every
    injected event -- pulled out to avoid repeating this 3-line lookup
    in all seven attack injectors plus the normal-event sampler."""
    fp = profile["fingerprint"]
    return {"device_os": fp["os"], "device_mac": fp["mac"], "protocol": fp["protocol"]}


# ---------------------------------------------------------------------------
# Entity profile construction
# ---------------------------------------------------------------------------

def build_entity_profiles(n_entities, rng):
    profiles = {}
    for _ in range(n_entities):
        etype = rng.choice(ENTITY_TYPES, p=ENTITY_TYPE_WEIGHTS)
        entity_id = (
            f"user::{fake.user_name()}_{uuid.uuid4().hex[:4]}" if etype == "user"
            else f"svc::{fake.word()}_{uuid.uuid4().hex[:4]}" if etype == "service_account"
            else f"dev::{fake.word()}_{uuid.uuid4().hex[:4]}"
        )
        home_geo = GEO_POOL[rng.integers(0, len(GEO_POOL))]
        n_secondary = rng.integers(0, 3)
        secondary_geos = [GEO_POOL[i] for i in
                           rng.choice(len(GEO_POOL), size=n_secondary, replace=False)]
        usual_geos = [home_geo] + secondary_geos

        n_resources = rng.integers(4, 10) if etype != "edge_device" else rng.integers(2, 4)
        usual_resources = list(rng.choice(RESOURCE_POOL, size=n_resources, replace=False))
        # Zipf-like weights over the usual resource set
        zipf_weights = 1.0 / np.arange(1, n_resources + 1)
        zipf_weights = zipf_weights / zipf_weights.sum()

        if etype == "user":
            work_hour_center = rng.uniform(8, 18)
            work_hour_spread = rng.uniform(1.5, 3.0)
            auth_method = rng.choice(["password", "token", "sso", "biometric"], p=[0.35, 0.3, 0.25, 0.1])
        elif etype == "service_account":
            work_hour_center = 12.0
            work_hour_spread = 8.0  # near-continuous
            auth_method = rng.choice(["token", "certificate"], p=[0.6, 0.4])
        else:  # edge_device
            work_hour_center = 12.0
            work_hour_spread = 7.0
            auth_method = "certificate"

        fingerprint = {
            "os": rng.choice(["Linux 5.15", "Windows 11", "iOS 17", "RTOS-fw2.3", "Android 14"]),
            "mac": fake.mac_address(),
            "protocol": rng.choice(["TLS1.3", "SSH2", "MQTT", "Modbus/TCP", "HTTPS"]),
        }

        session_dur_mean = rng.uniform(3, 45)  # minutes
        session_dur_std = max(1.0, session_dur_mean * 0.3)

        profiles[entity_id] = dict(
            entity_type=etype,
            usual_geos=usual_geos,
            usual_resources=usual_resources,
            resource_weights=zipf_weights,
            work_hour_center=work_hour_center,
            work_hour_spread=work_hour_spread,
            auth_method=auth_method,
            fingerprint=fingerprint,
            session_dur_mean=session_dur_mean,
            session_dur_std=session_dur_std,
        )
    return profiles


# ---------------------------------------------------------------------------
# Normal event sampling
# ---------------------------------------------------------------------------

def sample_normal_event(entity_id, profile, ts, rng):
    geo = profile["usual_geos"][rng.integers(0, len(profile["usual_geos"]))]
    resource = rng.choice(profile["usual_resources"], p=profile["resource_weights"])
    session_dur = max(0.2, rng.normal(profile["session_dur_mean"], profile["session_dur_std"]))
    is_priv = resource in PRIVILEGED_RESOURCES
    n_cmds = rng.integers(3, 9) if is_priv else rng.integers(1, 5)
    cmd_seq = list(rng.choice(COMMANDS, size=n_cmds, replace=True))
    success = rng.random() > 0.03  # occasional legitimate typo/fail

    return {
        "event_id": uuid.uuid4().hex,
        "entity_id": entity_id,
        "entity_type": profile["entity_type"],
        "timestamp": ts.isoformat(),
        "source_ip": fake.ipv4_public(),
        "geo_lat": geo[0], "geo_lon": geo[1], "geo_location": geo[2],
        "resource_accessed": resource,
        "auth_method": profile["auth_method"],
        "auth_success": success,
        "session_duration_min": round(session_dur, 2),
        "command_sequence": json.dumps(cmd_seq),
        **_device_fields(profile),
        "label": "normal",
    }


def sample_event_time(profile, day_start, rng):
    hour = rng.normal(profile["work_hour_center"], profile["work_hour_spread"]) % 24
    minute = rng.integers(0, 60)
    return day_start + timedelta(hours=float(hour), minutes=float(minute))


# ---------------------------------------------------------------------------
# Attack pattern injectors -- each returns a LIST of event dicts
# ---------------------------------------------------------------------------

def inject_brute_force(entity_id, profile, ts, rng):
    """Rapid repeated failed-auth attempts from one source IP, short window."""
    src_ip = fake.ipv4_public()
    n = rng.integers(8, 25)
    events = []
    t = ts
    for i in range(n):
        t = t + timedelta(seconds=int(rng.integers(2, 15)))
        events.append({
            "event_id": uuid.uuid4().hex, "entity_id": entity_id,
            "entity_type": profile["entity_type"], "timestamp": t.isoformat(),
            "source_ip": src_ip,
            "geo_lat": profile["usual_geos"][0][0], "geo_lon": profile["usual_geos"][0][1],
            "geo_location": profile["usual_geos"][0][2],
            "resource_accessed": rng.choice(profile["usual_resources"]),
            "auth_method": profile["auth_method"],
            "auth_success": i == n - 1 and rng.random() < 0.3,  # rarely succeeds at the end
            "session_duration_min": 0.1,
            "command_sequence": json.dumps([]),
            **_device_fields(profile),
            "label": "brute_force",
        })
    return events


def inject_impossible_travel(entity_id, profile, ts, rng):
    """Same entity logging in from two geographically distant points too fast."""
    home = profile["usual_geos"][0]
    far_candidates = [g for g in GEO_POOL if haversine_km(*home[:2], *g[:2]) > 3000]
    far_geo = far_candidates[rng.integers(0, len(far_candidates))] if far_candidates else GEO_POOL[0]
    gap_minutes = rng.integers(5, 90)  # implausible for the distance
    events = []
    ev1 = sample_normal_event(entity_id, profile, ts, rng)
    ev1["label"] = "impossible_travel"
    events.append(ev1)
    t2 = ts + timedelta(minutes=int(gap_minutes))
    ev2 = {
        "event_id": uuid.uuid4().hex, "entity_id": entity_id,
        "entity_type": profile["entity_type"], "timestamp": t2.isoformat(),
        "source_ip": fake.ipv4_public(),
        "geo_lat": far_geo[0], "geo_lon": far_geo[1], "geo_location": far_geo[2],
        "resource_accessed": rng.choice(profile["usual_resources"]),
        "auth_method": profile["auth_method"], "auth_success": True,
        "session_duration_min": round(max(0.2, rng.normal(5, 2)), 2),
        "command_sequence": json.dumps(list(rng.choice(COMMANDS, size=2))),
        **_device_fields(profile),
        "label": "impossible_travel",
    }
    events.append(ev2)
    return events


def inject_credential_stuffing(entities_subset, profiles, ts, rng):
    """Many entity_ids, few source_ips, high failure rate -- a cluster event."""
    src_ips = [fake.ipv4_public() for _ in range(rng.integers(1, 3))]
    events = []
    t = ts
    for entity_id in entities_subset:
        profile = profiles[entity_id]
        t = t + timedelta(seconds=int(rng.integers(1, 10)))
        events.append({
            "event_id": uuid.uuid4().hex, "entity_id": entity_id,
            "entity_type": profile["entity_type"], "timestamp": t.isoformat(),
            "source_ip": src_ips[rng.integers(0, len(src_ips))],
            "geo_lat": GEO_POOL[rng.integers(0, len(GEO_POOL))][0],
            "geo_lon": GEO_POOL[rng.integers(0, len(GEO_POOL))][1],
            "geo_location": "unknown_stuffing_origin",
            "resource_accessed": rng.choice(profile["usual_resources"]),
            "auth_method": profile["auth_method"],
            "auth_success": rng.random() < 0.05,
            "session_duration_min": 0.05,
            "command_sequence": json.dumps([]),
            "device_os": "unknown", "device_mac": "unknown",
            "protocol": profile["fingerprint"]["protocol"],
            "label": "credential_stuffing",
        })
    return events


def inject_lateral_movement(entity_id, profile, ts, rng):
    """Compromised entity touches an unusual breadth/sequence of resources."""
    unusual_resources = [r for r in RESOURCE_POOL if r not in profile["usual_resources"]]
    n_touch = rng.integers(5, 12)
    touched = list(rng.choice(unusual_resources, size=min(n_touch, len(unusual_resources)), replace=False))
    events = []
    t = ts
    for r in touched:
        t = t + timedelta(minutes=int(rng.integers(1, 6)))
        is_priv = r in PRIVILEGED_RESOURCES
        events.append({
            "event_id": uuid.uuid4().hex, "entity_id": entity_id,
            "entity_type": profile["entity_type"], "timestamp": t.isoformat(),
            "source_ip": fake.ipv4_public(),
            "geo_lat": profile["usual_geos"][0][0], "geo_lon": profile["usual_geos"][0][1],
            "geo_location": profile["usual_geos"][0][2],
            "resource_accessed": r,
            "auth_method": profile["auth_method"], "auth_success": True,
            "session_duration_min": round(max(0.2, rng.normal(8 if is_priv else 3, 2)), 2),
            "command_sequence": json.dumps(list(rng.choice(COMMANDS, size=int(rng.integers(3, 7))))),
            **_device_fields(profile),
            "label": "lateral_movement",
        })
    return events


def inject_device_spoofing(entity_id, profile, ts, rng):
    """device_id reappears with a mismatched fingerprint."""
    ev = sample_normal_event(entity_id, profile, ts, rng)
    ev["device_os"] = rng.choice([o for o in
        ["Linux 5.15", "Windows 11", "iOS 17", "RTOS-fw2.1-modified", "Android 14", "unknown-firmware"]
        if o != profile["fingerprint"]["os"]])
    ev["device_mac"] = fake.mac_address()  # different MAC than history
    ev["label"] = "device_spoofing"
    return [ev]


def inject_low_and_slow(entity_id, profile, ts, rng, days=10):
    """Gradual, small, off-hours access building up over days/weeks."""
    events = []
    unusual_resources = [r for r in RESOURCE_POOL if r not in profile["usual_resources"]]
    for d in range(days):
        t = ts + timedelta(days=d, hours=float(rng.uniform(1, 4)))  # off-hours (1-4am)
        r = unusual_resources[rng.integers(0, len(unusual_resources))]
        events.append({
            "event_id": uuid.uuid4().hex, "entity_id": entity_id,
            "entity_type": profile["entity_type"], "timestamp": t.isoformat(),
            "source_ip": fake.ipv4_public(),
            "geo_lat": profile["usual_geos"][0][0], "geo_lon": profile["usual_geos"][0][1],
            "geo_location": profile["usual_geos"][0][2],
            "resource_accessed": r,
            "auth_method": profile["auth_method"], "auth_success": True,
            "session_duration_min": round(max(0.2, rng.normal(2, 0.5)), 2),
            "command_sequence": json.dumps(["read", "export_data"]),
            **_device_fields(profile),
            "label": "low_and_slow_exfil",
        })
    return events


def inject_insider_drift(entity_id, profile, ts, rng, days=14):
    """Legitimate entity slowly expands footprint -- ambiguous edge case."""
    events = []
    all_candidates = profile["usual_resources"] + list(
        rng.choice(RESOURCE_POOL, size=3, replace=False))
    n_usual = len(profile["usual_resources"])
    for d in range(days):
        t = sample_event_time(profile, ts + timedelta(days=d), rng)  # normal hours
        # increasing chance of touching a "new" resource as days progress
        p_new = min(0.5, 0.05 * d)
        if rng.random() < p_new:
            # Same branching as before, just spelled out instead of nested
            # ternaries: pick the newest candidate, or (if more than one
            # "new" candidate exists) a random one from that extra set.
            if rng.random() < 0.5:
                r = all_candidates[-1]
            elif len(all_candidates) > n_usual:
                r = all_candidates[rng.integers(n_usual, len(all_candidates))]
            else:
                r = all_candidates[-1]
        else:
            r = rng.choice(profile["usual_resources"], p=profile["resource_weights"])
        events.append({
            "event_id": uuid.uuid4().hex, "entity_id": entity_id,
            "entity_type": profile["entity_type"], "timestamp": t.isoformat(),
            "source_ip": fake.ipv4_public(),
            "geo_lat": profile["usual_geos"][0][0], "geo_lon": profile["usual_geos"][0][1],
            "geo_location": profile["usual_geos"][0][2],
            "resource_accessed": r,
            "auth_method": profile["auth_method"], "auth_success": True,
            "session_duration_min": round(max(0.2, rng.normal(profile["session_dur_mean"], profile["session_dur_std"])), 2),
            "command_sequence": json.dumps(list(rng.choice(COMMANDS, size=int(rng.integers(1, 5))))),
            **_device_fields(profile),
            "label": "insider_drift",
        })
    return events


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(n_entities=300, days=30, seed=42, out_path="../data/access_logs.csv"):
    rng = np.random.default_rng(seed)
    profiles = build_entity_profiles(n_entities, rng)
    entity_ids = list(profiles.keys())
    start_date = datetime(2026, 6, 1)

    all_events = []

    # 1. Normal baseline traffic: each entity gets several events per day
    for entity_id, profile in profiles.items():
        events_per_day = rng.integers(1, 6) if profile["entity_type"] == "user" else rng.integers(3, 15)
        for d in range(days):
            day_start = start_date + timedelta(days=d)
            for _ in range(events_per_day):
                ts = sample_event_time(profile, day_start, rng)
                all_events.append(sample_normal_event(entity_id, profile, ts, rng))

    # 2. Injected attack patterns at controlled rates (of ENTITY-DAYS, roughly 0.5-3%)
    n_bruteforce = max(1, int(n_entities * days * 0.006))
    n_impossible = max(1, int(n_entities * days * 0.004))
    n_stuffing_events = max(1, int(n_entities * days * 0.003))
    n_lateral = max(1, int(n_entities * days * 0.005))
    n_spoof = max(1, int(n_entities * days * 0.006))
    n_slow = max(1, int(n_entities * 0.02))
    n_drift = max(1, int(n_entities * 0.03))

    def rand_ts():
        d = rng.integers(0, days)
        return start_date + timedelta(days=int(d), hours=float(rng.uniform(0, 23)))

    for _ in range(n_bruteforce):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        all_events.extend(inject_brute_force(eid, profiles[eid], rand_ts(), rng))

    for _ in range(n_impossible):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        all_events.extend(inject_impossible_travel(eid, profiles[eid], rand_ts(), rng))

    stuffing_events_left = n_stuffing_events
    while stuffing_events_left > 0:
        batch_size = min(stuffing_events_left, int(rng.integers(15, 60)))
        subset = list(rng.choice(entity_ids, size=min(batch_size, len(entity_ids)), replace=False))
        all_events.extend(inject_credential_stuffing(subset, profiles, rand_ts(), rng))
        stuffing_events_left -= batch_size

    for _ in range(n_lateral):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        all_events.extend(inject_lateral_movement(eid, profiles[eid], rand_ts(), rng))

    for _ in range(n_spoof):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        all_events.extend(inject_device_spoofing(eid, profiles[eid], rand_ts(), rng))

    for _ in range(n_slow):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        d_start = start_date + timedelta(days=int(rng.integers(0, max(1, days - 12))))
        all_events.extend(inject_low_and_slow(eid, profiles[eid], d_start, rng,
                                               days=min(10, days)))

    for _ in range(n_drift):
        eid = entity_ids[rng.integers(0, len(entity_ids))]
        d_start = start_date + timedelta(days=int(rng.integers(0, max(1, days - 15))))
        all_events.extend(inject_insider_drift(eid, profiles[eid], d_start, rng,
                                                days=min(14, days)))

    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save ground truth separately (as it would be hidden at inference)
    ground_truth = df[["event_id", "label"]].copy()
    df_no_label = df.drop(columns=["label"])

    df.to_csv(out_path, index=False)  # full file WITH labels, for training/eval
    df_no_label.to_csv(out_path.replace(".csv", "_unlabeled.csv"), index=False)
    ground_truth.to_csv(out_path.replace(".csv", "_ground_truth.csv"), index=False)

    # Save entity profiles (for cold-start reference / baseline construction)
    prof_export = {
        eid: {k: v for k, v in p.items() if k not in ("usual_geos",)}
        for eid, p in profiles.items()
    }
    with open(out_path.replace(".csv", "_profiles.json"), "w") as f:
        json.dump(prof_export, f, indent=2, default=str)

    print(f"Generated {len(df):,} events across {n_entities} entities over {days} days")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-entities", type=int, default=300)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="../data/access_logs.csv")
    args = parser.parse_args()
    generate(args.n_entities, args.days, args.seed, args.out)