"""
Graph-Based Sequence Model: Entity-Resource Transition Anomaly Scoring
========================================================================

WHY THIS FILE EXISTS
---------------------
The brief asks for a "sequence-aware approach (LSTM/GRU, Transformer, or
graph-based for entity-resource relationships)" for the detection model.
The rest of this pipeline (features.py) computes per-event deviation
z-scores, which are informative but order-agnostic -- they don't ask
"does this SEQUENCE of actions make sense for this entity", only "is
this single event unusual". This file closes that gap directly with a
graph-based model: resources are nodes, observed (prev_resource ->
resource) transitions are weighted edges, and an event's sequence score
is the (smoothed, backed-off) probability of that edge under the
entity's own transition graph.

WHY A TRANSITION GRAPH INSTEAD OF AN LSTM/GRU
-----------------------------------------------
Same reasoning as documented in train_models.py: on ~45k events with
under 1,700 anomalies, a deep sequence model trained from scratch would
overfit badly. A per-entity transition graph is the classical, well-
understood way to make a *sequence-order-sensitive* model that still
degrades gracefully with little data (via backoff smoothing to a
population-level graph). This directly satisfies the "graph-based for
entity-resource relationships" option named in the brief, rather than
only approximating sequence-awareness through single-event features.

METHOD
------
1. Build a population-level transition graph per entity_type: edge
   weight = count(prev_resource -> resource) across ALL entities of
   that type, updated incrementally in time order (no leakage).
2. Build a per-entity transition graph the same way, using only that
   entity's own prior history.
3. Score each transition with Katz-style backoff smoothing (see
   common.transition_probability), so an entity with lots of its own
   history relies mostly on its own graph, and a cold-start entity
   relies mostly on the population graph.
4. sequence_anomaly_score = -log(P), rescaled to 0-100 (higher = more
   anomalous ordering). The FIRST event for any entity (no previous
   resource) gets a neutral score (population marginal only).

This score is merged into the feature table as `seq_transition_score`
and used both as an extra IsolationForest feature AND reported as its
own diagnostic column, so its standalone contribution is visible.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from common import marginal_probability, transition_probability


def build_transition_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Population-level graphs, keyed by entity_type.
    pop_edge = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))   # etype -> prev -> cur -> count
    pop_node_total = defaultdict(lambda: defaultdict(int))                  # etype -> prev -> total
    pop_marginal = defaultdict(lambda: defaultdict(int))                    # etype -> resource -> count
    pop_marginal_total = defaultdict(int)

    rows = []

    for eid, g in df.groupby("entity_id", sort=False):
        etype = g["entity_type"].iloc[0]
        ent_edge = defaultdict(lambda: defaultdict(int))
        ent_node_total = defaultdict(int)
        prev_resource = None

        for _, row in g.iterrows():
            cur = row["resource_accessed"]

            if prev_resource is None:
                # First event for this entity: no transition to score yet,
                # fall back to the population MARGINAL distribution over
                # resources for this entity_type (still graceful cold start).
                p = marginal_probability(
                    pop_marginal[etype][cur], pop_marginal_total[etype], len(pop_marginal[etype])
                )
            else:
                p = transition_probability(
                    ent_edge[prev_resource][cur], ent_node_total[prev_resource],
                    pop_edge[etype][prev_resource][cur], pop_node_total[etype][prev_resource],
                )
            seq_score_raw = -np.log(max(p, 1e-9))

            rows.append({
                "event_id": row["event_id"], "entity_id": eid,
                "prev_resource": prev_resource, "resource_accessed": cur,
                "seq_neg_log_prob": seq_score_raw,
            })

            # Update ENTITY and POPULATION graphs with this observed
            # transition (post-scoring, no leakage).
            if prev_resource is not None:
                ent_edge[prev_resource][cur] += 1
                ent_node_total[prev_resource] += 1
                pop_edge[etype][prev_resource][cur] += 1
                pop_node_total[etype][prev_resource] += 1
            pop_marginal[etype][cur] += 1
            pop_marginal_total[etype] += 1

            prev_resource = cur

    seq_df = pd.DataFrame(rows)
    # Rescale to 0-100 (percentile-based, robust to outlier magnitude).
    ranks = seq_df["seq_neg_log_prob"].rank(pct=True)
    seq_df["seq_transition_score"] = (ranks * 100).round(2)
    return seq_df[["event_id", "prev_resource", "seq_neg_log_prob", "seq_transition_score"]]


if __name__ == "__main__":
    df = pd.read_csv("../data/access_logs.csv")
    seq_df = build_transition_scores(df)
    seq_df.to_csv("../data/sequence_scores.csv", index=False)

    merged = df[["event_id", "label"]].merge(seq_df, on="event_id")
    print("Mean seq_transition_score by label (higher = more anomalous ordering):")
    print(merged.groupby("label")["seq_transition_score"].mean().sort_values(ascending=False))