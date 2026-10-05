# AI-Powered Behavioral Anomaly Detection for Cybersecurity

A complete, working prototype: synthetic access-log generator → per-entity
baseline profiler → **graph-based entity-resource sequence model** →
unsupervised detector → anomaly classifier → SHAP explainability → analyst
dashboard. Every number in the report and slide deck comes from an actual,
fully reproducible run of the code in this folder (seed=42, single-threaded
training for determinism) — not estimates.

## Files in this delivery

| File | What it is |
|---|---|
| `src/generate_data.py` | Synthetic access-log generator — documented behavioral assumptions + 7 injected attack patterns |
| `src/features.py` | Per-entity baseline profiling + behavioral-deviation features (with cold-start handling) |
| `src/sequence_model.py` | **Graph-based sequence model** — per-entity resource-transition graph with population-level backoff smoothing; this is what makes the detector genuinely order/sequence-aware |
| `src/train_models.py` | IsolationForest (unsupervised detector, 15 features incl. the sequence score) + RandomForest (anomaly classifier) + SHAP explainer |
| `src/build_alert_queue.py` | Builds the ranked, explained alert queue that feeds the dashboard |
| `src/behavior_state.py` | Incremental (online) version of the baseline profiler + sequence model — powers the API |
| `src/api.py` | **Near-real-time scoring API** (FastAPI) — scores one new event at a time, not just a batch |
| `dashboard.html` | **Open this in a browser** — the analyst-facing console (ranked queue, risk scores, contributing factors, entity history) |
| `report.docx` | Full write-up: assumptions, architecture, metrics (including the two weak classifier categories, reported honestly), limitations |
| `presentation.pptx` | 12-slide deck covering the same material for a presentation/demo |
| `data/`, `alert_queue.json`, `*.json` | Underlying data and metrics, for inspection or re-use |

## Headline results (final, reproducible run)

- 45,580 synthetic events, 300 entities, 30 days, 3.6% labeled anomalous
- Detector ROC-AUC: **0.943** (unsupervised — trained without ever seeing the label column)
- Precision @ top 5% flagged: **54.8%** · Recall @ top 5% flagged: **76.6%**
- Classifier weighted accuracy: **98.4%**, with `brute_force`, `device_spoofing`, `credential_stuffing`, and `lateral_movement` all at or near perfect F1
- Two categories are honestly reported as weak: `insider_drift` (by design — it's the brief's own ambiguous edge case) and `impossible_travel` (too few examples, 72 total)

## How to run this on your local computer

### 1. Prerequisites
- **Python 3.10+** (check with `python3 --version`)
- **Node.js 18+** only if you want to regenerate `report.docx` / `presentation.pptx` yourself (optional — they're already included, pre-built)
- No GPU, no PyTorch, no heavy dependencies — everything is NumPy/pandas/scikit-learn

### 2. Set up a clean environment (recommended)
```bash
cd anomaly_detection_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install numpy pandas scikit-learn faker shap joblib
```

### 4. Run the full pipeline, in order
```bash
cd src

# 1. Generate the synthetic corpus (~45k events, takes ~10-20 seconds)
python3 generate_data.py --n-entities 300 --days 30 --out ../data/access_logs.csv

# 2. Build behavioral-deviation + graph-based sequence features (~1-2 min)
python3 features.py

# 3. Train the detector + classifier, print metrics, save models (~1-2 min)
python3 train_models.py

# 4. Build the ranked, explained alert queue for the dashboard
python3 build_alert_queue.py
```

You should see the same headline numbers printed to your terminal as listed
above (ROC-AUC 0.943, etc.) — training is deterministic (fixed seed,
single-threaded) so a fresh run should match exactly.

### 5. Run the near-real-time scoring API (optional but recommended)
This is what actually makes the system "near real-time" rather than a
batch script — it loads the trained models once and scores new events
one at a time over HTTP.
```bash
pip install fastapi uvicorn
cd src
uvicorn api:app --port 8000
```
On startup it replays the historical corpus once to warm-start every
entity's running profile (~20-30 seconds for the included 45k-event
corpus) — wait for `Application startup complete` in the terminal
before sending requests. Then, in another terminal:
```bash
# score a single new event
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "entity_id": "user::lisasolis_ca68", "entity_type": "user",
  "timestamp": "2026-07-01T16:40:00",
  "resource_accessed": "/admin/user_mgmt", "session_duration_min": 45.0,
  "auth_success": true, "geo_lat": 55.7558, "geo_lon": 37.6173,
  "device_os": "Linux 5.15", "device_mac": "aa:bb:cc:dd:ee:ff",
  "command_sequence": ["whoami", "sudo -l", "net user", "scp", "curl"]
}'

# see the highest-risk events scored so far
curl http://127.0.0.1:8000/alerts/top

# check an entity's current running profile
curl http://127.0.0.1:8000/entity/user::lisasolis_ca68
```
Or open `http://127.0.0.1:8000/docs` for interactive Swagger docs where
you can try requests directly in the browser.

**Send events in chronological order per entity** — the API tracks each
entity's running profile incrementally and flags (`out_of_order: true`)
if a timestamp arrives earlier than that entity's last-seen event,
rather than silently producing a nonsensical result.

**Honest scope note**: this is a single-process demo with in-memory
state that resets on restart, no authentication, and an approximated
IP-fan-out feature. See report.docx Section 10 for the full writeup,
including two real bugs found while testing this API end-to-end and how
they were fixed — not just the clean success case.

### 6. View the dashboard
The included `dashboard.html` already has data baked in and works standalone —
just double-click it or open it in any browser. No server needed.

If you want the dashboard to reflect a *new* run instead of the included one,
regenerate its embedded data after step 4:
```bash
cd ../src
python3 -c "
import pandas as pd, json
alerts = json.load(open('../outputs/alert_queue.json'))
entities = list({a['entity_id'] for a in alerts})
df = pd.read_csv('../data/access_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
history = {eid: df[df.entity_id==eid].sort_values('timestamp').tail(15)[
    ['timestamp','resource_accessed','geo_location','auth_success','session_duration_min','label']
].assign(timestamp=lambda x: x['timestamp'].astype(str)).to_dict('records') for eid in entities}
json.dump(history, open('../outputs/entity_history.json','w'), default=str)
"
# then merge alert_queue.json + alert_summary.json + detection_metrics.json +
# classification_report.json + entity_history.json into one JSON object and
# paste it into dashboard.html's <script id="data-blob"> tag, replacing the
# existing contents.
```
(This manual merge step exists because the dashboard is a single static file
with no backend — ask me to re-automate it if you want a one-command script.)

### 7. Regenerating report.docx / presentation.pptx (optional)
`build_scripts/build_report.js` and `build_scripts/build_deck.js` are
included — they use the `docx` and `pptxgenjs` npm packages:
```bash
npm install docx pptxgenjs
node build_scripts/build_report.js       # writes report.docx
node build_scripts/build_deck.js         # writes presentation.pptx
```
You only need this if you want to edit the numbers/wording yourself; the
included .docx/.pptx are already final and match the pipeline's output.

## Honest limitations (see report.docx Section 9 for full detail)

This is a synthetic-data prototype. Rule-based injected attacks are more
learnable than real adversarial behavior, so real-world precision/recall
would likely be lower. The sequence-awareness here is a graph-based
transition model (not a trained deep sequence encoder like an LSTM/GRU/
Transformer) — a deliberate choice given the corpus size (45k events,
under 1,700 anomalies would not support training a deep model without
overfitting), documented as a tradeoff rather than a shortcut taken
silently. Concept-drift handling and streaming/online scoring are not
fully solved here, only partially addressed and explicitly flagged.
`insider_drift` and `impossible_travel` remain the two weak spots in
per-category classification — both are explained, not hidden, in
Section 6 of the report.

# Anomaly_detection
