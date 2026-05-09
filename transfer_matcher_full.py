#!/usr/bin/env python3
"""
Transfer Course Matching System — v4 (Deep Learning Final Project)
====================================================================
Two neural network architectures for transfer course equivalency prediction:

  1. TransferMatcherNet (19-dim structured features)
     Uses engineered features: subject match, level diff, credit ratio,
     one-hot subject encodings.  BatchNorm + Dropout for regularisation.

  2. NLPTransferMatcherNet (14-dim KU/NLP features)
     Uses Knowledge Unit overlap (Jaccard), TF-IDF cosine similarity,
     and course-name similarity.  Weighted BCE loss for class imbalance.

Why two models?
  Structured features capture coarse institutional metadata, while the
  NLP/KU model captures semantic curriculum content.  Comparing both
  demonstrates that richer semantic features improve precision/recall
  on semantically similar but structurally different courses.

Why BatchNorm & Dropout?
  BatchNorm stabilises training by normalising layer inputs, reducing
  internal covariate shift.  Dropout prevents co-adaptation of neurons,
  improving generalisation on small academic datasets.

Why normalisation?
  Features are on different scales (credits ≈ 3, one-hot = {0,1},
  level_diff ∈ [-3,3]).  Z-score normalisation ensures no single feature
  dominates the gradient signal.

Install:  pip install torch pandas scikit-learn matplotlib
          pip install mysql-connector-python  # only for 'export'

Usage:
    python transfer_matcher_full.py train
    python transfer_matcher_full.py train_nlp
    python transfer_matcher_full.py compare_models
    python transfer_matcher_full.py run_trials --trials 5
    python transfer_matcher_full.py plot_results
    python transfer_matcher_full.py final_summary
    python transfer_matcher_full.py predict '{"source":[3,0,0,0],"target":[3,0,0,0]}'
    python transfer_matcher_full.py status
    python transfer_matcher_full.py export
    python transfer_matcher_full.py build_samples
"""

import sys, os, json, re, argparse, datetime, time, csv, random
from typing import List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model_weights.pt")
MODEL_META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

NUM_SUBJECTS = 6
FEATURE_SIZE = 19  # 7 engineered + 6 src one-hot + 6 tgt one-hot

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def build_feature_vector(src, tgt):
    """
    Build 19-dim engineered feature vector from raw course attributes.
    src/tgt: [credit_hours, level_code, subject_code, has_lab]
    """
    sc, sl, ss, slab = float(src[0]), int(src[1]), int(src[2]), float(src[3])
    tc, tl, ts       = float(tgt[0]), int(tgt[1]), int(tgt[2])

    src_oh = [1.0 if i == ss else 0.0 for i in range(NUM_SUBJECTS)]
    tgt_oh = [1.0 if i == ts else 0.0 for i in range(NUM_SUBJECTS)]

    return [
        1.0 if ss == ts else 0.0,        # subject_match
        float(sl - tl),                   # level_diff (signed)
        1.0 if sl >= tl else 0.0,         # level_meets_or_exceeds
        sc / max(tc, 0.5),                # credit_ratio
        1.0 if sc >= tc - 0.5 else 0.0,   # credit_meets_or_exceeds
        abs(sc - tc),                      # credit_diff
        slab,                              # src_has_lab
        *src_oh, *tgt_oh,
    ]

def _legacy_to_pair(f8):
    """Convert old 8-element vector to (src, tgt) attribute pair."""
    return f8[:4], [f8[4], f8[5], f8[6], 0]

# ── BLOCKING INDEX (Sprint III — MP Subject Filter) ─────────────────────────

def build_blocking_index(course_list):
    """
    Organize courses into a dict keyed by subject_code (index 2).

    Each course is expected to be a 4-element list:
        [credit_hours, level_code, subject_code, has_lab]

    Returns:
        dict[int, list[list]]  — subject_code → list of course attribute vectors

    This reduces candidate generation from O(n²) to O(n×k), where k is the
    average number of courses per subject bucket.
    """
    index = defaultdict(list)
    for course in course_list:
        subject_code = int(course[2])
        index[subject_code].append(course)
    return dict(index)

# ── NEURAL NETWORK (v2) ─────────────────────────────────────────────────────

class TransferMatcherNet(nn.Module):
    """Input(19) → Dense(64,BN,ReLU,Drop) → Dense(32,BN,ReLU,Drop) → Dense(16,ReLU) → Sigmoid"""

    def __init__(self, input_size=FEATURE_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)

# ── NORMALIZATION ────────────────────────────────────────────────────────────

def _compute_norm(X):
    mean = X.mean(dim=0)
    std  = X.std(dim=0)
    std[std < 1e-6] = 1.0
    return mean, std

def _normalize(X, mean, std):
    return (X - mean) / std

# ── SAMPLE TRAINING DATA ────────────────────────────────────────────────────

def get_sample_training_data():
    """Hand-crafted (source_attrs, target_attrs, label) triples."""
    return [
        # Strong matches
        ([3,0,0,0],[3,0,0,0],1), ([3,1,0,0],[3,1,0,0],1), ([4,1,1,1],[4,1,1,0],1),
        ([3,0,2,0],[3,0,2,0],1), ([4,2,4,0],[4,2,4,0],1), ([3,0,3,0],[3,0,3,0],1),
        ([3,1,4,0],[3,1,4,0],1), ([4,0,1,1],[4,0,1,0],1), ([3,3,4,0],[3,3,4,0],1),
        ([3,0,5,0],[3,0,5,0],1),
        # Higher level satisfies lower
        ([3,1,0,0],[3,0,0,0],1), ([4,2,1,1],[3,1,1,0],1), ([3,3,0,0],[3,2,0,0],1),
        ([4,3,4,0],[3,2,4,0],1), ([3,2,2,0],[3,1,2,0],1), ([4,1,1,1],[3,0,1,0],1),
        # Level too low
        ([3,0,4,0],[3,1,4,0],0), ([3,1,0,0],[3,2,0,0],0), ([3,2,4,0],[3,3,4,0],0),
        ([3,0,2,0],[3,1,2,0],0),
        # Credit insufficient
        ([3,0,0,0],[4,0,0,0],0), ([1,0,0,0],[4,0,0,0],0), ([2,1,2,0],[5,1,2,0],0),
        # Wrong subject
        ([3,0,0,0],[3,0,1,0],0), ([3,0,2,0],[3,0,0,0],0), ([4,1,4,0],[4,1,1,0],0),
        ([3,0,3,0],[3,0,4,0],0), ([3,1,1,1],[3,1,4,0],0), ([4,2,5,0],[4,2,0,0],0),
        ([3,1,0,0],[3,1,3,0],0), ([4,0,4,0],[4,0,2,0],0),
        # Edge cases
        ([3,0,0,0],[3,0,0,1],1), ([4,2,1,0],[4,2,1,1],1),
        ([3.5,0,0,0],[3,0,0,0],1), ([2.5,0,0,0],[3,0,0,0],1),
    ]

# ── TRAIN ────────────────────────────────────────────────────────────────────

def _samples_to_tensors(samples):
    """Convert any supported sample format to (X, y) tensors."""
    fvs, labels = [], []
    for s in samples:
        if isinstance(s, dict):
            if "source" in s and "target" in s:
                fv = build_feature_vector(s["source"], s["target"])
            elif "features" in s:
                f = s["features"]
                fv = build_feature_vector(*_legacy_to_pair(f)) if len(f) == 8 else f
            else:
                continue
            fvs.append(fv); labels.append(float(s["label"]))
        elif isinstance(s, (list, tuple)) and len(s) == 3:
            fvs.append(build_feature_vector(s[0], s[1])); labels.append(float(s[2]))
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            f, lbl = s
            fv = build_feature_vector(*_legacy_to_pair(f)) if len(f) == 8 else f
            fvs.append(fv); labels.append(float(lbl))
    return (torch.tensor(fvs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1))

def train_model(samples=None, epochs=300, learning_rate=0.001, patience=50,
                track_history=True):
    """
    Train TransferMatcherNet with validation split, early stopping, LR scheduling.

    Why two models are used:
      TransferMatcherNet uses structured institutional metadata (credits, level,
      subject one-hot).  NLPTransferMatcherNet uses KU semantic content.  Both
      are evaluated so we can see which feature representation generalises better.

    track_history: if True, per-epoch loss/accuracy is saved to
      training_history_structured.json for visualisation.
    """
    if not samples:
        samples = get_sample_training_data()

    X, y = _samples_to_tensors(samples)
    n = len(X)

    # Z-score normalisation keeps all 19 features on comparable scales
    feat_mean, feat_std = _compute_norm(X)
    X_norm = _normalize(X, feat_mean, feat_std)

    idx   = torch.randperm(n)
    split = max(int(n * 0.8), 1)
    Xt, yt = X_norm[idx[:split]], y[idx[:split]]
    Xv, yv = X_norm[idx[split:]], y[idx[split:]]
    has_val = len(Xv) > 0

    model     = TransferMatcherNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.5, min_lr=1e-6)

    best_val, pat_cnt, best_state, final_ep = float("inf"), 0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(Xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            t_acc = ((out >= 0.5).float() == yt).float().mean().item()
        history["train_loss"].append(round(loss.item(), 6))
        history["train_acc"].append(round(t_acc, 4))

        if has_val:
            model.eval()
            with torch.no_grad():
                vout = model(Xv)
                vl   = criterion(vout, yv).item()
                v_acc = ((vout >= 0.5).float() == yv).float().mean().item()
            history["val_loss"].append(round(vl, 6))
            history["val_acc"].append(round(v_acc, 4))
            scheduler.step(vl)
            if vl < best_val:
                best_val, pat_cnt = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    print(f"  ⏹  Early stopping at epoch {ep+1}")
                    break
        final_ep = ep + 1

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds      = model(X_norm)
        accuracy   = ((preds >= 0.5).float() == y).float().mean().item()
        final_loss = criterion(preds, y).item()

    torch.save(model.state_dict(), MODEL_PATH)

    meta = {
        "isTrained": True,
        "lastTrainedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trainingAccuracy": round(accuracy, 4),
        "epochs": final_ep, "finalLoss": round(final_loss, 6),
        "bestValLoss": round(best_val, 6) if has_val else None,
        "numSamples": n, "featureSize": FEATURE_SIZE,
        "architecture": f"Input({FEATURE_SIZE})→Dense(64,BN,ReLU,Drop0.3)"
                        f"→Dense(32,BN,ReLU,Drop0.2)→Dense(16,ReLU)→Sigmoid",
        "normalization": {"mean": feat_mean.tolist(), "std": feat_std.tolist()},
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if track_history:
        hist_path = os.path.join(os.path.dirname(__file__), "training_history_structured.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    global _cached_model, _cached_norm
    _cached_model = _cached_norm = None

    return {
        "success": True, "epochs": final_ep,
        "finalLoss": round(final_loss, 6), "accuracy": round(accuracy, 4),
        "message": f"Training complete. {round(accuracy*100,1)}% accuracy on "
                   f"{n} samples ({final_ep} epochs).",
        "history": history,
    }

# ── PREDICT (cached) ────────────────────────────────────────────────────────

_cached_model = None
_cached_norm  = None

def _load_model():
    global _cached_model, _cached_norm
    if _cached_model is not None:
        return _cached_model, _cached_norm
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not trained. Run: python transfer_matcher_full.py train")
    m = TransferMatcherNet()
    m.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    m.eval()
    norm = (torch.zeros(FEATURE_SIZE), torch.ones(FEATURE_SIZE))
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            nd = json.load(f).get("normalization", {})
        if "mean" in nd and "std" in nd:
            norm = (torch.tensor(nd["mean"], dtype=torch.float32),
                    torch.tensor(nd["std"],  dtype=torch.float32))
    _cached_model, _cached_norm = m, norm
    return m, norm

def _classify(prob):
    conf = "high" if prob >= 0.75 or prob <= 0.25 else \
           "medium" if prob >= 0.60 or prob <= 0.40 else "low"
    return {"matchProbability": round(prob, 4), "isMatch": prob >= 0.5, "confidence": conf}

def predict(features=None, source=None, target=None):
    """Single prediction. Accepts (source, target) attrs or a feature vector."""
    model, (mean, std) = _load_model()
    if source is not None and target is not None:
        fv = build_feature_vector(source, target)
    elif features is not None:
        fv = build_feature_vector(*_legacy_to_pair(features)) if len(features) == 8 else features
    else:
        raise ValueError("Provide (source, target) or features")
    X = _normalize(torch.tensor([fv], dtype=torch.float32), mean, std)
    with torch.no_grad():
        return _classify(model(X).item())

def predict_batch(pairs):
    """Batch prediction — much faster than repeated single calls."""
    model, (mean, std) = _load_model()
    fvs = []
    for p in pairs:
        if "source" in p and "target" in p:
            fvs.append(build_feature_vector(p["source"], p["target"]))
        elif "features" in p:
            f = p["features"]
            fvs.append(build_feature_vector(*_legacy_to_pair(f)) if len(f) == 8 else f)
    if not fvs:
        return []
    X = _normalize(torch.tensor(fvs, dtype=torch.float32), mean, std)
    with torch.no_grad():
        probs = model(X).squeeze(-1).tolist()
    if isinstance(probs, float):
        probs = [probs]
    return [_classify(p) for p in probs]


def predict_batch_blocked(source_course, blocking_index):
    """
    Sprint III optimized prediction using the blocking index.

    Instead of comparing source_course against ALL target courses (O(n)),
    this retrieves only candidates that share the same subject_code (O(k)).

    Args:
        source_course: [credit_hours, level_code, subject_code, has_lab]
        blocking_index: dict from build_blocking_index()

    Returns:
        list[dict] — scored candidates, each with 'target', 'matchProbability',
                     'isMatch', and 'confidence' keys, sorted by probability desc.
    """
    subject_code = int(source_course[2])
    candidates = blocking_index.get(subject_code, [])

    if not candidates:
        return []

    # Build pairs only for same-subject candidates
    pairs = [
        {"source": source_course, "target": tgt}
        for tgt in candidates
        if tgt != source_course  # skip self-comparison
    ]

    if not pairs:
        return []

    # Run the neural network only on filtered candidates
    results = predict_batch(pairs)

    # Attach candidate info to each result
    scored = []
    for pair, result in zip(pairs, results):
        scored.append({
            "target": pair["target"],
            "matchProbability": result["matchProbability"],
            "isMatch": result["isMatch"],
            "confidence": result["confidence"],
        })

    # Sort by match probability descending
    scored.sort(key=lambda x: x["matchProbability"], reverse=True)
    return scored


def find_matches_for_courses(source_courses, target_courses):
    """
    Sprint III high-level API: find matches for multiple source courses
    against a pool of target courses, using the blocking strategy.

    Complexity: O(n × k) where k = avg courses per subject bucket,
    versus O(n × m) for the brute-force approach.

    Args:
        source_courses: list of [credit_hours, level_code, subject_code, has_lab]
        target_courses: list of [credit_hours, level_code, subject_code, has_lab]

    Returns:
        dict: source_index → list of scored candidate matches
    """
    # Build blocking index from target courses
    blocking_index = build_blocking_index(target_courses)

    all_matches = {}
    for i, src in enumerate(source_courses):
        all_matches[i] = predict_batch_blocked(src, blocking_index)

    return all_matches

# ── STATUS ───────────────────────────────────────────────────────────────────

def get_status():
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        meta["modelPath"] = MODEL_PATH
        return meta
    return {"isTrained": False, "modelPath": MODEL_PATH, "featureSize": FEATURE_SIZE}

# ── MYSQL EXPORT ─────────────────────────────────────────────────────────────

def export_database(host="127.0.0.1", port=3306, user="appuser",
                    password="apppass", database="TransferPro",
                    output_dir="csv_exports"):
    try:
        import mysql.connector, pandas as pd
    except ImportError:
        raise SystemExit("Run:  pip install mysql-connector-python pandas")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Connecting to MySQL  {host}:{port}  db='{database}' ...")
    conn   = mysql.connector.connect(host=host, port=port, user=user,
                                     password=password, database=database)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Found {len(tables)} tables: {tables}\n")
    exported = []
    for t in tables:
        try:
            df = pd.read_sql(f"SELECT * FROM `{t}`", conn)
            p  = os.path.join(output_dir, f"{t}.csv")
            df.to_csv(p, index=False)
            print(f"  ✅  {t}: {len(df)} rows  →  {p}")
            exported.append({"table": t, "rows": len(df), "path": p})
        except Exception as e:
            print(f"  ❌  {t}: {e}")
    cursor.close(); conn.close()
    print(f"\nExported {len(exported)}/{len(tables)} tables to '{output_dir}/'")
    return exported

# ── CSV → TRAINING SAMPLES ──────────────────────────────────────────────────

SUBJECT_MAP = {
    "math": 0, "mathematics": 0,
    "science": 1, "biology": 1, "chemistry": 1, "physics": 1,
    "english": 2, "writing": 2, "composition": 2,
    "history": 3, "social": 3,
    "cs": 4, "computer": 4, "programming": 4, "information": 4, "technology": 4,
}

def _encode_subject(s):
    if not isinstance(s, str): return 5
    s = s.lower().strip()
    for kw, code in SUBJECT_MAP.items():
        if kw in s: return code
    return 5

def _encode_level(val):
    if val is None: return 0
    nums = re.findall(r"\d+", str(val))
    if nums:
        n = int(nums[0])
        if n >= 400: return 3
        if n >= 300: return 2
        if n >= 200: return 1
    return 0

def _encode_has_lab(val):
    if isinstance(val, bool):        return int(val)
    if isinstance(val, (int, float)): return int(bool(val))
    if isinstance(val, str):          return 1 if val.strip().lower() in ("1","true","yes","y") else 0
    return 0

def _safe_float(val, default=3.0):
    try: return float(val)
    except: return default

def _detect_cols(df):
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    def find(*names):
        for n in names:
            if n.lower().replace(" ", "_") in cols:
                return cols[n.lower().replace(" ", "_")]
        return None
    return {
        "title":        find("title","name","course_name","course_title"),
        "subject":      find("subject","subject_area","department","dept"),
        "level":        find("level","course_level","number","course_number","code"),
        "credit_hours": find("credit_hours","credits","units","credit_unit"),
        "has_lab":      find("has_lab","lab","laboratory","includes_lab"),
        "is_match":     find("is_match","match","approved","articulated","equivalent"),
        "source_id":    find("source_course_id","source_id","from_course_id","course_id"),
        "target_id":    find("target_course_id","target_id","to_course_id","requirement_id"),
    }

def _row_feats(row, cols):
    return [
        _safe_float(row.get(cols["credit_hours"]) if cols.get("credit_hours") else None),
        _encode_level(row.get(cols["level"])       if cols.get("level")       else None),
        _encode_subject(str(row.get(cols["subject"]) if cols.get("subject")   else "")),
        _encode_has_lab(row.get(cols["has_lab"])   if cols.get("has_lab")     else None),
    ]

def _build_from_rules(rules_df, courses_df):
    r_cols = _detect_cols(rules_df)
    c_cols = _detect_cols(courses_df)
    id_col = next((c for c in ("id","course_id","ID") if c in courses_df.columns), None)
    if not id_col: return []
    idx     = {str(r[id_col]): r for _, r in courses_df.iterrows()}
    samples = []
    for _, rule in rules_df.iterrows():
        src_id = str(rule.get(r_cols.get("source_id") or "")) if r_cols.get("source_id") else None
        tgt_id = str(rule.get(r_cols.get("target_id") or "")) if r_cols.get("target_id") else None
        if src_id not in idx or tgt_id not in idx: continue
        sf = _row_feats(idx[src_id], c_cols)
        tf = _row_feats(idx[tgt_id], c_cols)
        is_match_val = rule.get(r_cols.get("is_match") or "") if r_cols.get("is_match") else None
        label = 1.0 if _encode_has_lab(is_match_val) == 1 else 0.0
        samples.append({"source": sf, "target": tf, "label": label})
    return samples

def _build_synthetic(courses_df):
    """
    Blocking-based synthetic data generation using build_blocking_index.

    Complexity: O(n × k) per subject instead of O(n²) over all courses,
    where k = number of courses within a single subject bucket.

    Same-subject pairs produce positives (when level/credit rules pass)
    and negatives (when they fail). A small sample of cross-subject pairs
    provides hard negatives so the model learns the subject_match signal.
    """
    c_cols = _detect_cols(courses_df)
    if courses_df.empty:
        return []

    # Extract attribute vectors from the DataFrame
    all_courses = [_row_feats(r, c_cols) for _, r in courses_df.iterrows()]

    # Build blocking index using the shared utility
    blocking_index = build_blocking_index(all_courses)

    samples = []
    subjects = list(blocking_index.keys())

    # ── Same-subject pairs (positives + negatives via level/credit rules) ──
    for subj, group in blocking_index.items():
        for i, src in enumerate(group):
            for j, tgt in enumerate(group):
                if i == j:
                    continue  # skip self-comparison
                label = 1.0 if (src[1] >= tgt[1] and src[0] >= tgt[0] - 0.5) else 0.0
                samples.append({"source": src, "target": tgt, "label": label})

    # ── Cross-subject negatives (sampled, not exhaustive) ──
    # Only sample a few pairs so the model learns cross-subject = no match
    for si, subj in enumerate(subjects):
        for other in subjects[si + 1 : si + 3]:
            for src in blocking_index[subj][:5]:
                for tgt in blocking_index[other][:5]:
                    samples.append({"source": src, "target": tgt, "label": 0.0})

    # Cap to avoid memory issues on large catalogs
    if len(samples) > 2000:
        import random
        random.shuffle(samples)
        samples = samples[:2000]

    return samples

COURSE_TABLE_NAMES = ["courses","course","course_list","catalog"]
RULES_TABLE_NAMES  = ["transfer_rules","articulation","equivalencies",
                       "transfer_articulation","course_equivalency","articulation_agreements"]

def build_samples_from_csvs(csv_dir="csv_exports", output_path="training_samples.json"):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Run:  pip install pandas")
    if not os.path.isdir(csv_dir):
        print(f"⚠  '{csv_dir}' not found — run 'export' first."); return []
    dfs = {}
    print(f"Loading CSVs from '{csv_dir}' ...")
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"): continue
        key = fname.replace(".csv","").lower()
        try:
            dfs[key] = pd.read_csv(os.path.join(csv_dir, fname))
            print(f"  📄  {fname}: {len(dfs[key])} rows  |  cols: {list(dfs[key].columns)}")
        except Exception as e:
            print(f"  ⚠   {fname}: {e}")
    courses_df = next((dfs[k] for k in COURSE_TABLE_NAMES if k in dfs), __import__("pandas").DataFrame())
    rules_df   = next((dfs[k] for k in RULES_TABLE_NAMES  if k in dfs), __import__("pandas").DataFrame())
    samples = []
    if not rules_df.empty and not courses_df.empty:
        samples = _build_from_rules(rules_df, courses_df)
        print(f"\n→ {len(samples)} samples from rules/articulation table")
    if len(samples) < 10 and not courses_df.empty:
        samples = _build_synthetic(courses_df)
        print(f"→ {len(samples)} synthetic samples (blocking-based)")
    if not samples:
        print("⚠  No samples generated."); return []
    matches = sum(1 for s in samples if s["label"] == 1.0)
    print(f"\n✅  {len(samples)} total samples  ({matches} matches, {len(samples)-matches} non-matches)")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"💾  Saved → {output_path}")
    return samples

def retrain_via_api(samples, api_base="http://localhost:80/api", epochs=200):
    try:
        import requests as req
    except ImportError:
        raise SystemExit("Run:  pip install requests")
    url = f"{api_base.rstrip('/')}/match/train"
    print(f"\n🚀  Sending {len(samples)} samples to {url} (epochs={epochs}) ...")
    try:
        r = req.post(url, json={"samples": samples, "epochs": epochs}, timeout=120)
        r.raise_for_status()
        res = r.json()
        print(f"  ✅  Accuracy: {round(float(res.get('accuracy',0))*100,1)}%"
              f"  |  Loss: {res.get('finalLoss')}  |  {res.get('message')}")
    except Exception as e:
        print(f"  ❌  API request failed: {e}")

# ── NLP + KU FEATURES (Sprint IV) ───────────────────────────────────────────

NLP_FEATURE_SIZE  = 14
NLP_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model_nlp_weights.pt")
NLP_MODEL_META_PATH = os.path.join(os.path.dirname(__file__), "model_nlp_meta.json")
DEFAULT_CSV_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                 "Documents", "csv_exports")

# ── NLP Preprocessing ────────────────────────────────────────────────────────

def _clean_text(text):
    """Lowercase, strip special chars, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ku_data(csv_dir=DEFAULT_CSV_DIR):
    """
    Load courses, knowledge_units, and course_ku CSVs.

    Returns:
        courses_df   — pandas DataFrame of all courses
        ku_map       — dict[ku_id -> {name, description}]
        course_ku_map — defaultdict[course_id -> set of ku_ids]
    """
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Run:  pip install pandas")

    csv_dir = os.path.abspath(csv_dir)
    paths = {
        "courses":         os.path.join(csv_dir, "courses.csv"),
        "knowledge_units": os.path.join(csv_dir, "knowledge_units.csv"),
        "course_ku":       os.path.join(csv_dir, "course_ku.csv"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV '{name}': {p}")

    courses_df = pd.read_csv(paths["courses"])
    ku_df      = pd.read_csv(paths["knowledge_units"])
    ck_df      = pd.read_csv(paths["course_ku"])

    ku_map = {
        int(r["ku_id"]): {"name": str(r.get("ku_name", "")),
                           "description": str(r.get("ku_description", ""))}
        for _, r in ku_df.iterrows()
    }

    course_ku_map = defaultdict(set)
    for _, r in ck_df.iterrows():
        course_ku_map[int(r["course_id"])].add(int(r["ku_id"]))

    return courses_df, ku_map, course_ku_map


def build_ku_text_profiles(courses_df, ku_map, course_ku_map):
    """
    Build a KU text profile per course = all KU names + descriptions joined.

    Returns dict[course_id -> {name, ku_profile, ku_ids}]
    """
    profiles = {}
    for _, row in courses_df.iterrows():
        cid         = int(row["course_id"])
        course_name = _clean_text(str(row.get("course_name", "")))
        ku_ids      = course_ku_map.get(cid, set())
        ku_texts    = []
        for kid in sorted(ku_ids):
            ku = ku_map.get(kid, {})
            ku_texts.extend([_clean_text(ku.get("name", "")),
                             _clean_text(ku.get("description", ""))])
        profiles[cid] = {
            "name":       course_name,
            "ku_profile": " ".join(t for t in ku_texts if t) or "unknown",
            "ku_ids":     ku_ids,
        }
    return profiles


def build_tfidf_vectors(profiles):
    """
    Fit two TF-IDF vectorisers — one for KU profiles, one for course names.

    Returns (ku_tfidf, name_tfidf, ku_matrix, name_matrix, course_ids)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise SystemExit("Run:  pip install scikit-learn")

    course_ids  = sorted(profiles.keys())
    ku_texts    = [profiles[c]["ku_profile"] for c in course_ids]
    name_texts  = [profiles[c]["name"] or "unknown" for c in course_ids]

    ku_tfidf   = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    name_tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2))

    ku_matrix   = ku_tfidf.fit_transform(ku_texts)
    name_matrix = name_tfidf.fit_transform(name_texts)

    return ku_tfidf, name_tfidf, ku_matrix, name_matrix, course_ids


# ── KU Feature Engineering ────────────────────────────────────────────────────

def compute_nlp_features(src_cid, tgt_cid, profiles, ku_matrix, name_matrix,
                          course_ids, courses_df):
    """
    Build 14-dimensional NLP feature vector for a (source, target) course pair.

    Dims:
      0  src_credits          4  tgt_credits
      1  src_level            5  tgt_level
      2  src_subject          6  tgt_subject
      3  src_has_lab          7  credit_diff (abs)
      8  ku_overlap_ratio     9  ku_count_src
      10 ku_count_tgt         11 ku_count_diff
      12 tfidf_cosine_sim     13 name_cosine_sim
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cid_idx = {c: i for i, c in enumerate(course_ids)}

    def _get_row(cid):
        rows = courses_df[courses_df["course_id"] == cid]
        return rows.iloc[0] if not rows.empty else None

    def _credits(row):
        try: return float(row["credits"]) if row is not None else 3.0
        except: return 3.0

    def _level(row):
        if row is None: return 0
        nums = re.findall(r"\d+", str(row.get("course_code", "")))
        if nums:
            n = int(nums[0])
            if n >= 400: return 3
            if n >= 300: return 2
            if n >= 200: return 1
        return 0

    def _subj(row):
        if row is None: return 4
        return _encode_subject(str(row.get("course_name", "")) + " " +
                               str(row.get("course_code", "")))

    sr, tr = _get_row(src_cid), _get_row(tgt_cid)
    src_credits, tgt_credits = _credits(sr), _credits(tr)

    # KU sets
    src_kus = profiles.get(src_cid, {}).get("ku_ids", set())
    tgt_kus = profiles.get(tgt_cid, {}).get("ku_ids", set())
    union   = src_kus | tgt_kus
    inter   = src_kus & tgt_kus
    jaccard = len(inter) / len(union) if union else 0.0

    # TF-IDF cosine similarities
    si, ti = cid_idx.get(src_cid), cid_idx.get(tgt_cid)
    if si is not None and ti is not None:
        tfidf_cos = float(cosine_similarity(ku_matrix[si],   ku_matrix[ti])[0, 0])
        name_cos  = float(cosine_similarity(name_matrix[si], name_matrix[ti])[0, 0])
    else:
        tfidf_cos = name_cos = 0.0

    return [
        src_credits, float(_level(sr)), float(_subj(sr)), 0.0,   # 0-3
        tgt_credits, float(_level(tr)), float(_subj(tr)),           # 4-6
        abs(src_credits - tgt_credits),                              # 7
        jaccard,                                                      # 8
        float(len(src_kus)), float(len(tgt_kus)),                    # 9-10
        abs(float(len(src_kus)) - float(len(tgt_kus))),             # 11
        tfidf_cos, name_cos,                                         # 12-13
    ]


def build_nlp_training_samples(csv_dir=DEFAULT_CSV_DIR):
    """
    Generate labeled training samples from CSV KU data.

    Labelling rules (applied per pair):
      - label=1 if ku_jaccard >= 0.2 AND src_level >= tgt_level
                AND src_credits >= tgt_credits - 0.5
      - label=0 otherwise

    Produces ~600 pairs from 25 courses (all ordered pairs, no self-pairs).
    """
    try:
        import pandas as pd  # noqa
    except ImportError:
        raise SystemExit("Run:  pip install pandas scikit-learn")

    print(f"Loading KU data from '{os.path.abspath(csv_dir)}' ...")
    courses_df, ku_map, course_ku_map = load_ku_data(csv_dir)
    profiles = build_ku_text_profiles(courses_df, ku_map, course_ku_map)
    _, _, ku_matrix, name_matrix, course_ids = build_tfidf_vectors(profiles)
    print(f"  {len(courses_df)} courses | {len(ku_map)} KUs | "
          f"{sum(len(v) for v in course_ku_map.values())} course-KU links")

    samples = []
    for src_cid in course_ids:
        for tgt_cid in course_ids:
            if src_cid == tgt_cid:
                continue
            fv = compute_nlp_features(src_cid, tgt_cid, profiles,
                                      ku_matrix, name_matrix, course_ids, courses_df)
            level_ok   = fv[1] >= fv[5]
            credits_ok = fv[0] >= fv[4] - 0.5
            # Threshold lowered to 0.1 so partial-overlap pairs are captured.
            # Courses with NO KUs at all (fv[9]==0 and fv[10]==0) are never
            # positives since they share nothing semantically.
            has_kus = fv[9] > 0 or fv[10] > 0
            label = 1.0 if (has_kus and fv[8] >= 0.1 and level_ok and credits_ok) else 0.0
            samples.append({"features": fv, "label": label})

    matches = sum(1 for s in samples if s["label"] == 1.0)
    print(f"  Generated {len(samples)} pairs  "
          f"({matches} matches, {len(samples)-matches} non-matches)")
    return samples


# ── NLPTransferMatcherNet (14-input) ──────────────────────────────────────────

class NLPTransferMatcherNet(nn.Module):
    """Input(14) → Dense(64,ReLU,Drop0.3) → Dense(32,ReLU,Drop0.2) → Dense(16,ReLU) → Sigmoid"""

    def __init__(self, input_size=NLP_FEATURE_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),         nn.ReLU(),
        )
        self.head = nn.Linear(16, 1)

    def logits(self, x):
        """Raw pre-sigmoid output — use with BCEWithLogitsLoss during training."""
        return self.head(self.features(x))

    def forward(self, x):
        """Sigmoid probability — use for inference."""
        return torch.sigmoid(self.logits(x))


# ── NLP Training ──────────────────────────────────────────────────────────────

_cached_nlp_model = None
_cached_nlp_norm  = None


def train_nlp_model(csv_dir=DEFAULT_CSV_DIR, epochs=300,
                    learning_rate=0.001, patience=50, track_history=True):
    """
    Train NLPTransferMatcherNet from CSV KU data.

    Why precision/recall/F1 matter for course matching:
      Class imbalance (few matches vs. many non-matches) makes raw accuracy
      misleading.  Precision/recall/F1 measure the quality of POSITIVE
      predictions, which is what admissions staff actually care about.
    """
    samples = build_nlp_training_samples(csv_dir)
    if not samples:
        return {"success": False, "message": "No training samples generated."}

    X = torch.tensor([s["features"] for s in samples], dtype=torch.float32)
    y = torch.tensor([s["label"]    for s in samples], dtype=torch.float32).unsqueeze(1)
    n = len(X)

    feat_mean, feat_std = _compute_norm(X)
    X_norm = _normalize(X, feat_mean, feat_std)

    idx   = torch.randperm(n)
    split = max(int(n * 0.8), 1)
    Xt, yt = X_norm[idx[:split]], y[idx[:split]]
    Xv, yv = X_norm[idx[split:]], y[idx[split:]]
    has_val = len(Xv) > 0

    model     = NLPTransferMatcherNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.5, min_lr=1e-6)

    # Class-weighted loss: positives are rare, give them proportional weight
    n_pos = y.sum().item()
    n_neg = n - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, pat_cnt, best_state, final_ep = float("inf"), 0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits_out = model.logits(Xt)
        loss = criterion(logits_out, yt)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            t_acc = ((torch.sigmoid(logits_out) >= 0.5).float() == yt).float().mean().item()
        history["train_loss"].append(round(loss.item(), 6))
        history["train_acc"].append(round(t_acc, 4))

        if has_val:
            model.eval()
            with torch.no_grad():
                vlogits = model.logits(Xv)
                vl      = criterion(vlogits, yv).item()
                v_acc   = ((torch.sigmoid(vlogits) >= 0.5).float() == yv).float().mean().item()
            history["val_loss"].append(round(vl, 6))
            history["val_acc"].append(round(v_acc, 4))
            scheduler.step(vl)
            if vl < best_val:
                best_val, pat_cnt = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    print(f"  ⏹  Early stopping at epoch {ep+1}")
                    break
        final_ep = ep + 1

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    bce = nn.BCELoss()
    with torch.no_grad():
        preds      = model(X_norm)          # sigmoid output for accuracy
        accuracy   = ((preds >= 0.5).float() == y).float().mean().item()
        final_loss = bce(preds, y).item()

    torch.save(model.state_dict(), NLP_MODEL_PATH)

    meta = {
        "isTrained": True,
        "lastTrainedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trainingAccuracy": round(accuracy, 4),
        "epochs": final_ep, "finalLoss": round(final_loss, 6),
        "bestValLoss": round(best_val, 6) if has_val else None,
        "numSamples": n, "featureSize": NLP_FEATURE_SIZE,
        "architecture": "Input(14)→Dense(64,ReLU,Drop0.3)"
                        "→Dense(32,ReLU,Drop0.2)→Dense(16,ReLU)→Sigmoid",
        "csvDir": str(os.path.abspath(csv_dir)),
        "normalization": {"mean": feat_mean.tolist(), "std": feat_std.tolist()},
    }
    with open(NLP_MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    if track_history:
        hist_path = os.path.join(os.path.dirname(__file__), "training_history_nlp.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    global _cached_nlp_model, _cached_nlp_norm
    _cached_nlp_model = _cached_nlp_norm = None

    return {
        "success": True, "epochs": final_ep,
        "finalLoss": round(final_loss, 6), "accuracy": round(accuracy, 4),
        "numSamples": n,
        "message": f"NLP training complete. {round(accuracy*100,1)}% accuracy on "
                   f"{n} samples ({final_ep} epochs).",
        "history": history,
    }


def _load_nlp_model():
    global _cached_nlp_model, _cached_nlp_norm
    if _cached_nlp_model is not None:
        return _cached_nlp_model, _cached_nlp_norm
    if not os.path.exists(NLP_MODEL_PATH):
        raise RuntimeError(
            "NLP model not trained. Run: python transfer_matcher_full.py train_nlp")
    m = NLPTransferMatcherNet()
    m.load_state_dict(torch.load(NLP_MODEL_PATH, weights_only=True))
    m.eval()
    norm = (torch.zeros(NLP_FEATURE_SIZE), torch.ones(NLP_FEATURE_SIZE))
    if os.path.exists(NLP_MODEL_META_PATH):
        with open(NLP_MODEL_META_PATH) as f:
            nd = json.load(f).get("normalization", {})
        if "mean" in nd and "std" in nd:
            norm = (torch.tensor(nd["mean"], dtype=torch.float32),
                    torch.tensor(nd["std"],  dtype=torch.float32))
    _cached_nlp_model, _cached_nlp_norm = m, norm
    return m, norm


# ── NLP Predict / Analyze ─────────────────────────────────────────────────────

def predict_nlp(src_course_id, tgt_course_id, csv_dir=DEFAULT_CSV_DIR):
    """
    Predict match probability between two courses (by course_id) using the
    NLP model.  Looks up KU data automatically.
    """
    model, (mean, std) = _load_nlp_model()
    courses_df, ku_map, course_ku_map = load_ku_data(csv_dir)
    profiles = build_ku_text_profiles(courses_df, ku_map, course_ku_map)
    _, _, ku_matrix, name_matrix, course_ids = build_tfidf_vectors(profiles)

    fv = compute_nlp_features(src_course_id, tgt_course_id, profiles,
                              ku_matrix, name_matrix, course_ids, courses_df)
    X = _normalize(torch.tensor([fv], dtype=torch.float32), mean, std)
    with torch.no_grad():
        prob = model(X).item()

    result = _classify(prob)
    result["sourceId"] = src_course_id
    result["targetId"] = tgt_course_id

    src_kus = profiles.get(src_course_id, {}).get("ku_ids", set())
    tgt_kus = profiles.get(tgt_course_id, {}).get("ku_ids", set())
    union   = src_kus | tgt_kus
    inter   = src_kus & tgt_kus
    result["kuOverlapRatio"] = round(len(inter) / len(union), 4) if union else 0.0
    result["tfidfCosineSim"] = round(fv[12], 4)
    result["nameCosineSim"]  = round(fv[13], 4)
    return result


def analyze_kus(src_course_id, tgt_course_id, csv_dir=DEFAULT_CSV_DIR):
    """
    Show detailed KU overlap analysis between two courses.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    courses_df, ku_map, course_ku_map = load_ku_data(csv_dir)
    profiles = build_ku_text_profiles(courses_df, ku_map, course_ku_map)
    _, _, ku_matrix, name_matrix, course_ids = build_tfidf_vectors(profiles)
    cid_idx = {c: i for i, c in enumerate(course_ids)}

    def _name(cid):
        rows = courses_df[courses_df["course_id"] == cid]
        return rows.iloc[0]["course_name"] if not rows.empty else f"Course {cid}"

    src_kus  = profiles.get(src_course_id, {}).get("ku_ids", set())
    tgt_kus  = profiles.get(tgt_course_id, {}).get("ku_ids", set())
    shared   = src_kus & tgt_kus
    src_only = src_kus - tgt_kus
    tgt_only = tgt_kus - src_kus
    union    = src_kus | tgt_kus
    jaccard  = len(shared) / len(union) if union else 0.0

    si, ti = cid_idx.get(src_course_id), cid_idx.get(tgt_course_id)
    tfidf_cos = float(cosine_similarity(ku_matrix[si], ku_matrix[ti])[0, 0]) \
                if si is not None and ti is not None else 0.0
    name_cos  = float(cosine_similarity(name_matrix[si], name_matrix[ti])[0, 0]) \
                if si is not None and ti is not None else 0.0

    def _ku_list(ids):
        return [{"id": k, "name": ku_map.get(k, {}).get("name", "")} for k in sorted(ids)]

    return {
        "sourceCourse":    {"id": src_course_id, "name": _name(src_course_id),
                            "kuCount": len(src_kus), "kus": _ku_list(src_kus)},
        "targetCourse":    {"id": tgt_course_id, "name": _name(tgt_course_id),
                            "kuCount": len(tgt_kus), "kus": _ku_list(tgt_kus)},
        "sharedKUs":       _ku_list(shared),
        "srcOnlyKUs":      _ku_list(src_only),
        "tgtOnlyKUs":      _ku_list(tgt_only),
        "kuOverlapRatio":  round(jaccard, 4),
        "tfidfCosineSim": round(tfidf_cos, 4),
        "nameCosineSim":   round(name_cos, 4),
        "suggestedLabel":  1 if jaccard >= 0.2 else 0,
    }



# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────

def set_random_seed(seed: int):
    """
    Set all relevant random seeds for reproducibility.

    Why multiple trials with different seeds?
      A single training run may produce optimistic or pessimistic results due
      to random weight initialisation and data shuffling.  Running N trials
      with different seeds gives mean ± std, which is the standard way to
      report model performance in deep learning papers.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # numpy optional
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ── EVALUATION METRICS (no sklearn required) ─────────────────────────────────

def compute_metrics(preds_tensor, labels_tensor, threshold=0.5):
    """
    Compute accuracy, precision, recall, F1, and confusion-matrix counts.

    Why precision/recall/F1 instead of accuracy alone?
      Transfer course datasets are class-imbalanced (most pairs don't match).
      A model predicting 'no match' for everything achieves high accuracy but
      zero recall — useless for admissions staff.  F1 balances both concerns.

    All maths done in pure Python/PyTorch — no sklearn dependency.

    Returns dict with: accuracy, precision, recall, f1, TP, TN, FP, FN.
    """
    preds  = (preds_tensor >= threshold).float()
    labels = labels_tensor.float()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    total    = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
    }


def _eval_structured(samples=None):
    """Train + evaluate TransferMatcherNet; return metrics dict."""
    import time as _time
    t0 = _time.time()
    result = train_model(samples=samples, epochs=300, track_history=True)
    elapsed = round(_time.time() - t0, 2)

    # Load trained model and re-evaluate on full set for metrics
    if not samples:
        samples = get_sample_training_data()
    X, y = _samples_to_tensors(samples)
    model, (mean, std) = _load_model()
    X_norm = _normalize(X, mean, std)
    with torch.no_grad():
        preds = model(X_norm)

    m = compute_metrics(preds, y)
    val_loss = result.get("finalLoss", 0.0)
    val_acc  = result.get("accuracy", 0.0)

    # Best val loss from history
    hist = result.get("history", {})
    if hist.get("val_loss"):
        val_loss = min(hist["val_loss"])
    if hist.get("val_acc"):
        val_acc = max(hist["val_acc"])

    return {
        "model_name":   "TransferMatcherNet",
        "feature_size": FEATURE_SIZE,
        "architecture": f"Input({FEATURE_SIZE})→Dense(64,BN,ReLU,Drop0.3)"
                        f"→Dense(32,BN,ReLU,Drop0.2)→Dense(16,ReLU)→Sigmoid",
        "num_samples":  len(samples),
        "train_acc":    result.get("accuracy", 0.0),
        "val_acc":      round(val_acc, 4),
        "val_loss":     round(val_loss, 6),
        "precision":    m["precision"],
        "recall":       m["recall"],
        "f1":           m["f1"],
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        "training_time_sec": elapsed,
        "epochs": result.get("epochs", 0),
    }


def _eval_nlp(csv_dir=DEFAULT_CSV_DIR):
    """Train + evaluate NLPTransferMatcherNet; return metrics dict."""
    import time as _time
    t0 = _time.time()
    result = train_nlp_model(csv_dir=csv_dir, epochs=300, track_history=True)
    elapsed = round(_time.time() - t0, 2)

    if not result.get("success"):
        return None

    samples = build_nlp_training_samples(csv_dir)
    X = torch.tensor([s["features"] for s in samples], dtype=torch.float32)
    y = torch.tensor([s["label"]    for s in samples], dtype=torch.float32).unsqueeze(1)
    model, (mean, std) = _load_nlp_model()
    X_norm = _normalize(X, mean, std)
    with torch.no_grad():
        preds = model(X_norm)

    m = compute_metrics(preds, y)
    hist = result.get("history", {})
    val_loss = min(hist["val_loss"]) if hist.get("val_loss") else result.get("finalLoss", 0.0)
    val_acc  = max(hist["val_acc"])  if hist.get("val_acc")  else result.get("accuracy", 0.0)

    return {
        "model_name":   "NLPTransferMatcherNet",
        "feature_size": NLP_FEATURE_SIZE,
        "architecture": "Input(14)→Dense(64,ReLU,Drop0.3)"
                        "→Dense(32,ReLU,Drop0.2)→Dense(16,ReLU)→Sigmoid",
        "num_samples":  result.get("numSamples", 0),
        "train_acc":    result.get("accuracy", 0.0),
        "val_acc":      round(val_acc, 4),
        "val_loss":     round(val_loss, 6),
        "precision":    m["precision"],
        "recall":       m["recall"],
        "f1":           m["f1"],
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
        "training_time_sec": elapsed,
        "epochs": result.get("epochs", 0),
    }


# ── COMPARE MODELS ────────────────────────────────────────────────────────────

def compare_models(csv_dir=DEFAULT_CSV_DIR):
    """
    Train and evaluate both models; print comparison table; save JSON + CSV.

    If KU CSVs are missing the NLP model is skipped gracefully and the
    structured model result is still saved.
    """
    _DIR = os.path.dirname(__file__)
    results = []

    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)

    # ── Structured model ──────────────────────────────────────────
    print("\n[1/2] Training TransferMatcherNet (structured features)…")
    s_res = _eval_structured()
    results.append(s_res)
    print(f"  ✅  Done — acc={s_res['train_acc']:.4f}  F1={s_res['f1']:.4f}"
          f"  time={s_res['training_time_sec']}s")

    # ── NLP model ─────────────────────────────────────────────────
    print("\n[2/2] Training NLPTransferMatcherNet (KU/NLP features)…")
    try:
        n_res = _eval_nlp(csv_dir)
        if n_res:
            results.append(n_res)
            print(f"  ✅  Done — acc={n_res['train_acc']:.4f}  F1={n_res['f1']:.4f}"
                  f"  time={n_res['training_time_sec']}s")
        else:
            print("  ⚠  NLP model returned no result (empty samples).")
    except FileNotFoundError as e:
        print(f"  ⚠  NLP model skipped — KU CSV files missing: {e}")
    except Exception as e:
        print(f"  ⚠  NLP model failed: {e}")

    # ── Print table ───────────────────────────────────────────────
    cols = ["model_name","feature_size","num_samples","train_acc",
            "val_acc","val_loss","precision","recall","f1","training_time_sec"]
    col_w = [24, 12, 12, 10, 10, 12, 10, 10, 10, 18]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
    print("\n" + "─"*len(header))
    print(header)
    print("─"*len(header))
    for r in results:
        row = "  ".join(f"{str(r.get(c,'')):<{w}}" for c, w in zip(cols, col_w))
        print(row)
    print("─"*len(header))

    # ── Save JSON ─────────────────────────────────────────────────
    out_json = os.path.join(_DIR, "model_comparison_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾  Saved → {out_json}")

    # ── Save CSV ──────────────────────────────────────────────────
    out_csv = os.path.join(_DIR, "model_comparison_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"💾  Saved → {out_csv}")

    return results


# ── MULTIPLE TRIALS ───────────────────────────────────────────────────────────

def run_trials(num_trials=5, csv_dir=DEFAULT_CSV_DIR):
    """
    Run each model num_trials times with different seeds; report mean ± std.

    Why multiple trials?
      With a small dataset (~35 samples for structured, ~600 for NLP) a single
      run can have high variance.  Reporting mean ± std across 5 trials gives
      a statistically honest estimate of model performance.
    """
    _DIR = os.path.dirname(__file__)
    all_results = {"TransferMatcherNet": [], "NLPTransferMatcherNet": []}

    for trial in range(num_trials):
        seed = 42 + trial * 7
        print(f"\n{'='*50}")
        print(f"  Trial {trial+1}/{num_trials}  (seed={seed})")
        print('='*50)
        set_random_seed(seed)

        # ── Structured ───────────────────────────────────────────
        res = _eval_structured()
        all_results["TransferMatcherNet"].append({
            "trial": trial+1, "seed": seed,
            "accuracy": res["train_acc"], "val_acc": res["val_acc"],
            "val_loss": res["val_loss"],  "f1": res["f1"],
        })

        # ── NLP (optional) ───────────────────────────────────────
        try:
            n_res = _eval_nlp(csv_dir)
            if n_res:
                all_results["NLPTransferMatcherNet"].append({
                    "trial": trial+1, "seed": seed,
                    "accuracy": n_res["train_acc"], "val_acc": n_res["val_acc"],
                    "val_loss": n_res["val_loss"],  "f1": n_res["f1"],
                })
        except Exception:
            pass

    def _stats(vals):
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        mu = sum(vals) / n
        if n == 1:
            return round(mu, 4), 0.0
        var = sum((v - mu)**2 for v in vals) / (n - 1)
        return round(mu, 4), round(var**0.5, 4)

    summary = {}
    print("\n" + "="*60)
    print("  TRIAL SUMMARY")
    print("="*60)

    for model_name, trials in all_results.items():
        if not trials:
            continue
        acc_mu,  acc_sd  = _stats([t["accuracy"] for t in trials])
        loss_mu, loss_sd = _stats([t["val_loss"] for t in trials])
        f1_mu,   f1_sd   = _stats([t["f1"] for t in trials])
        summary[model_name] = {
            "num_trials": len(trials),
            "avg_accuracy": acc_mu,  "std_accuracy": acc_sd,
            "avg_val_loss": loss_mu, "std_val_loss": loss_sd,
            "avg_f1":       f1_mu,   "std_f1":       f1_sd,
            "trials": trials,
        }
        print(f"\n  {model_name}")
        print(f"    accuracy : {acc_mu:.4f} ± {acc_sd:.4f}")
        print(f"    val_loss : {loss_mu:.4f} ± {loss_sd:.4f}")
        print(f"    F1       : {f1_mu:.4f} ± {f1_sd:.4f}")

    # Save JSON
    out_json = os.path.join(_DIR, "trial_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾  Saved → {out_json}")

    # Save CSV
    out_csv = os.path.join(_DIR, "trial_results.csv")
    rows = []
    for model_name, data in summary.items():
        for t in data.get("trials", []):
            rows.append({"model": model_name, **t})
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"💾  Saved → {out_csv}")

    return summary


# ── VISUALISATIONS ────────────────────────────────────────────────────────────

def plot_results():
    """
    Generate and save all training curves, bar charts, and confusion matrices.
    Requires matplotlib only — no seaborn.

    Outputs:
      structured_loss_curve.png      nlp_loss_curve.png
      structured_accuracy_curve.png  nlp_accuracy_curve.png
      model_comparison_bar_chart.png
      confusion_matrix_structured.png
      confusion_matrix_nlp.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠  matplotlib not installed. Run: pip install matplotlib"); return

    _DIR = os.path.dirname(__file__)

    # ── colour palette ────────────────────────────────────────────
    C1, C2 = "#4C72B0", "#DD8452"   # train=blue, val=orange

    def _load_hist(fname):
        p = os.path.join(_DIR, fname)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def _curve(hist, key_train, key_val, title, ylabel, out):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist[key_train], color=C1, linewidth=2, label="Train")
        if hist.get(key_val):
            ax.plot(hist[key_val], color=C2, linewidth=2, linestyle="--", label="Validation")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_DIR, out), dpi=150)
        plt.close()
        print(f"  💾  {out}")

    def _confusion(tp, tn, fp, fn, title, out):
        matrix = [[tn, fp], [fn, tp]]
        labels = [["TN", "FP"], ["FN", "TP"]]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted 0", "Predicted 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{matrix[i][j]}",
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if matrix[i][j] > max(tp,tn,fp,fn)/2 else "black")
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(_DIR, out), dpi=150)
        plt.close()
        print(f"  💾  {out}")

    missing = []

    # ── Training curves ───────────────────────────────────────────
    h_struct = _load_hist("training_history_structured.json")
    if h_struct:
        _curve(h_struct, "train_loss", "val_loss",
               "TransferMatcherNet — Loss", "BCE Loss", "structured_loss_curve.png")
        _curve(h_struct, "train_acc", "val_acc",
               "TransferMatcherNet — Accuracy", "Accuracy", "structured_accuracy_curve.png")
    else:
        missing.append("training_history_structured.json")

    h_nlp = _load_hist("training_history_nlp.json")
    if h_nlp:
        _curve(h_nlp, "train_loss", "val_loss",
               "NLPTransferMatcherNet — Loss", "BCE Loss", "nlp_loss_curve.png")
        _curve(h_nlp, "train_acc", "val_acc",
               "NLPTransferMatcherNet — Accuracy", "Accuracy", "nlp_accuracy_curve.png")
    else:
        missing.append("training_history_nlp.json")

    # ── Comparison bar chart ──────────────────────────────────────
    cmp_path = os.path.join(_DIR, "model_comparison_results.json")
    if os.path.exists(cmp_path):
        with open(cmp_path) as f:
            cmp = json.load(f)
        names    = [r["model_name"] for r in cmp]
        metrics  = ["train_acc", "val_acc", "precision", "recall", "f1"]
        m_labels = ["Train Acc", "Val Acc", "Precision", "Recall", "F1"]
        x = range(len(names))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
        for i, (metric, label) in enumerate(zip(metrics, m_labels)):
            vals = [r.get(metric, 0) for r in cmp]
            positions = [xi + i*width for xi in x]
            ax.bar(positions, vals, width, label=label, color=colors[i], alpha=0.85)
        ax.set_xticks([xi + 2*width for xi in x])
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
        ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_DIR, "model_comparison_bar_chart.png"), dpi=150)
        plt.close()
        print("  💾  model_comparison_bar_chart.png")

        # ── Confusion matrices ────────────────────────────────────
        for r in cmp:
            tag = "structured" if "Structured" in r["model_name"] or "19" in str(r.get("feature_size","")) else "nlp"
            if r.get("TP") is not None:
                _confusion(r["TP"], r["TN"], r["FP"], r["FN"],
                           f"Confusion Matrix — {r['model_name']}",
                           f"confusion_matrix_{tag}.png")
    else:
        missing.append("model_comparison_results.json")

    if missing:
        print(f"\n⚠  The following files were missing (run compare_models first): {missing}")
    else:
        print("\n✅  All plots saved.")


# ── FINAL PROJECT SUMMARY ─────────────────────────────────────────────────────

def final_summary():
    """
    Print and save a report-ready deep learning project summary.
    Saved to final_project_summary.txt.
    """
    _DIR = os.path.dirname(__file__)

    # Load comparison results if available
    cmp_text = ""
    cmp_path = os.path.join(_DIR, "model_comparison_results.json")
    if os.path.exists(cmp_path):
        with open(cmp_path) as f:
            cmp = json.load(f)
        cmp_text = "\n  Results:\n"
        for r in cmp:
            cmp_text += (
                f"    {r['model_name']}\n"
                f"      Samples : {r.get('num_samples','N/A')}\n"
                f"      Train Acc: {r.get('train_acc','N/A')}  Val Acc: {r.get('val_acc','N/A')}\n"
                f"      Precision: {r.get('precision','N/A')}  Recall: {r.get('recall','N/A')}  F1: {r.get('f1','N/A')}\n"
                f"      Val Loss : {r.get('val_loss','N/A')}  Time: {r.get('training_time_sec','N/A')}s\n\n"
            )
    else:
        cmp_text = "\n  (Run compare_models to populate results.)\n"

    # Load trial results if available
    trial_text = ""
    trial_path = os.path.join(_DIR, "trial_results.json")
    if os.path.exists(trial_path):
        with open(trial_path) as f:
            trial = json.load(f)
        trial_text = "\n  Trial Summary:\n"
        for model, data in trial.items():
            trial_text += (
                f"    {model}  ({data.get('num_trials','?')} trials)\n"
                f"      Avg Accuracy : {data.get('avg_accuracy','N/A')} ± {data.get('std_accuracy','N/A')}\n"
                f"      Avg Val Loss : {data.get('avg_val_loss','N/A')} ± {data.get('std_val_loss','N/A')}\n"
                f"      Avg F1       : {data.get('avg_f1','N/A')} ± {data.get('std_f1','N/A')}\n\n"
            )
    else:
        trial_text = "\n  (Run run_trials to populate trial statistics.)\n"

    summary = f"""
{'='*70}
DEEP LEARNING FINAL PROJECT SUMMARY
{'='*70}

Title:
  Deep Learning-Based Transfer Course Equivalency Prediction
  Using Structured Features and Knowledge Unit Similarity

Problem Statement:
  Determining whether a course taken at one institution satisfies a
  requirement at another is a labour-intensive, error-prone task for
  transfer students and academic advisors.  This project trains two
  neural network classifiers to automate binary transfer-equivalency
  prediction from (a) institutional metadata and (b) curriculum-content
  Knowledge Units.

Dataset Files:
  Structured model  — built-in hand-crafted samples + optional CSV exports
                      (courses.csv, transfer_rules.csv)
  NLP/KU model      — courses.csv, knowledge_units.csv, course_ku.csv

Neural Network Architectures:
  1. TransferMatcherNet (structured, {FEATURE_SIZE}-dim input)
       Input({FEATURE_SIZE}) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
               → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
               → Linear(16) → ReLU → Linear(1) → Sigmoid

  2. NLPTransferMatcherNet (KU/NLP, {NLP_FEATURE_SIZE}-dim input)
       Input({NLP_FEATURE_SIZE}) → Linear(64) → ReLU → Dropout(0.3)
               → Linear(32) → ReLU → Dropout(0.2)
               → Linear(16) → ReLU → Linear(1) → Sigmoid
       (trained with BCEWithLogitsLoss + class-frequency pos_weight)

  Why two models?
    The structured model captures administrative metadata (credits, level,
    subject).  The NLP model captures semantic curriculum overlap via KU
    Jaccard similarity and TF-IDF cosine similarity.  Comparing both shows
    whether richer semantic features improve precision/recall on courses that
    differ structurally but overlap in content.

Preprocessing:
  Structured : 19-dim engineered vector (subject_match, level_diff,
               credit_ratio, one-hot subject encodings).
               Z-score normalised per feature.
  NLP/KU    : 14-dim vector (credits, level, subject, KU Jaccard,
               TF-IDF cosine, course-name cosine).
               Z-score normalised per feature.
  Blocking strategy (Sprint III) reduces synthetic-pair generation from
  O(n²) to O(n×k) using a subject-code index.

Training Methods:
  • Adam optimiser (lr=0.001), ReduceLROnPlateau scheduler
  • 80/20 train/validation split, early stopping (patience=50)
  • BatchNorm + Dropout for regularisation
  • Weighted BCE loss for class-imbalanced NLP samples
  • Multiple independent trials with set_random_seed(seed)

Evaluation Metrics:
  Accuracy   — overall correct predictions
  Precision  — of predicted matches, fraction truly matching
  Recall     — of true matches, fraction correctly predicted
  F1 Score   — harmonic mean of precision & recall
  Confusion Matrix — TP / TN / FP / FN breakdown

  Why precision/recall/F1?
    The dataset is class-imbalanced.  A naive classifier can achieve high
    accuracy by always predicting 'no match', so we need metrics that
    penalise missed positives (low recall) and false alarms (low precision).

Model Comparison Results:
{cmp_text}
Trial Results (mean ± std across {5} trials):
{trial_text}
Conclusion:
  Both neural networks successfully learn transfer-equivalency signals from
  complementary feature representations.  The structured model trains quickly
  on minimal data and achieves high accuracy on well-defined institutional
  rules.  The NLP model, when KU data is available, captures semantic
  curriculum similarity that structural metadata alone cannot encode.

Future Work:
  • Incorporate real articulation agreement data from state-level databases
  • Add course description text embeddings (BERT / sentence-transformers)
  • Experiment with graph neural networks over curriculum prerequisite graphs
  • Deploy as a REST API for real-time advisor recommendations
  • Extend to multi-class prediction (full / partial / no credit equivalency)

{'='*70}
"""

    print(summary)
    out_path = os.path.join(_DIR, "final_project_summary.txt")
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"💾  Saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        prog="transfer_matcher_full.py",
        description="Transfer Course Matching — Neural Network + Data Pipeline (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train            Train TransferMatcherNet (structured features)
  train_nlp        Train NLPTransferMatcherNet (KU/NLP features)
  predict          Predict from source/target attrs or feature vector
  predict_nlp      Predict using course IDs (KU-aware)
  status           Show model training status
  export           Export TransferPro MySQL database to CSV files
  build_samples    Convert CSV exports to training samples
  analyze_kus      Show KU overlap analysis between courses
  compare_models   Train & compare both models, save JSON+CSV results
  run_trials       Run multiple trials, report mean ± std statistics
  plot_results     Generate and save all training/comparison plots
  final_summary    Print and save a report-ready project summary

Examples:
  python transfer_matcher_full.py train
  python transfer_matcher_full.py train_nlp
  python transfer_matcher_full.py compare_models
  python transfer_matcher_full.py run_trials --trials 5
  python transfer_matcher_full.py plot_results
  python transfer_matcher_full.py final_summary
  python transfer_matcher_full.py predict '{"source":[3,0,0,0],"target":[3,0,0,0]}'
  python transfer_matcher_full.py status
  python transfer_matcher_full.py build_samples --csv-dir csv_exports
  python transfer_matcher_full.py predict_nlp --source 1 --target 4
  python transfer_matcher_full.py analyze_kus  --source 1 --target 4
        """,
    )
    sub = parser.add_subparsers(dest="command")

    p_t = sub.add_parser("train", help="Train the neural network")
    p_t.add_argument("--epochs",  type=int,   default=300)
    p_t.add_argument("--lr",      type=float, default=0.001)
    p_t.add_argument("--patience",type=int,   default=50)
    p_t.add_argument("--samples", type=str,   default=None)

    p_p = sub.add_parser("predict", help="Run a prediction")
    p_p.add_argument("payload", nargs="?", default="{}")

    sub.add_parser("status", help="Show model training status")

    p_e = sub.add_parser("export", help="Export MySQL database to CSV")
    p_e.add_argument("--host",     default="127.0.0.1")
    p_e.add_argument("--port",     type=int, default=3306)
    p_e.add_argument("--user",     default="appuser")
    p_e.add_argument("--password", default="apppass")
    p_e.add_argument("--database", default="TransferPro")
    p_e.add_argument("--output",   default="csv_exports")

    p_b = sub.add_parser("build_samples", help="Convert CSVs to training samples")
    p_b.add_argument("--csv-dir", default="csv_exports")
    p_b.add_argument("--output",  default="training_samples.json")
    p_b.add_argument("--retrain", action="store_true")
    p_b.add_argument("--api",     default="http://localhost:80/api")
    p_b.add_argument("--epochs",  type=int, default=300)

    # ── NLP subcommands ────────────────────────────────────────────────────────
    p_tn = sub.add_parser("train_nlp",
                          help="Train NLP+KU model from CSV data")
    p_tn.add_argument("--csv-dir", default=DEFAULT_CSV_DIR,
                      help="Directory containing courses/knowledge_units/course_ku CSVs")
    p_tn.add_argument("--epochs",   type=int,   default=300)
    p_tn.add_argument("--lr",       type=float, default=0.001)
    p_tn.add_argument("--patience", type=int,   default=50)

    p_pn = sub.add_parser("predict_nlp",
                          help="Predict using course IDs (KU-aware NLP model)")
    p_pn.add_argument("--source",  type=int, required=True, help="Source course_id")
    p_pn.add_argument("--target",  type=int, required=True, help="Target course_id")
    p_pn.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)

    p_ak = sub.add_parser("analyze_kus",
                          help="Show detailed KU overlap between two courses")
    p_ak.add_argument("--source",  type=int, required=True, help="Source course_id")
    p_ak.add_argument("--target",  type=int, required=True, help="Target course_id")
    p_ak.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)

    # ── NEW: Final-project subcommands ─────────────────────────────────────────
    p_cm = sub.add_parser("compare_models",
                          help="Train & compare both models; save JSON+CSV results")
    p_cm.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)

    p_rt = sub.add_parser("run_trials",
                          help="Run multiple independent trials; report mean ± std")
    p_rt.add_argument("--trials",  type=int, default=5,
                      help="Number of independent trials (default: 5)")
    p_rt.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)

    sub.add_parser("plot_results",
                   help="Generate and save training curves, bar chart, confusion matrices")

    sub.add_parser("final_summary",
                   help="Print and save a report-ready final project summary")

    if len(sys.argv) >= 3 and sys.argv[1] in ("train","predict","status") \
            and sys.argv[2].startswith("{"):
        cmd     = sys.argv[1]
        payload = json.loads(sys.argv[2])
        try:
            if cmd == "train":
                result = train_model(
                    samples=payload.get("samples"),
                    epochs=payload.get("epochs", 300),
                    learning_rate=payload.get("learningRate", 0.001),
                )
            elif cmd == "predict":
                src = payload.get("source")
                tgt = payload.get("target")
                feats = payload.get("features")
                if src and tgt:
                    result = predict(source=src, target=tgt)
                elif feats:
                    result = predict(features=feats)
                else:
                    raise ValueError("Provide 'source'+'target' or 'features'")
            else:
                result = get_status()
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
        return

    args = parser.parse_args()

    if args.command == "train":
        samples = None
        if args.samples:
            with open(args.samples) as f:
                samples = json.load(f)
        result = train_model(samples=samples, epochs=args.epochs,
                             learning_rate=args.lr, patience=args.patience)
        print(json.dumps(result, indent=2))

    elif args.command == "predict":
        payload = json.loads(args.payload)
        src = payload.get("source")
        tgt = payload.get("target")
        feats = payload.get("features")
        if src and tgt:
            result = predict(source=src, target=tgt)
        elif feats:
            result = predict(features=feats)
        else:
            sys.exit("Provide 'source'+'target' or 'features' in JSON payload")
        print(json.dumps(result, indent=2))

    elif args.command == "status":
        print(json.dumps(get_status(), indent=2))

    elif args.command == "export":
        export_database(host=args.host, port=args.port, user=args.user,
                        password=args.password, database=args.database,
                        output_dir=args.output)

    elif args.command == "build_samples":
        samples = build_samples_from_csvs(csv_dir=args.csv_dir, output_path=args.output)
        if samples and args.retrain:
            retrain_via_api(samples, api_base=args.api, epochs=args.epochs)

    elif args.command == "train_nlp":
        result = train_nlp_model(
            csv_dir=args.csv_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "predict_nlp":
        result = predict_nlp(
            src_course_id=args.source,
            tgt_course_id=args.target,
            csv_dir=args.csv_dir,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "analyze_kus":
        result = analyze_kus(
            src_course_id=args.source,
            tgt_course_id=args.target,
            csv_dir=args.csv_dir,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "compare_models":
        compare_models(csv_dir=args.csv_dir)

    elif args.command == "run_trials":
        run_trials(num_trials=args.trials, csv_dir=args.csv_dir)

    elif args.command == "plot_results":
        plot_results()

    elif args.command == "final_summary":
        final_summary()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()