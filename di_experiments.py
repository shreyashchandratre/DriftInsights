#!/usr/bin/env python3
"""
Drift-Insights: clean, reproducible evaluation harness.

Protocol (identical for every stream):
  * M0 is trained on the first `train_end` samples (all pre-drift).
  * The remaining samples are streamed; ADWIN (delta=0.002) monitors the 0/1 error.
  * On an alarm at index a (labels up to a are available):
      - Explanation shift: W_after = last h samples, W_before = the h before that,
        h = min(200, (a+1)//2); mean|SHAP| per window; dE_i, dE_total (L2 norm).
      - Candidate adaptations are built from data available at a only:
          reweight         : continue boosting on last 1000 samples, exp. decay weights (gamma=0.995)
          incremental      : continue boosting on last 500 samples
          full_history     : retrain from scratch on ALL accumulated samples
          full_recent      : retrain from scratch on last 500 samples
      - Each candidate (and the unadapted model) is scored on the NEXT H=500 samples,
        which no candidate has seen. Gain = candidate acc - unadapted acc.
      - The pipeline continues with the strategy selected by the severity policy:
          dE<0.05 none | <0.15 reweight | <0.30 incremental | else full_history
  * Detection: an alarm in (d, d+2000] is the detection of drift d (first one counts);
    other alarms outside such windows are false alarms.
"""
import argparse, glob, json, os, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from river.drift import ADWIN

TH_MINOR, TH_MOD, TH_SEV = 0.05, 0.15, 0.30
WIN_VALID, H_EVAL, N_INC, N_RW, GAMMA = 2000, 500, 500, 1000, 0.995
STRATS = ["reweight", "incremental", "full_history", "full_recent"]


def make_clf(n_est, seed, n_jobs):
    return xgb.XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.05,
                             random_state=seed, n_jobs=n_jobs, verbosity=0,
                             tree_method="hist")


def score(model, X, y, metric="acc"):
    """Accuracy, or AUROC for heavily imbalanced streams (e.g. Sepsis)."""
    if metric == "auroc":
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    return float((model.predict(X) == y).mean())


# ------------------------------ generators ---------------------------------
def gen_sea(seed):
    rng = np.random.default_rng(seed)
    n, seg = 24000, 6000
    X = rng.uniform(0, 10, (n, 3))
    th = np.repeat([8, 9, 7, 9.5], seg)
    y = (X[:, 0] + X[:, 1] <= th).astype(int)
    flip = rng.random(n) < 0.10
    y[flip] ^= 1
    return dict(X=X, y=y, train_end=2000, drifts=[6000, 12000, 18000],
                names=["f0", "f1", "f2"], G=[0, 1], k=2)


def gen_stagger(seed):
    rng = np.random.default_rng(seed)
    n = 15000
    X = rng.integers(0, 3, (n, 3))  # color, size, shape
    c, s, sh = X[:, 0], X[:, 1], X[:, 2]
    y = np.zeros(n, int)
    y[:5000] = ((c == 0) & (s == 0))[:5000]
    y[5000:10000] = ((c == 1) | (sh == 0))[5000:10000]
    y[10000:] = ((s == 1) | (s == 2))[10000:]
    flip = rng.random(n) < 0.05
    y[flip] ^= 1
    return dict(X=X.astype(float), y=y, train_end=1500, drifts=[5000, 10000],
                names=["color", "size", "shape"], G=None, k=None)


def gen_gaussian(seed):
    rng = np.random.default_rng(seed)
    n, d = 15000, 7500
    X = rng.normal(0, 1, (n, 6))
    X[d:, 0] += 1.5
    X[d:, 1] -= 1.5
    X[d:, 2] += 1.0
    w0 = np.array([1.2, -1.0, 0.8, 0, 0, 0])
    w1 = np.array([0.2, 1.5, -1.2, 0.8, 0, 0])
    y = np.where(np.arange(n) < d, X @ w0 > 0, X @ w1 > 0).astype(int)
    flip = rng.random(n) < 0.05
    y[flip] ^= 1
    return dict(X=X, y=y, train_end=2000, drifts=[d],
                names=[f"feat_{i}" for i in range(6)], G=[0, 1, 2, 3], k=4)


GENS = dict(sea=gen_sea, stagger=gen_stagger, gaussian=gen_gaussian)


# ------------------------------ core loop ----------------------------------
def mean_abs_shap(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    return np.mean(np.abs(sv), axis=0)


def tier_of(dE):
    if dE < TH_MINOR: return "none"
    if dE < TH_MOD: return "reweight"
    if dE < TH_SEV: return "incremental"
    return "full_history"


def build_candidate(strategy, model, X, y, a, seed, n_jobs):
    """Candidate model using only samples 0..a."""
    if strategy == "none":
        return model
    if strategy == "reweight":
        lo = max(0, a + 1 - N_RW)
        Xr, yr = X[lo:a + 1], y[lo:a + 1]
        age = np.arange(len(yr))[::-1]
        w = GAMMA ** age
        return make_clf(50, seed, n_jobs).fit(Xr, yr, sample_weight=w, xgb_model=model.get_booster())
    if strategy == "incremental":
        lo = max(0, a + 1 - N_INC)
        return make_clf(50, seed, n_jobs).fit(X[lo:a + 1], y[lo:a + 1], xgb_model=model.get_booster())
    if strategy == "full_history":
        return make_clf(300, seed, n_jobs).fit(X[:a + 1], y[:a + 1])
    if strategy == "full_recent":
        lo = max(0, a + 1 - N_INC)
        return make_clf(300, seed, n_jobs).fit(X[lo:a + 1], y[lo:a + 1])
    raise ValueError(strategy)


def run_stream(stream, seed, n_jobs, evaluate_all=False, max_events=None):
    X, y, te, drifts = stream["X"], stream["y"], stream["train_end"], stream["drifts"]
    G, k, names = stream["G"], stream["k"], stream["names"]
    n = len(y)
    win = stream.get("win", WIN_VALID)
    metric = stream.get("metric", "acc")
    model = make_clf(300, seed, n_jobs).fit(X[:te], y[:te])
    adwin = ADWIN(delta=0.002)
    detected = {d: None for d in drifts}
    events, t = [], te
    false_alarms = 0
    while t < n and (max_events is None or len(events) < max_events):
        err = (model.predict(X[t:]) != y[t:]).astype(int)
        alarm = None
        for i, e in enumerate(err):
            adwin.update(int(e))
            if adwin.drift_detected:
                alarm = t + i
                break
        if alarm is None:
            break
        a = alarm
        # --- classify the alarm
        kind, which = "false", None
        for d in drifts:
            if d < a <= d + win:
                if detected[d] is None:
                    kind, which = "true", d
                    detected[d] = a - d
                else:
                    kind = "dup"
                break
        if kind == "false" and drifts:
            false_alarms += 1
        if a + 1 < 400:
            t = a + 1; adwin = ADWIN(delta=0.002); continue
        # --- explanation shift
        h = min(200, (a + 1) // 2)
        Xb, Xa = X[a + 1 - 2 * h:a + 1 - h], X[a + 1 - h:a + 1]
        t0 = time.perf_counter()
        pb, pa = mean_abs_shap(model, Xb), mean_abs_shap(model, Xa)
        shap_ms = (time.perf_counter() - t0) * 1000
        dE = np.abs(pa - pb)
        dE_tot = float(np.linalg.norm(dE))
        rank = list(np.argsort(-dE))
        rec = dict(seed=seed, idx=int(a), kind=kind, drift=which,
                   delay=(a - which) if which is not None else None,
                   dE_total=dE_tot, shap_ms=shap_ms,
                   top=[names[i] for i in rank[:5]], dE_top=[float(dE[i]) for i in rank[:5]],
                   tier=tier_of(dE_tot))
        if G is not None and kind == "true":
            rec["ap"] = len(set(rank[:k]) & set(G)) / k
        # --- adaptation evaluation (counterfactual, same future for all)
        H = min(stream.get("H", H_EVAL), n - a - 1)
        if (kind == "true" or evaluate_all) and H >= 100:
            Xf, yf = X[a + 1:a + 1 + H], y[a + 1:a + 1 + H]
            acc0 = score(model, Xf, yf, metric)
            rec["acc_base"] = acc0
            rec["gain"] = {}
            for s in STRATS:
                m2 = build_candidate(s, model, X, y, a, seed, n_jobs)
                rec["gain"][s] = score(m2, Xf, yf, metric) - acc0
        # --- pipeline continues with the policy-selected strategy
        strat = rec["tier"]
        if strat != "none":
            model = build_candidate(strat, model, X, y, a, seed, n_jobs)
        events.append(rec)
        adwin = ADWIN(delta=0.002)
        t = a + 1
    # stable samples for FAR
    stable = n - te - sum(min(win, n - d) for d in drifts) if drifts else None
    return dict(events=events, detected=detected, false_alarms=false_alarms, stable=stable)


# ------------------------- real PhysioNet/CinC 2019 -------------------------
S3_TRAIN = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/"


def sepsis_download(root):
    from concurrent.futures import ThreadPoolExecutor
    dir_a = os.path.join(root, "training_setA")
    dir_b = os.path.join(root, "training_setB")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    tasks = []
    for i in range(1, 20337):
        fn = f"p{i:06d}.psv"
        tasks.append((f"{S3_TRAIN}training_setA/{fn}", os.path.join(dir_a, fn)))
    for i in range(100001, 120001):
        fn = f"p{i:06d}.psv"
        tasks.append((f"{S3_TRAIN}training_setB/{fn}", os.path.join(dir_b, fn)))

    def _dl(item):
        url, dest = item
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                with open(dest, "wb") as f:
                    f.write(data)
            return True
        except Exception:
            return False

    print(f"Verifying/downloading {len(tasks)} PhysioNet files...", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=96) as ex:
        for ok in ex.map(_dl, tasks):
            if ok: done += 1
            if done % 10000 == 0:
                print(f"Downloaded {done}/{len(tasks)} files ({time.time()-t0:.1f}s)", flush=True)
    print(f"PhysioNet Sepsis dataset ready: {done}/{len(tasks)} files in {time.time()-t0:.1f}s", flush=True)


def sepsis_read(root):
    """Returns {'A': (X, y, pid), 'B': (X, y, pid)} using every .psv under root."""
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    def _read_file(f):
        d = pd.read_csv(f, sep="|")
        y = d["SepsisLabel"].values.astype(int)
        X = d.drop(columns=["SepsisLabel"]).values.astype(np.float32)
        return X, y, len(d)

    for hosp in ["A", "B"]:
        files = sorted(f for f in glob.glob(os.path.join(root, "**", "*.psv"), recursive=True)
                       if f"set{hosp}" in f.replace("\\", "/"))
        if not files:
            raise SystemExit(f"No .psv files for hospital set{hosp} under {root}. "
                             "Use --sepsis_download or extract training_setA/B there.")
        print(f"Reading set{hosp} ({len(files)} files)...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=32) as ex:
            res = list(ex.map(_read_file, files))
        Xs = [r[0] for r in res]
        ys = [r[1] for r in res]
        ps = [np.full(r[2], i) for i, r in enumerate(res)]
        cols = list(pd.read_csv(files[0], sep="|", nrows=1).drop(columns=["SepsisLabel"]).columns)
        out[hosp] = (np.vstack(Xs), np.concatenate(ys), np.concatenate(ps), cols)
        print(f"set{hosp}: {len(files)} patients, {len(out[hosp][1])} rows, "
              f"sepsis rate {out[hosp][1].mean():.4f} ({time.time()-t0:.1f}s)", flush=True)
    return out


def make_sepsis_gen(data, n_train=20000, n_pre=30000, n_post=30000, H=5000, win=5000):
    """Natural hospital shift: M0 on part of set A; stream = rest of set A, then set B.
    Patients are split first, then rows are sampled at random (seeded)."""
    XA, yA, pA, cols = data["A"]
    XB, yB, pB, _ = data["B"]

    def gen(seed):
        rng = np.random.default_rng(seed)
        pats = rng.permutation(np.unique(pA))
        tr_p = set(pats[: int(0.3 * len(pats))].tolist())
        in_tr = np.isin(pA, list(tr_p))
        i_tr = rng.choice(np.where(in_tr)[0], min(n_train, in_tr.sum()), replace=False)
        i_pre = rng.choice(np.where(~in_tr)[0], min(n_pre, (~in_tr).sum()), replace=False)
        i_post = rng.choice(len(yB), min(n_post, len(yB)), replace=False)
        X = np.vstack([XA[i_tr], XA[i_pre], XB[i_post]])
        y = np.concatenate([yA[i_tr], yA[i_pre], yB[i_post]])
        te = len(i_tr)
        return dict(X=X, y=y, train_end=te, drifts=[te + len(i_pre)], names=cols,
                    G=None, k=None, metric="auroc", H=H, win=win)
    return gen


# ------------------------------ driver -------------------------------------
def load_elec2(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "target"]
    return dict(X=df[feats].values.astype(float), y=df["target"].values.astype(int),
                train_end=2000, drifts=[], names=feats, G=None, k=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="?", const="", default="sea,stagger,gaussian")
    ap.add_argument("--seeds", default="42,0,1,7,13,99,123,256,314,777")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--elec2", default=None, help="path to elec2.csv (feat_*, target)")
    ap.add_argument("--sepsis_dir", default=None, help="folder with (or to receive) PhysioNet 2019 data")
    ap.add_argument("--sepsis_download", action="store_true")
    ap.add_argument("--sepsis_seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--sep_train", type=int, default=20000)
    ap.add_argument("--sep_pre", type=int, default=30000)
    ap.add_argument("--sep_post", type=int, default=30000)
    ap.add_argument("--sep_win", type=int, default=25000, help="detection window after hospital shift (default: 25000)")
    ap.add_argument("--sep_h", type=int, default=5000, help="evaluation horizon for adaptation on Sepsis (default: 5000)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {}
    jobs = [(s, GENS[s]) for s in (args.streams or "").split(",") if s]
    for name, gen in jobs:
        out[name] = []
        for sd in seeds:
            t0 = time.time()
            res = run_stream(gen(sd), sd, args.n_jobs)
            out[name].append(res)
            print(f"[{name}] seed {sd}: {len(res['events'])} events, "
                  f"FA={res['false_alarms']}, delays={res['detected']}, {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(args.out, "w"))
    if args.sepsis_dir:
        if args.sepsis_download:
            sepsis_download(args.sepsis_dir)
        gen = make_sepsis_gen(sepsis_read(args.sepsis_dir), args.sep_train, args.sep_pre, args.sep_post,
                              H=args.sep_h, win=args.sep_win)
        out["sepsis_real"] = []
        for sd in [int(x) for x in args.sepsis_seeds.split(",")]:
            t0 = time.time()
            res = run_stream(gen(sd), sd, args.n_jobs)
            out["sepsis_real"].append(res)
            print(f"[sepsis_real] seed {sd}: {len(res['events'])} events, FA={res['false_alarms']}, "
                  f"delays={res['detected']}, {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(args.out, "w"))
    if args.elec2:
        st = load_elec2(args.elec2)
        res = run_stream(st, 42, args.n_jobs, evaluate_all=True)
        out["elec2"] = [res]
        print(f"[elec2] {len(res['events'])} events", flush=True)
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
