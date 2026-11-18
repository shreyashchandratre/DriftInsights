# DriftInsights: Complete Benchmark Evaluation & Comparison

This document records the reproducible benchmark evaluation results across all synthetic and real-world data streams using the unified evaluation harness (`di_experiments.py`) and summary script (`di_summary.py`).

---

## 1. Synthetic Reproduction & Hardware Comparison (30 Seeds: 0–29)

Comparison between initial runs in a single-core Linux container and multi-threaded reproduction on a 6-core local workstation (`--n_jobs 6`):

| Dataset | Metric | Single-Core Container | 6-Core Local Machine | Consistency |
|---|---|:---:|:---:|:---:|
| **SEA** (24,000 samples) | Detection Rate | 50 / 90 (55.6%) | 50 / 90 (55.6%) | **Exact** |
| | Mean Detection Delay | 977.0 ± 500.0 | 979.2 ± 497.6 | $\Delta < 2.2$ samples |
| | False Alarm Rate (/1k) | 0.037 ± 0.015 | 0.037 ± 0.015 | **Exact** |
| | Attribution Precision (AP@2) | 0.820 ± 0.242 | 0.820 ± 0.242 | **Exact** |
| | Mean $\Delta E_\text{total}$ | 0.132 ± 0.074 | 0.133 ± 0.074 | **Exact** |
| | Average SHAP Time | 269 ms | 252 ms | -17 ms |
| | Severity Policy Gain | +2.78 pp | +2.74 pp | $\Delta = 0.04$ pp |
| | Oracle Upper Bound | +4.73 pp | +4.72 pp | $\Delta = 0.01$ pp |
| **STAGGER** (15,000 samples) | Detection Rate | 60 / 60 (100.0%) | 60 / 60 (100.0%) | **Exact** |
| | Mean Detection Delay | 152.0 ± 120.0 (med 79) | 151.5 ± 120.3 (med 79) | $\Delta = 0.5$ samples |
| | False Alarms | 0 | 0 | **Exact** |
| | Mean $\Delta E_\text{total}$ | 0.100 ± 0.046 | 0.100 ± 0.046 | **Exact** |
| | Severity Policy Gain | +23.8 pp | +23.83 pp | **Exact** |
| | Oracle Upper Bound | +32.1 pp | +32.06 pp | **Exact** |
| | Best Strategy (`reweight`) | 58 / 60 (96.7%) | 58 / 60 (96.7%) | **Exact** |
| **Gaussian** (15,000 samples) | Detection Rate | 30 / 30 (100.0%) | 30 / 30 (100.0%) | **Exact** |
| | Mean Detection Delay | 35.0 ± 0.0 | 35.0 ± 0.0 | **Exact** |
| | False Alarms | 0 | 0 | **Exact** |
| | Attribution Precision (AP@4) | 0.767 ± 0.160 | 0.767 ± 0.160 | **Exact** |
| | Mean $\Delta E_\text{total}$ | 0.192 ± 0.100 | 0.192 ± 0.100 | **Exact** |
| | Severity Policy Gain | +30.1 pp | +30.13 pp | **Exact** |
| | Oracle Upper Bound | +39.0 pp | +38.96 pp | **Exact** |

---

### Verbatim Output (`python di_summary.py results_synthetic.json`)

```text
====================================================================== sea seeds: 30
detection: 50/90 (55.6%)  delay mean=979.2 sd=497.6 median=1015  FAR/1k stable: 0.037 ± 0.015  total FA=18
true-detection events: 50
dE_total mean=0.133 sd=0.074 median=0.117  SHAP ms mean=252
tiers: {'none': 3, 'reweight': 33, 'incremental': 11, 'full_history': 3}
attribution precision mean=0.820 sd=0.242 (n=50)
rank-1 feature counts: {'f0': 28, 'f1': 17, 'f2': 5}
top-2 sets: [(('f0', 'f1'), 21), (('f1', 'f0'), 11), (('f0', 'f2'), 7)]
adaptation events: 50  acc_base mean=0.789
  reweight      gain pp mean=  2.80 95%CI ± 0.67
  incremental   gain pp mean=  3.00 95%CI ± 0.65
  full_history  gain pp mean=  2.84 95%CI ± 0.57
  full_recent   gain pp mean=  3.94 95%CI ± 0.88
  policy        gain pp mean=  2.74 95%CI ± 0.67
  oracle        gain pp mean=  4.72 95%CI ± 0.79
  regret always-reweight: 1.92
  regret always-incremental: 1.72
  regret always-full_history: 1.87
  regret policy: 1.98
  best strategy counts: {'full_recent': 24, 'full_history': 10, 'incremental': 8, 'reweight': 8}
  policy picks best of 3: 28.0%

====================================================================== stagger seeds: 30
detection: 60/60 (100.0%)  delay mean=151.5 sd=120.3 median=79  FAR/1k stable: 0.000 ± 0.000  total FA=0
true-detection events: 60
dE_total mean=0.100 sd=0.046 median=0.094  SHAP ms mean=189
tiers: {'none': 8, 'reweight': 44, 'incremental': 8, 'full_history': 0}
rank-1 feature counts: {'color': 30, 'size': 27, 'shape': 3}
top-2 sets: [(('color', 'size'), 24), (('size', 'color'), 16), (('size', 'shape'), 11)]
adaptation events: 60  acc_base mean=0.334
  reweight      gain pp mean= 31.99 95%CI ± 8.39
  incremental   gain pp mean= 22.59 95%CI ± 6.38
  full_history  gain pp mean= 10.70 95%CI ± 3.06
  full_recent   gain pp mean= 24.50 95%CI ± 6.82
  policy        gain pp mean= 23.83 95%CI ± 7.82
  oracle        gain pp mean= 32.06 95%CI ± 8.41
  regret always-reweight: 0.07
  regret always-incremental: 9.47
  regret always-full_history: 21.36
  regret policy: 8.23
  best strategy counts: {'reweight': 58, 'full_recent': 2}
  policy picks best of 3: 73.3%

====================================================================== gaussian seeds: 30
detection: 30/30 (100.0%)  delay mean=35.0 sd=0.0 median=35  FAR/1k stable: 0.000 ± 0.000  total FA=0
true-detection events: 30
dE_total mean=0.192 sd=0.100 median=0.164  SHAP ms mean=253
tiers: {'none': 1, 'reweight': 12, 'incremental': 14, 'full_history': 3}
attribution precision mean=0.767 sd=0.160 (n=30)
rank-1 feature counts: {'feat_1': 15, 'feat_5': 1, 'feat_0': 12, 'feat_2': 2}
top-2 sets: [(('feat_1', 'feat_2'), 8), (('feat_0', 'feat_1'), 8), (('feat_1', 'feat_0'), 7)]
adaptation events: 30  acc_base mean=0.111
  reweight      gain pp mean= 37.18 95%CI ± 2.18
  incremental   gain pp mean= 30.61 95%CI ± 2.51
  full_history  gain pp mean=  6.58 95%CI ± 1.03
  full_recent   gain pp mean= 36.68 95%CI ± 2.84
  policy        gain pp mean= 30.13 95%CI ± 4.64
  oracle        gain pp mean= 38.96 95%CI ± 2.16
  regret always-reweight: 1.78
  regret always-incremental: 8.35
  regret always-full_history: 32.38
  regret policy: 8.83
  best strategy counts: {'reweight': 18, 'full_recent': 12}
  policy picks best of 3: 40.0%
```

---

## 2. Electricity Stream Benchmark (`elec2.json`)

- **Dataset**: `data/elec2.csv` (45,312 samples, 6 features, target)
- **Protocol**: Real descriptive stream with no injected ground-truth drift.
- **Evaluation**: 52 sequential ADWIN alarms evaluated with counterfactual scoring on the subsequent $H=500$ samples.

```text
====================================================================== elec2 seeds: 1
no ground-truth drifts (real stream): descriptive summary over all alarms; alarms = 52
true-detection events: 52
dE_total mean=0.622 sd=0.500 median=0.465  SHAP ms mean=320
tiers: {'none': 0, 'reweight': 5, 'incremental': 13, 'full_history': 34}
rank-1 feature counts: {'feat_2': 38, 'feat_3': 5, 'feat_1': 2, 'feat_4': 3, 'feat_6': 3, 'feat_5': 1}
top-2 sets: [(('feat_2', 'feat_4'), 11), (('feat_2', 'feat_3'), 10), (('feat_2', 'feat_5'), 8)]
adaptation events: 52  acc_base mean=0.734
  reweight      gain pp mean=  3.84 95%CI ± 2.31
  incremental   gain pp mean=  3.99 95%CI ± 2.40
  full_history  gain pp mean=  2.55 95%CI ± 1.48
  full_recent   gain pp mean=  3.31 95%CI ± 2.65
  policy        gain pp mean=  3.59 95%CI ± 1.76
  oracle        gain pp mean=  6.62 95%CI ± 2.16
  regret always-reweight: 2.78
  regret always-incremental: 2.63
  regret always-full_history: 4.07
  regret policy: 3.03
  best strategy counts: {'full_recent': 12, 'full_history': 15, 'reweight': 15, 'incremental': 10}
  policy picks best of 3: 38.5%
```

---

## 3. Real Clinical Sepsis Benchmark (`sepsis_real.json`, 10 Seeds)

- **Dataset**: PhysioNet CinC 2019 (40,028 real ICU patients across Hospital A and Hospital B).
- **Drift Injected**: Natural hospital transfer at sample 50,000 (`--sep_win 25000`, `--sep_h 5000`).
- **Metric**: Area Under ROC Curve (AUROC) due to severe ~1.4% class imbalance.

```text
====================================================================== sepsis_real seeds: 10
detection: 10/10 (100.0%)  delay mean=9999.0 sd=4409.2 median=8703  FAR/1k stable: 0.000 ± 0.000  total FA=0
true-detection events: 10
dE_total mean=0.070 sd=0.021 median=0.063  SHAP ms mean=202
tiers: {'none': 1, 'reweight': 9, 'incremental': 0, 'full_history': 0}
rank-1 feature counts: {'HospAdmTime': 2, 'Glucose': 1, 'ICULOS': 5, 'HR': 1, 'MAP': 1}
top-2 sets: [(('ICULOS', 'HospAdmTime'), 3), (('HospAdmTime', 'Resp'), 2), (('Glucose', 'Age'), 1)]
adaptation events: 10  acc_base mean=0.670
  reweight      gain pp mean= -0.43 95%CI ± 1.15
  incremental   gain pp mean= -0.02 95%CI ± 1.65
  full_history  gain pp mean=  9.96 95%CI ± 3.10
  full_recent   gain pp mean= -5.28 95%CI ± 6.13
  policy        gain pp mean= -0.48 95%CI ± 1.13
  oracle        gain pp mean=  9.96 95%CI ± 3.10
  regret always-reweight: 10.39
  regret always-incremental: 9.98
  regret always-full_history: 0.00
  regret policy: 10.44
  best strategy counts: {'full_history': 10}
  policy picks best of 3: 0.0%
```

---

## 4. How to Reproduce

With the environment activated (`pip install -r requirements.txt`):

```bash
# 1. Reproduce synthetic benchmarks (30 seeds)
python di_experiments.py --streams "sea,stagger,gaussian" --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" --n_jobs 6 --out results_synthetic.json
python di_summary.py results_synthetic.json

# 2. Reproduce Electricity stream
python di_experiments.py --streams "" --elec2 data/elec2.csv --n_jobs 6 --out elec2.json
python di_summary.py elec2.json

# 3. Reproduce Sepsis hospital shift (requires downloaded data in data/sepsis)
python di_experiments.py --streams "" --sepsis_dir data/sepsis --sep_win 25000 --n_jobs 6 --out sepsis_real.json
python di_summary.py sepsis_real.json
```
