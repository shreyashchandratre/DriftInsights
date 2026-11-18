import json, sys, numpy as np
from scipy import stats
R = json.load(open(sys.argv[1]))
S = ["reweight","incremental","full_history","full_recent"]
def ci(v):
    v=np.asarray(v,float)
    if len(v)<2: return (v.mean(),float('nan'))
    return v.mean(), stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
for name, runs in R.items():
    print("="*70, name, "seeds:", len(runs))
    # detection
    tot=det=0; delays=[]; fa=[]; 
    for r in runs:
        for d,v in r["detected"].items():
            tot+=1
            if v is not None: det+=1; delays.append(v)
        if r["stable"]: fa.append(r["false_alarms"]/r["stable"]*1000)
    if tot==0:
        print("no ground-truth drifts (real stream): descriptive summary over all alarms; alarms =", sum(len(r['events']) for r in runs))
    else:
      print(f"detection: {det}/{tot} ({100*det/tot:.1f}%)  delay mean={np.mean(delays):.1f} sd={np.std(delays,ddof=1):.1f} median={np.median(delays):.0f}  FAR/1k stable: {np.mean(fa):.3f} ± {ci(fa)[1]:.3f}  total FA={sum(r['false_alarms'] for r in runs)}")
    ev=[e for r in runs for e in r["events"] if (e["kind"]=="true" or tot==0)]
    print("true-detection events:", len(ev))
    if not ev: continue
    dE=[e["dE_total"] for e in ev]
    print(f"dE_total mean={np.mean(dE):.3f} sd={np.std(dE,ddof=1):.3f} median={np.median(dE):.3f}  SHAP ms mean={np.mean([e['shap_ms'] for e in ev]):.0f}")
    tiers={t:sum(e['tier']==t for e in ev) for t in ["none","reweight","incremental","full_history"]}
    print("tiers:",tiers)
    aps=[e["ap"] for e in ev if "ap" in e]
    if aps: print(f"attribution precision mean={np.mean(aps):.3f} sd={np.std(aps,ddof=1):.3f} (n={len(aps)})")
    from collections import Counter
    c=Counter(e["top"][0] for e in ev); print("rank-1 feature counts:",dict(c))
    c2=Counter(tuple(e["top"][:2]) for e in ev).most_common(3); print("top-2 sets:",c2)
    # adaptation (seed-level means)
    ge=[e for e in ev if "gain" in e]
    print(f"adaptation events: {len(ge)}  acc_base mean={np.mean([e['acc_base'] for e in ge]):.3f}")
    for s in S:
        v=[100*e["gain"][s] for e in ge]; m=np.mean(v); h=ci(v)[1]
        print(f"  {s:13s} gain pp mean={m:6.2f} 95%CI ±{h:5.2f}")
    pol=[]; 
    for e in ge:
        t=e["tier"]; pol.append(0.0 if t=="none" else 100*e["gain"][t])
    orc=[100*max(0,max(e["gain"].values())) for e in ge]
    for lab,fn in [("policy",pol),("oracle",orc)]:
        print(f"  {lab:13s} gain pp mean={np.mean(fn):6.2f} 95%CI ±{ci(fn)[1]:5.2f}")
    for s in ["reweight","incremental","full_history"]:
        reg=np.mean(np.array(orc)-np.array([100*e['gain'][s] for e in ge]))
        print(f"  regret always-{s}: {reg:.2f}")
    print(f"  regret policy: {np.mean(np.array(orc)-np.array(pol)):.2f}")
    best=Counter(max(S,key=lambda s:e['gain'][s]) for e in ge); print("  best strategy counts:",dict(best))
    match=np.mean([ (e['tier']==max(["reweight","incremental","full_history"],key=lambda s:e['gain'][s])) for e in ge]); print(f"  policy picks best of 3: {100*match:.1f}%")
