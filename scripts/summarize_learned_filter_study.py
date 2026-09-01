#!/usr/bin/env python3
"""Compare learned-filter and locked heuristic-filter ASR arms over five seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SEEDS = [11, 22, 33, 44, 55]
METRICS = ["overall", "domain_terms", "common_terms"]

def ci(values, n=10_000):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return {"estimate":None,"ci_low":None,"ci_high":None,"n_trials":0,"n_resamples":n}
    rng=np.random.default_rng(20260831); draws=values[rng.integers(0,len(values),(n,len(values)))].mean(1)
    low,high=np.quantile(draws,[.025,.975])
    return {"estimate":float(values.mean()),"ci_low":float(low),"ci_high":float(high),"n_trials":len(values),"n_resamples":n}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--heuristic-dir",default="experiments/results/crawl_training_study/augmented"); p.add_argument("--learned-dir",default="experiments/results/learned_filter_study/learned"); p.add_argument("--output",default="experiments/results/learned_filter_study/summary.json"); p.add_argument("--markdown",default="experiments/results/learned_filter_study/REPORT.md"); a=p.parse_args()
    result={"schema_version":1,"seeds":SEEDS,"n_trials":5,"comparisons":{}}
    for corpus in ["earnings21","openslr31"]:
        h=[json.loads((Path(a.heuristic_dir)/corpus/f"seed_{s}.json").read_text()) for s in SEEDS]
        l=[json.loads((Path(a.learned_dir)/corpus/f"seed_{s}.json").read_text()) for s in SEEDS]
        result["comparisons"][corpus]={}
        for metric in METRICS:
            hv=np.asarray([r["wer"][metric] for r in h],float); lv=np.asarray([r["wer"][metric] for r in l],float)
            result["comparisons"][corpus][metric]={"heuristic_mean":float(np.nanmean(hv)) if np.isfinite(hv).any() else None,"learned_mean":float(np.nanmean(lv)) if np.isfinite(lv).any() else None,"learned_minus_heuristic":ci(lv-hv)}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    def fmt(x): return "undefined" if x["estimate"] is None else f"{100*x['estimate']:+.2f} pp ({100*x['ci_low']:+.2f}, {100*x['ci_high']:+.2f})"
    rows=[]
    for corpus in ["earnings21","openslr31"]:
        for metric in METRICS:
            x=result["comparisons"][corpus][metric]; hm="undefined" if x["heuristic_mean"] is None else f"{100*x['heuristic_mean']:.2f}%"; lm="undefined" if x["learned_mean"] is None else f"{100*x['learned_mean']:.2f}%"; rows.append(f"| {corpus} | {metric} | {hm} | {lm} | {fmt(x['learned_minus_heuristic'])} |")
    s=result["comparisons"]["openslr31"]["overall"]["learned_minus_heuristic"]; e=result["comparisons"]["earnings21"]["overall"]["learned_minus_heuristic"]; passed=s["ci_high"] is not None and s["ci_high"]<0 and e["ci_high"]<=.01
    text="# Learned-filter downstream study\n\nGenerated from five paired trials. Differences are learned minus heuristic; intervals are 10,000-resample paired seed-bootstrap 95% CIs.\n\n| Corpus | WER slice | Heuristic mean | Learned mean | Paired difference |\n|---|---|---:|---:|---:|\n"+"\n".join(rows)+f"\n\nLocked replacement gate: **{'passed' if passed else 'not passed'}**. `N_trials=5`.\n"
    Path(a.markdown).write_text(text); print(text)
if __name__=="__main__": main()
