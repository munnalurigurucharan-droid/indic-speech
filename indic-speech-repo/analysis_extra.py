"""
Extra analyses used in the report (run after speech_pipeline.py and make_plots.py):
  - response-language adherence (script of each response)
  - paired bootstrap, baseline vs ASR-correction (pairs = consecutive rows of llm.csv)
  - paired bootstrap, clean vs noisy ASR error
  - WER terciles and Spearman correlations
  - judge vs audit agreement, if results/handcheck_audit.csv exists
Writes results/extra_stats.txt
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = sys.argv[1] if len(sys.argv) > 1 else "results"
rng = np.random.default_rng(0)
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))

d = pd.read_csv(f"{R}/llm.csv")
a = pd.read_csv(f"{R}/asr.csv")
RANGES = {"hi": (0x900, 0x97F), "te": (0xC00, 0xC7F)}


def frac(text, lo, hi):
    ch = [c for c in str(text) if c.isalpha()]
    return sum(lo <= ord(c) <= hi for c in ch) / len(ch) if ch else 0.0


def latin(text):
    ch = [c for c in str(text) if c.isalpha()]
    return sum(c.isascii() for c in ch) / len(ch) if ch else 0.0


d["in_lang"] = [frac(r.response, *RANGES[r.lang]) >= 0.5 for r in d.itertuples()]
d["english"] = d.response.map(latin) >= 0.5
d["other"] = ~d.in_lang & ~d.english
P("== Response language (share of responses) ==")
P(d.groupby("lang")[["in_lang", "english", "other"]].mean().round(3).to_string())

# pairs: llm.csv rows come as (baseline, correct) for each ASR row
d["pair"] = np.arange(len(d)) // 2
assert (d.groupby("pair")["mode"].apply(list).map(lambda m: m == ["baseline", "correct"])).all()
base = d[d["mode"] == "baseline"].set_index("pair")
corr = d[d["mode"] == "correct"].set_index("pair")


def boot(x, n=10000):
    x = np.asarray(x, dtype=float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


P("\n== Correction minus baseline, paired bootstrap (mean, 95% CI) ==")
for (lang, cond), idx in base.groupby(["lang", "cond"]).groups.items():
    for m in ["correctness", "hallucination", "used_wer", "used_cer", "rtf"]:
        mu, lo, hi = boot(corr.loc[idx, m] - base.loc[idx, m])
        P(lang, cond, m, f"{mu:+.3f} ({lo:+.3f}, {hi:+.3f})", "*" if lo > 0 or hi < 0 else "")
c = corr.copy()
c["dcer"] = c.used_cer - c.asr_cer
P("CER lowered / unchanged / raised by correction:",
  c.groupby("lang").apply(lambda g: (int((g.dcer < -1e-9).sum()), int((g.dcer.abs() <= 1e-9).sum()), int((g.dcer > 1e-9).sum()))).to_dict())

P("\n== Clean to noisy, paired bootstrap on ASR error ==")
a["pair"] = np.arange(len(a)) // 2
for lang in ["hi", "te"]:
    g = a[a.lang == lang]
    cl, no = g[g.cond == "clean"].set_index("pair"), g[g.cond == "noisy"].set_index("pair")
    for m in ["wer", "cer"]:
        mu, lo, hi = boot(no[m] - cl[m])
        P(lang, m, f"{mu:+.3f} ({lo:+.3f}, {hi:+.3f})")

P("\n== WER terciles and Spearman (baseline rows) ==")
for lang, g in base.groupby("lang"):
    g = g.copy()
    g["bin"] = pd.qcut(g.asr_wer, 3, labels=["low", "mid", "high"], duplicates="drop")
    P(lang, g.groupby("bin", observed=True)[["asr_wer", "correctness", "hallucination"]].mean().round(3).to_string())
    P(lang, "Spearman WER-correctness", spearmanr(g.asr_wer, g.correctness))
    P(lang, "Spearman WER-hallucination", spearmanr(g.asr_wer, g.hallucination))

try:
    h = pd.read_csv(f"{R}/handcheck_audit.csv")
    def kappa(x, y):
        x, y = np.asarray(x), np.asarray(y)
        po = (x == y).mean()
        pe = sum((x == k).mean() * (y == k).mean() for k in np.unique(np.concatenate([x, y])))
        return (po - pe) / (1 - pe)
    P("\n== Judge vs audit (n=%d) ==" % len(h))
    P("correctness agreement", (h.correctness == h.audit_correctness).mean(), "kappa", round(kappa(h.correctness, h.audit_correctness), 2))
    P("hallucination agreement", (h.hallucination == h.audit_hallucination).mean(), "kappa", round(kappa(h.hallucination, h.audit_hallucination), 2))
    P(h.groupby("lang")[["correctness", "audit_correctness", "hallucination", "audit_hallucination"]].mean().round(2).to_string())
except FileNotFoundError:
    pass

open(f"{R}/extra_stats.txt", "w").write("\n".join(out))
print("\n".join(out))
