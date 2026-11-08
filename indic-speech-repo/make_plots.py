"""
Make report figures and tables from the pipeline outputs.

    python make_plots.py                                   # uses results/
    python make_plots.py --ablate results_large results_medium results_snr5
        # compares asr.csv across runs (WER, CER, ASR RTF) -> figures/ablation_asr.png

Outputs go to figures/ (PNG) and tables.md (markdown tables to paste into the report).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "figures"


def save(name):
    os.makedirs(FIG, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}.png", dpi=200)
    plt.close()
    print("saved", f"{FIG}/{name}.png")


def grouped_bar(df, idx, col, title, ylabel, name, err=None):
    p = df.pivot_table(index=idx[0], columns=idx[1], values=col, aggfunc="mean")
    ax = p.plot(kind="bar", rot=0, figsize=(6, 4), edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(title=idx[1])
    save(name)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in np.unique(np.concatenate([a, b])))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(a):
    d = pd.read_csv(f"{a.results}/llm.csv")
    base = d[d["mode"] == "baseline"]
    md = []

    # --- data quality: judge parse failures
    nan_c = d.correctness.isna().sum()
    nan_h = d.hallucination.isna().sum()
    md.append(f"Judge parse failures: correctness {nan_c}/{len(d)}, hallucination {nan_h}/{len(d)}\n")
    print(md[-1])

    # --- 1. ASR error by language x condition (one row per audio, so use baseline rows)
    grouped_bar(base, ("lang", "cond"), "asr_wer", "Whisper WER by language and condition", "WER", "wer_by_lang_cond")
    grouped_bar(base, ("lang", "cond"), "asr_cer", "Whisper CER by language and condition", "CER", "cer_by_lang_cond")

    # --- 2. Does ASR error drive wrong answers?
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)
    for lang, x in base.groupby("lang"):
        ax.scatter(x.asr_wer.clip(upper=1.5), x.correctness + rng.uniform(-0.08, 0.08, len(x)), alpha=0.5, label=lang)
    ax.set_xlabel("ASR WER (clipped at 1.5)")
    ax.set_ylabel("Judge correctness (0 to 2, jittered)")
    ax.set_title("Correctness vs ASR error (baseline)")
    ax.legend()
    save("correctness_vs_wer")

    # --- 3. Latency breakdown
    t = d.groupby(["lang", "cond", "mode"])[["asr_time", "llm_time", "dur"]].mean()
    labels = [f"{l}\n{c}\n{m}" for l, c, m in t.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, t.asr_time / t.dur, label="ASR", edgecolor="black")
    ax.bar(labels, t.llm_time / t.dur, bottom=t.asr_time / t.dur, label="LLM", edgecolor="black")
    ax.set_ylabel("Real-time factor (mean time / audio duration)")
    ax.set_title("Latency breakdown (ASR vs LLM)")
    ax.legend()
    save("rtf_breakdown")

    # --- 4. Improvement vs baseline
    for col, nm, yl in [("correctness", "correctness_base_vs_correct", "Mean correctness (0 to 2)"),
                        ("hallucination", "hallucination_base_vs_correct", "Hallucination rate"),
                        ("used_wer", "wer_base_vs_correct", "WER of the text given to the LLM")]:
        d["lang_cond"] = d.lang + "-" + d.cond
        grouped_bar(d, ("lang_cond", "mode"), col, f"{yl}: baseline vs ASR-correction", yl, nm)

    # --- tables
    s = pd.read_csv(f"{a.results}/summary.csv")
    md.append("## Summary\n\n" + s.to_markdown(index=False) + "\n")
    delta = d.pivot_table(index=["lang", "cond"], columns="mode",
                          values=["correctness", "hallucination", "used_wer", "rtf"], aggfunc="mean")
    out = pd.DataFrame({f"{m}_delta": delta[m]["correct"] - delta[m]["baseline"]
                        for m in ["correctness", "hallucination", "used_wer", "rtf"]}).round(3)
    md.append("## Change from baseline when using ASR-correction (correct minus baseline)\n\n" + out.to_markdown() + "\n")
    print(out)

    # --- judge vs human agreement
    hp = f"{a.results}/handcheck.csv"
    if os.path.exists(hp):
        h = pd.read_csv(hp).dropna(subset=["human_correctness", "human_hallucination"])
        if len(h):
            agree_c = (h.correctness == h.human_correctness).mean()
            agree_h = (h.hallucination == h.human_hallucination).mean()
            kc = cohen_kappa(h.correctness, h.human_correctness)
            kh = cohen_kappa(h.hallucination, h.human_hallucination)
            line = (f"Judge vs human (n={len(h)}): correctness agreement {agree_c:.2f} (kappa {kc:.2f}), "
                    f"hallucination agreement {agree_h:.2f} (kappa {kh:.2f})\n")
            md.append(line)
            print(line)

    # --- ablation across ASR runs
    if a.ablate:
        rows = []
        for folder in a.ablate:
            x = pd.read_csv(f"{folder}/asr.csv")
            x["run"] = os.path.basename(folder.rstrip("/"))
            x["asr_rtf"] = x.asr_time / x.dur
            rows.append(x)
        x = pd.concat(rows)
        agg = x.groupby(["run", "lang", "cond"])[["wer", "cer", "asr_rtf"]].mean().round(3)
        md.append("## ASR ablation\n\n" + agg.to_markdown() + "\n")
        print(agg)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, nm in zip(axes, ["wer", "asr_rtf"], ["WER", "ASR real-time factor"]):
            x.pivot_table(index=["lang", "cond"], columns="run", values=col, aggfunc="mean").plot(kind="bar", ax=ax, rot=0, edgecolor="black")
            ax.set_ylabel(nm)
            ax.set_title(nm + " across runs")
        save("ablation_asr")

    with open("tables.md", "w") as f:
        f.write("\n".join(md))
    print("wrote tables.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--ablate", nargs="*", default=[])
    main(p.parse_args())
