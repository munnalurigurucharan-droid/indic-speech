"""
Task 2: Indic speech -> Whisper ASR -> LLM -> text response, plus evaluation.
Languages: Hindi (hi), Telugu (te). Data: FLEURS test split.
Conditions: clean vs noisy (white noise at --snr dB).
Modes: baseline (summarise ASR text) vs correct (LLM fixes ASR text first = improvement).

Kaggle/Colab (GPU T4) setup cell:
    !pip install -q faster-whisper jiwer "datasets==3.6.0" bitsandbytes accelerate scipy

Run (each stage caches to results/, so a crashed session can resume):
    python speech_pipeline.py asr
    python speech_pipeline.py llm
    python speech_pipeline.py summary
    python speech_pipeline.py all
"""
import argparse
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

OUT = "results"
SR = 16000
LANGS = {"hi": ("hi_in", "Hindi"), "te": ("te_in", "Telugu")}


# ---------------------------------------------------------------- text / audio utils
def normalize(t: str) -> str:
    """Same normalisation for refs and hyps: NFC, lowercase, drop zero-width chars and punctuation."""
    t = unicodedata.normalize("NFC", t).lower()
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def wer_cer(ref: str, hyp: str):
    import jiwer
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return np.nan, np.nan
    return jiwer.wer(r, h), jiwer.cer(r, h)


def add_noise(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    noise = rng.standard_normal(len(x)).astype(np.float32)
    p_sig = np.mean(x ** 2) + 1e-12
    scale = np.sqrt(p_sig / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return (x + scale * noise).astype(np.float32)


def load_fleurs(lang: str, n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("google/fleurs", LANGS[lang][0], split="test", trust_remote_code=True)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    return [ds[int(i)] for i in idx]


# ---------------------------------------------------------------- stage 1: ASR
def stage_asr(a):
    from faster_whisper import WhisperModel
    os.makedirs(OUT, exist_ok=True)
    asr = WhisperModel(a.whisper, device="cuda", compute_type="float16")

    def transcribe(wav, lang):
        t0 = time.perf_counter()
        segs, _ = asr.transcribe(wav, language=lang, beam_size=5, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs)  # generator is consumed inside the timer
        return text, time.perf_counter() - t0

    rows = []
    for lang in a.langs:
        data = load_fleurs(lang, a.n, a.seed)
        rng = np.random.default_rng(a.seed)
        transcribe(np.asarray(data[0]["audio"]["array"], dtype=np.float32), lang)  # warm-up, untimed
        for k, ex in enumerate(data):
            wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
            dur = len(wav) / SR
            for cond in ("clean", "noisy"):
                x = wav if cond == "clean" else add_noise(wav, a.snr, rng)
                hyp, t = transcribe(x, lang)
                ref = ex["raw_transcription"]
                w, c = wer_cer(ref, hyp)
                rows.append(dict(id=ex["id"], lang=lang, cond=cond, ref=ref, hyp=hyp,
                                 dur=dur, asr_time=t, wer=w, cer=c))
            if k % 10 == 0:
                print(f"[asr] {lang} {k}/{len(data)}")
    pd.DataFrame(rows).to_csv(f"{OUT}/asr.csv", index=False)
    print("saved", f"{OUT}/asr.csv")


# ---------------------------------------------------------------- stage 2: LLM + judge
def load_llm(name, four_bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(device_map="cuda")
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = torch.float16
    return tok, AutoModelForCausalLM.from_pretrained(name, **kw)


def chat(tok, model, system, user, max_new=160):
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    dt = time.perf_counter() - t0
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip(), dt


SYS = "You are a careful assistant. Reply only in the language of the user's text."


def p_answer(hyp):
    return ("Below is an automatic transcript of a spoken sentence (it may contain recognition errors). "
            "Summarise its meaning in ONE short sentence, in the same language as the transcript. "
            f"Do not add information that is not in the transcript.\n\nTranscript: {hyp}")


def p_correct(hyp, lang_name):
    return (f"The following is speech-recognition output in {lang_name} and may contain errors. "
            "Fix likely recognition errors (wrong or misspelled words) using context. "
            "Do not add or remove information. Output ONLY the corrected text.\n\n" + hyp)


JUDGE_SYS = "You are a strict evaluator. Output only JSON."


def p_judge(ref, resp):
    return ("REFERENCE (ground-truth transcript of what was said):\n" + ref +
            "\n\nRESPONSE (a one-sentence summary produced by a system):\n" + resp +
            "\n\nScore the RESPONSE against the REFERENCE.\n"
            "correctness: 2 = accurately captures the main content; 1 = partially correct or vague; "
            "0 = wrong, unrelated or empty.\n"
            "hallucination: 1 if the RESPONSE states specific information (names, numbers, events, claims) "
            "that is NOT supported by the REFERENCE, else 0.\n"
            'Return exactly: {"correctness": <0|1|2>, "hallucination": <0|1>}')


def parse_judge(s):
    c = re.search(r'"correctness"\s*:\s*([012])', s)
    h = re.search(r'"hallucination"\s*:\s*([01])', s)
    return (int(c.group(1)) if c else np.nan, int(h.group(1)) if h else np.nan)


def stage_llm(a):
    df = pd.read_csv(f"{OUT}/asr.csv").fillna({"hyp": ""})
    tok, model = load_llm(a.llm, a.four_bit)
    chat(tok, model, SYS, "warm up", max_new=4)  # warm-up
    rows = []
    for i, r in df.iterrows():
        lang_name = LANGS[r.lang][1]
        for mode in ("baseline", "correct"):
            hyp_used, llm_time = r.hyp, 0.0
            if mode == "correct":
                hyp_used, t1 = chat(tok, model, SYS, p_correct(r.hyp, lang_name), max_new=200)
                llm_time += t1
            resp, t2 = chat(tok, model, SYS, p_answer(hyp_used))
            llm_time += t2
            verdict, _ = chat(tok, model, JUDGE_SYS, p_judge(r.ref, resp), max_new=40)  # not timed in RTF
            corr, hall = parse_judge(verdict)
            w2, c2 = wer_cer(r.ref, hyp_used)
            rows.append(dict(id=r.id, lang=r.lang, cond=r.cond, mode=mode, ref=r.ref, asr_hyp=r.hyp,
                             used_hyp=hyp_used, response=resp, dur=r.dur, asr_time=r.asr_time,
                             llm_time=llm_time, rtf=(r.asr_time + llm_time) / r.dur,
                             asr_wer=r.wer, asr_cer=r.cer, used_wer=w2, used_cer=c2,
                             correctness=corr, hallucination=hall, judge_raw=verdict))
        if i % 10 == 0:
            print(f"[llm] {i}/{len(df)}")
            pd.DataFrame(rows).to_csv(f"{OUT}/llm_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/llm.csv", index=False)
    print("saved", f"{OUT}/llm.csv")


# ---------------------------------------------------------------- stage 3: summary
def stage_summary(a):
    d = pd.read_csv(f"{OUT}/llm.csv")
    d["correct_full"] = (d.correctness == 2).astype(float)
    g = d.groupby(["lang", "cond", "mode"])
    s = g.agg(n=("id", "count"), asr_wer=("asr_wer", "mean"), asr_cer=("asr_cer", "mean"),
              used_wer=("used_wer", "mean"), used_cer=("used_cer", "mean"),
              correctness=("correctness", "mean"), fully_correct=("correct_full", "mean"),
              hallucination=("hallucination", "mean"), rtf_mean=("rtf", "mean"),
              rtf_p50=("rtf", "median"), rtf_p95=("rtf", lambda x: x.quantile(0.95)),
              asr_time=("asr_time", "mean"), llm_time=("llm_time", "mean")).round(3)
    s.to_csv(f"{OUT}/summary.csv")
    print(s.to_string())

    b = d[d["mode"] == "baseline"]  # bottleneck evidence: does ASR error drive wrong answers?
    print("\nSpearman(ASR WER, correctness), baseline:")
    for lang, x in b.groupby("lang"):
        print(lang, round(x[["asr_wer", "correctness"]].corr(method="spearman").iloc[0, 1], 3))
    print("Latency share (baseline): ASR %.0f%%, LLM %.0f%%" % (
        100 * b.asr_time.sum() / (b.asr_time.sum() + b.llm_time.sum()),
        100 * b.llm_time.sum() / (b.asr_time.sum() + b.llm_time.sum())))

    h = b.sample(min(20, len(b)), random_state=a.seed)[["id", "lang", "cond", "ref", "response", "correctness", "hallucination"]]
    h["human_correctness"] = ""
    h["human_hallucination"] = ""
    h.to_csv(f"{OUT}/handcheck.csv", index=False)  # fill by hand -> report judge-human agreement
    print("\nWrote summary.csv and handcheck.csv (label 20 rows by hand to validate the judge).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["asr", "llm", "summary", "all"])
    p.add_argument("--langs", nargs="+", default=["hi", "te"])
    p.add_argument("--n", type=int, default=60, help="examples per language (60x2 = 120 total)")
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--whisper", default="large-v3")
    p.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--four_bit", action="store_true", help="use with 7B models on a 16GB GPU")
    a = p.parse_args()
    if a.stage in ("asr", "all"):
        stage_asr(a)
    if a.stage in ("llm", "all"):
        stage_llm(a)
    if a.stage in ("summary", "all"):
        stage_summary(a)
