# Indic Speech to Text-Response Pipeline (Hindi and Telugu)

Speech input, Whisper ASR, an LLM, then a text response, with evaluation code (WER/CER, LLM-judged correctness and hallucination, latency and real-time factor) and one improvement experiment (ASR-transcript correction) compared against the baseline. The full write-up is in the PDF report.

## Pipeline

```
FLEURS audio (hi, te) -> Whisper large-v3 (faster-whisper) -> Qwen2.5-3B-Instruct -> one-sentence response
                      clean / noisy (white noise, SNR = 10 dB)
```

- **Data:** FLEURS test split, 60 random clips per language (seed 0), 120 clips total (55 Hindi and 54 Telugu distinct sentences, since FLEURS has several speakers per sentence). Each clip is run clean and noisy.
- **ASR:** `faster-whisper` large-v3, fp16, beam size 5, language forced to `hi` or `te`.
- **LLM:** `Qwen/Qwen2.5-3B-Instruct` (fp16, greedy). Task: summarise the transcript in one sentence, in the language of the transcript, without adding information.
- **Improvement (`correct` mode):** the LLM first corrects likely ASR errors, then the summary is produced from the corrected text. Compared with `baseline` on the same clips.

## Hardware and runtime

Kaggle notebook, 1x Tesla T4, Python 3.13, about 69 minutes for the full run (240 ASR runs, 480 responses and judgements).

## Reproduce

```bash
pip install -r requirements.txt

python speech_pipeline.py all --n 5      # smoke test
python speech_pipeline.py all            # full run: --n 60 --snr 10 --seed 0
python make_plots.py                     # figures/ and tables.md
python analysis_extra.py results         # response language, paired bootstrap, terciles -> results/extra_stats.txt
```

Each stage can be run alone: `python speech_pipeline.py asr`, `llm`, `summary`. Notes:

- The script passes `trust_remote_code=True` to `load_dataset`. Recent `datasets` versions print "trust_remote_code is not supported anymore" and ignore it; FLEURS still loads. This was the case in the reported run.
- Results of an ablation run overwrite `results/`, so rename the folder before starting another run.
- `python make_plots.py --ablate results_a results_b` compares `asr.csv` files from several ASR runs.

## Files

| File | Contents |
|---|---|
| `speech_pipeline.py` | ASR, LLM, judge and summary stages |
| `make_plots.py` | Figures, markdown tables, judge-agreement, ASR-ablation comparison |
| `analysis_extra.py` | Response-language check, paired bootstrap, WER terciles, judge audit statistics |
| `results/asr.csv` | Per clip and condition: reference, ASR text, duration, ASR time, WER, CER |
| `results/llm.csv` | Per clip, condition and mode: response, judge scores, times, RTF, WER/CER of the text given to the LLM |
| `results/summary.csv` | Aggregates by language, condition and mode |
| `results/handcheck.csv` | 20 sampled responses (human label columns empty) |
| `results/handcheck_audit.csv` | The same 20 responses with audit labels from a stronger LLM (Claude); not human labels |
| `results/extra_stats.txt` | Output of `analysis_extra.py` |
| `figures/`, `tables.md` | Plots and tables used in the report |

## Metric definitions

- **WER/CER:** `jiwer` on normalised text (Unicode NFC, lowercase, zero-width characters removed, punctuation and symbols replaced by spaces), applied identically to references and hypotheses.
- **Correctness (0 to 2):** LLM judge compares the response with the reference transcript. 2 = captures the main content, 1 = partial or vague, 0 = wrong, unrelated or empty.
- **Hallucination (0/1):** 1 if the response states specific information not supported by the reference transcript.
- **RTF = (ASR time + LLM time) / audio duration**, measured after a warm-up call; judge time excluded.
- **Response language:** a response counts as in-language if at least half of its letters are in the target script.
- **Differences** between modes or conditions use a paired bootstrap over clips (10,000 resamples, 95% intervals).

## Main results (baseline, Whisper large-v3 + Qwen2.5-3B)

| Language | Condition | WER | CER | Correctness (0 to 2) | Hallucination | RTF mean |
|---|---|---|---|---|---|---|
| Hindi | Clean | 0.287 | 0.106 | 1.23 | 33% | 0.40 |
| Hindi | Noisy | 0.360 | 0.157 | 0.93 | 53% | 0.41 |
| Telugu | Clean | 0.714 | 0.273 | 1.13 | 38% | 0.70 |
| Telugu | Noisy | 0.782 | 0.339 | 1.03 | 47% | 0.64 |

Correction mode did not help: it raised CER significantly in both languages, hurt Telugu on every metric, and raised RTF by 0.56 to 0.82. See the report for the paired statistics.

## Limitations

- The judge is the same small model as the system and is unreliable for Telugu: in a 20-response audit, agreement with a stronger LLM was 40% for correctness and 45% for hallucination, and for Telugu the judge's mean correctness was 1.00 against 0.18 in the audit. Telugu correctness and hallucination figures should not be read as evidence of good Telugu performance.
- The audit was done by an LLM, not a human, and covers only 20 responses.
- Most replies (80% for Hindi inputs, 75% for Telugu inputs) were in English, not the input language.
- FLEURS is read speech, white noise is a simplification, and 60 clips per language only supports detecting large effects.
- Single run, single seed.
