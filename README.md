# Nothing Lost in Translation

> Lexically-constrained English→Russian neural machine translation built on a **from-scratch beam search decoder**. The notebook re-implements Hugging Face's `.generate()` beam search by hand, shows the custom decoder *matches and beats* it on BLEU, then extends it with **Grid Beam Search** so that any user-supplied n-gram is *guaranteed* to appear in the translation.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Translator" src="https://img.shields.io/badge/MT-MarianMT%20(opus--mt--en--ru)-orange">
  <img alt="Fine-tune" src="https://img.shields.io/badge/Fine--tune-google%2Fmt5--base-blueviolet">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Metric" src="https://img.shields.io/badge/Metric-sacreBLEU-success">
</p>

*Graduate Assessment 1 (GA1) — NLP 673, University of Maryland Baltimore County. Sole author: **Anupreet Singh**.*

---

## Table of Contents

- [Overview](#overview)
- [Why this is interesting](#why-this-is-interesting)
- [How it works](#how-it-works)
- [Lexical constraints (Grid Beam Search)](#lexical-constraints-grid-beam-search)
- [Why constrained decoding matters](#why-constrained-decoding-matters)
- [Results](#results)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Limitations & future work](#limitations--future-work)
- [Author & contributions](#author--contributions)

---

## Overview

A translation model's decoder is usually a black box: you call `.generate()` and trust whatever beam search hands back. This project opens that box. It re-implements beam search **from scratch** — driving the model's raw forward pass token by token — and then goes one step further than the library does: it adds **hard lexical constraints**, so specific words or phrases can be *forced into* (or kept out of) the output rather than left to the model's discretion.

The whole pipeline lives in a single annotated notebook that runs end to end: prepare WMT data, fine-tune `mt5-base`, decode with MarianMT, reproduce the baseline with a hand-written decoder, then layer Grid Beam Search on top.

## Why this is interesting

- **Beam search, demystified.** The custom decoder reproduces `.generate()` from first principles — fan-out, forward pass, candidate expansion, beam selection, termination — so every step is visible and editable instead of hidden in library internals.
- **It doesn't just match the baseline — it beats it.** The hand-written decoder scores **27.48 BLEU vs. 27.11** for `model.generate()` on the WMT16 EN→RU validation split.
- **Hard guarantees, not soft hints.** Grid Beam Search *guarantees* a requested n-gram survives into the final translation — the kind of control a prompt or a temperature knob can't give you.
- **Real-world relevant.** The same machinery underpins glossary compliance, legal/medical terminology, brand consistency, and content-safety blocklists (see below).

## How it works

A single decoding step takes the source sentence through five stages — fan-out → forward pass → candidate expansion → beam selection → termination — looping steps 2–4 until every beam ends:

**1. Input fan-out.** Each source sentence is repeated `num_beams` times with `repeat_interleave` (one row per beam), and `decoder_input_ids` is seeded with the model's `decoder_start_token_id`.

**2. Forward pass.** The model produces logits for the next token of every beam; only the last position's logits are kept and converted to log-probabilities via `log_softmax`.

**3. Candidate expansion.** The top-`num_beams` tokens are taken from each beam, their scores added to the running `beam_scores`, then reshaped per source sentence so all `num_beams × num_beams` candidates compete against each other.

**4. Beam selection.** The global top-`num_beams` candidates are selected; bookkeeping indices map each survivor back to the beam it extended, and the chosen token is appended to that beam's sequence.

**5. Termination.** Generation stops on EOS (early stopping) or `max_length`; the top-scoring beam per sentence is returned and decoded back to text.

## Lexical constraints (Grid Beam Search)

Beyond reproducing the baseline, the decoder can **force chosen n-grams into the output**. For each sentence, `find_missing_tokens` compares the reference translation against the baseline output and returns an n-gram that's missing — preferring **3-grams**, falling back to **2-grams**, then **1-grams**, and `None` if nothing is missing. `constrained_beam_search` then decodes while *forcing* that n-gram into the sequence, so the constraint is guaranteed to appear. This is repeated over **four iterations**, accumulating constraints per sentence, with a single-instance inspection step to confirm the requested tokens actually survive into the final translation.

<details>
<summary><strong>Notebook walkthrough — the five stages</strong></summary>

1. **Data preparation & fine-tuning** — Loads the [WMT16 `ru-en`](https://huggingface.co/datasets/wmt16) dataset (WMT19 is available as a larger option), tokenizes EN/RU pairs, and fine-tunes [`google/mt5-base`](https://huggingface.co/google/mt5-base) with the Hugging Face `Trainer`.
2. **Baseline decoding** — Translates the validation split with the standard `model.generate()` using [`Helsinki-NLP/opus-mt-en-ru`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) (MarianMT) and scores it with `sacrebleu`.
3. **Custom beam search** — Replaces `.generate()` with a hand-written `scratch_beam_search` that drives the model's raw forward pass directly, reproducing and slightly improving on the baseline BLEU.
4. **Lexical constraints (4 iterations)** — Adds Grid Beam Search so n-grams missing from the baseline translation are forced into the output, applied iteratively.
5. **Single-instance inspection** — Runs the constrained decoder on individual sentences to confirm the forced tokens survive into the final translation.

</details>

## Why constrained decoding matters

Here the constraints are mined automatically from reference translations, but the same machinery accepts *any* user-supplied vocabulary. An algorithm that can **force terms in — or, by inverting the logic, keep terms out** — is valuable wherever a translation cannot be left to the model's discretion:

- **Domain jargon & specialized terminology** — where a term has one correct rendering and a "fluent" paraphrase changes the meaning:
  - *Legal* — contracts, statutes, and patents rely on terms of art ("force majeure", "indemnify", "without prejudice"). Constrained decoding pins those exact phrases so the model can't soften them into an everyday paraphrase that carries a different legal meaning.
  - *Medical & pharmaceutical* — drug names, dosages, anatomical terms, and ICD codes must come through verbatim; a paraphrase of a dosage is a safety incident.
- **Named entities** — names of people, places, and organizations the model would otherwise mistranslate or transliterate inconsistently across a document.
- **Brand & product consistency** — trademarks, product names, slogans, and UI strings that must stay identical (or untranslated) across every locale.
- **Terminology / glossary compliance** — forcing an approved bilingual glossary term keeps output consistent across documents and translators.
- **Content safety / negative constraints** — the inverse use: *blocking* slurs, profanity, competitor names, or leaked codenames by banning those token sequences.
- **Interactive / human-in-the-loop translation** — a post-editor who corrects one word can feed it back as a constraint and have the system re-decode the rest of the sentence around that fixed choice.

## Results

Validation split of WMT16 EN→RU, scored with `sacrebleu`:

| Decoder | BLEU |
|---------|------|
| `model.generate()` (baseline) | 27.11 |
| **`scratch_beam_search` (custom)** | **27.48** |
| Constrained (iteration 1) | 26.34 |
| Constrained (iteration 2) | 26.60 |
| Constrained (iteration 3) | 26.47 |
| Constrained (iteration 4) | 26.32 |

The custom beam search **outperforms the default `.generate()`** (27.48 vs. 27.11). The constrained variants trade a small amount of BLEU for a hard guarantee that the missing reference n-grams appear in the output.

## Getting started

The whole project is a single notebook.

1. **Open the notebook** — [GA1 new beam search.ipynb](GA1%20new%20beam%20search.ipynb) in Jupyter, VS Code, or Colab.
2. **Install dependencies** — the first cells `pip install` `transformers`, `"numpy<2"`, `datasets`, `torch`, `accelerate>=0.26.0`, and `sacrebleu`. (numpy must stay on 1.x — numpy 2.0 doesn't play well with the torch version used here.)
3. **Set the device** — the `device` is set per cell: use `"cuda"` for an NVIDIA GPU, `"mps"` for Apple Silicon, or `"cpu"` otherwise.
4. **Update the paths** — dataset and model paths in the notebook are absolute and point at the original author's machine; change them to your local paths before running.
5. **Run the cells in order** — data prep → fine-tune → baseline → custom beam search → constraints.

> **Reading the notebook:** markdown cells labeled **Tech Note / TN** are the author's personal developer notes; the un-labeled markdown cells are the explanatory walkthrough.

## Project structure

```
Nothing-Lost-in-Translation/
├── GA1 new beam search.ipynb   # Full pipeline: data prep, fine-tuning, baseline,
│                               # custom beam search, lexical constraints
└── GA1 M4.pdf                  # Written report / assessment writeup
```

## Tech stack

| Component | Choice |
| --- | --- |
| Translation model (decoding) | `Helsinki-NLP/opus-mt-en-ru` (MarianMT) |
| Fine-tuning target | `google/mt5-base` |
| ML framework | PyTorch + Hugging Face `transformers` / `accelerate` |
| Data | WMT16 `ru-en` (WMT19 optional) via `datasets` |
| Evaluation | `sacrebleu` |
| Tokenization | `sentencepiece` |

Special token IDs for the Helsinki model (used by the custom decoder): `eos = 0`, `unk = 1`, `pad = decoder_start = 62517`.

## Limitations & future work

- **Hard-coded absolute paths** — dataset/model locations point at the author's machine and must be edited before the notebook runs elsewhere.
- **Constraints mined from references** — the demo discovers missing n-grams by comparing against the gold translation; a production setup would take constraints from a user-supplied glossary instead.
- **Small BLEU cost under constraints** — forcing n-grams trades ~1 BLEU point; smarter constraint placement could narrow that gap.
- **Single language pair** — only EN→RU is evaluated; the decoder is language-agnostic and could be extended to other pairs.
- **Ideas:** negative-constraint (blocklist) demo, glossary-driven constraint input, batched constrained decoding, and a small interactive translation UI.

## Author & contributions

**Anupreet Singh** — sole author. Designed and built the entire project end to end:

- Re-implemented Hugging Face beam search **from scratch** (`scratch_beam_search`), driving the model's raw forward pass — and showed it **beats** `.generate()` on BLEU (27.48 vs. 27.11).
- Implemented **Grid Beam Search** lexical constraints (`find_missing_tokens`, `constrained_beam_search`) with the 3-gram→2-gram→1-gram fallback and the iterative constraint-accumulation loop.
- Built the full data + fine-tuning pipeline (WMT16, `mt5-base` via `Trainer`), the MarianMT baseline, `sacrebleu` evaluation, and the single-instance verification step.
- Authored the annotated walkthrough and the written report.
