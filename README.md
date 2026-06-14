# Nothing Lost in Translation

Lexically-constrained English→Russian neural machine translation built on top of a custom, from-scratch **beam search** decoder. The project re-implements Hugging Face's `.generate()` beam search by hand, shows that the custom decoder matches/beats it on BLEU, and then extends it with **Grid Beam Search (GBS)** so that user-supplied n-grams are *guaranteed* to appear in the output.

> Originally submitted as Graduate Assessment 1 (GA1) for **NLP 673** at UMBC, by Anupreet Singh.

## Contents

| File | Description |
|------|-------------|
| [GA1 new beam search.ipynb](GA1%20new%20beam%20search.ipynb) | The full implementation: data prep, fine-tuning, baseline decoding, custom beam search, and lexical constraints. |
| [GA1 M4.pdf](GA1%20M4.pdf) | The written report / writeup for the assessment. |

## What it does

The notebook walks through the full pipeline end to end:

1. **Data preparation & fine-tuning** — Loads the [WMT16 `ru-en`](https://huggingface.co/datasets/wmt16) dataset (WMT19 is available as a larger option), tokenizes EN/RU pairs, and fine-tunes `google/mt5-base` with the Hugging Face `Trainer`.
2. **Baseline decoding** — Translates the validation split with the standard `model.generate()` using `Helsinki-NLP/opus-mt-en-ru` (MarianMT) and scores it with `sacrebleu`.
3. **Custom beam search** — Replaces `.generate()` with a hand-written `scratch_beam_search` that drives the model's raw forward pass directly, and reproduces (and slightly improves on) the baseline BLEU.
4. **Lexical constraints (4 iterations)** — Adds Grid Beam Search so that n-grams missing from the baseline translation are forced into the output, applied iteratively.
5. **Single-instance inspection** — Runs the constrained decoder on individual sentences to confirm the requested tokens actually survive into the final translation.

### How the custom decoder works (start → finish)

The flow for one decoding step:

1. **Input fan-out** — Each source sentence is repeated `num_beams` times with `repeat_interleave` (one row per beam), and `decoder_input_ids` is seeded with the model's `decoder_start_token_id`.
2. **Forward pass** — The model produces logits for the next token of every beam; only the last position's logits are kept and turned into log-probabilities via `log_softmax`.
3. **Candidate expansion** — The top-`num_beams` tokens are taken from each beam, their scores added to the running `beam_scores`, then reshaped per source sentence so all `num_beams × num_beams` candidates compete.
4. **Beam selection** — The global top-`num_beams` candidates are selected; beam bookkeeping indices map each survivor back to the beam it extended, and the chosen token is appended to that beam's sequence.
5. **Termination** — Generation stops on EOS (early stopping) or `max_length`; the top-scoring beam per sentence is returned and decoded back to text.

### Lexical constraints (Grid Beam Search)

For each sentence, `find_missing_tokens` compares the reference translation against the baseline output and returns an n-gram that is missing — preferring **3-grams**, falling back to **2-grams**, then **1-grams**, and `None` if nothing is missing. `constrained_beam_search` then decodes while *forcing* that n-gram into the sequence, so the constraint is guaranteed to appear. This is repeated over four iterations, accumulating constraints per sentence.

## Results

Validation split of WMT16 EN→RU, scored with `sacrebleu`:

| Decoder | BLEU |
|---------|------|
| `model.generate()` (baseline) | **27.11** |
| `scratch_beam_search` (custom) | **27.48** |
| Constrained (iteration 1) | 26.34 |
| Constrained (iteration 2) | 26.60 |
| Constrained (iteration 3) | 26.47 |
| Constrained (iteration 4) | 26.32 |

The custom beam search **outperforms the default `.generate()`** (27.48 vs. 27.11). The constrained variants trade a small amount of BLEU for a hard guarantee that the missing reference n-grams are included in the output.

## Models & tooling

- **Translation model:** [`Helsinki-NLP/opus-mt-en-ru`](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru) (MarianMT) for decoding experiments.
- **Fine-tuning target:** [`google/mt5-base`](https://huggingface.co/google/mt5-base).
- **Stack:** `transformers`, `torch`, `datasets`, `sacrebleu`, `sentencepiece`, `accelerate`.

Special token IDs for the Helsinki model (used by the custom decoder): `eos = 0`, `unk = 1`, `pad = decoder_start = 62517`.

## Notes

- The notebook's `device` is set per cell — use `"cuda"` for an NVIDIA GPU, `"mps"` for Apple Silicon, or `"cpu"` otherwise.
- Dataset and model paths in the notebook are absolute and point at the original author's machine; update them to your local paths before running.
- Markdown cells labeled **Tech Note / TN** are the author's personal developer notes; the un-labeled markdown cells are the explanatory walkthrough.
