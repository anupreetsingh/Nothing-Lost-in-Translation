# Nothing Lost in Translation

**Lexically-constrained English→Russian machine translation, built by taking the decoder apart.** This project re-implements the beam search mechanism commonly used in Hugging Face's `.generate()` by reconstructing decoding directly from the model's raw forward pass. It verifies that the hand-written decoder matches the library implementation on BLEU, then extends it to do something `.generate()` cannot: **guarantee** that user-specified phrases appear in the translation (or are kept out of it) — with translation quality *rising*, not falling, as constraints are added.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Framework" src="https://img.shields.io/badge/PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Model" src="https://img.shields.io/badge/MarianMT-opus--mt--en--ru-orange">
  <img alt="Metric" src="https://img.shields.io/badge/Metric-sacreBLEU-success">
  <img alt="Method" src="https://img.shields.io/badge/Method-Grid%20Beam%20Search%20(Hokamp%20%26%20Liu%202017)-blueviolet">
</p>

*Grew out of Graduate Assessment 1 (GA1) — NLP 673, University of Maryland Baltimore County. Sole author: **Anupreet Singh**.*

---

## Results at a glance

First 100 sentences of the WMT16 EN→RU validation split, beam width 4, scored with sacreBLEU:

| Decoder | Constraint information used | BLEU |
|---|---|---|
| `model.generate()` (Hugging Face) | none | 24.23 |
| `scratch_beam_search` (this repo, from scratch) | none | **25.45** |
| + Stamping, 4 cycles (oracle ceiling) | phrases **and** their reference positions | 81.70 |
| + **Grid Beam Search**, 4 cycles (deployable) | bare phrases only | **40.59** |

In every constrained run, **100% of forced phrases actually appear in the output** — verified at the token-ID level and in the detokenized text. Grid Beam Search gains **+15 BLEU** over the baseline using only information a real user could supply.

## Try it yourself

Two ways in, depending on how far you want to go — a zero-setup demo, or the full research notebook that reproduces every number below.

### The interactive demo — no setup

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anupreetsingh/Nothing-Lost-in-Translation/blob/main/DEMO.ipynb)

Type an English sentence, pick Russian phrases that **must** appear (or must **not**), and watch the constrained decoder build a fluent translation around your requirements. That is the whole project in one interactive cell: translation as a *controllable* process instead of a black box.

[DEMO.ipynb](DEMO.ipynb) is a self-contained, three-cell playground built on the same Grid Beam Search decoder as the research notebook — open the Colab link and run the cells, nothing to install.

### The full research notebook — reproduce the results

[Final Research Draft.ipynb](Final%20Research%20Draft.ipynb) rebuilds the decoder from the model's raw forward pass and runs every experiment behind the tables below. Open it locally (create a virtual environment and select it as the kernel — Section 0 of the notebook walks through it) or upload it to Colab, then run the cells top to bottom: the first cell installs every dependency, the device (CUDA / Apple Silicon MPS / CPU) is auto-detected, and the WMT16 data downloads itself. Nothing needs editing to run.

The experiment's scale and scope can be shifted in one **Run configuration** cell subsection under Section-3 of the notebook by manipulating the following values:

| Knob | Default | Effect |
|------|---------|--------|
| `N_EVAL` | 100 | Evaluation sentences; raise for more stable BLEU, lower for speed |
| `CYCLES` | 4 | Pick–revise rounds (one new constraint per sentence per round) |
| `NUM_BEAMS` | 4 | Beam width; Hokamp & Liu used 10 |
| `MINE_MODE` | `"word"` | Mining granularity: whitespace words or raw token IDs |

## Repository layout

```
Nothing-Lost-in-Translation/
├── Final Research Draft.ipynb   # The research notebook: baselines, decoder rebuild,
│                                # constraint mining, stamping, Grid Beam Search,
│                                # negative constraints
├── DEMO.ipynb                   # Interactive demo: your sentence, your required /
│                                # blocked Russian phrases (Colab-ready)
├── assets/
│   └── grid-beam-search.svg     # Grid Beam Search diagram used in this README
└── GA1 Submission/
    ├── GA1.ipynb                # Original coursework notebook (historical record)
    └── GA1.pdf                  # Original written report
```

---

## Table of contents

- [The problem](#the-problem)
- [Background: from greedy decoding to beam search](#background-from-greedy-decoding-to-beam-search)
- [The journey, end to end](#the-journey-end-to-end)
  - [1. Data and model selection](#1-data-and-model-selection)
  - [2. A baseline to beat](#2-a-baseline-to-beat)
  - [3. The tokenizer detail that decides everything](#3-the-tokenizer-detail-that-decides-everything)
  - [4. Rebuilding beam search from scratch](#4-rebuilding-beam-search-from-scratch)
  - [5. Mining constraints: the pick–revise loop](#5-mining-constraints-the-pickrevise-loop)
  - [6. Track A — Stamping, the oracle ceiling](#6-track-a--stamping-the-oracle-ceiling)
  - [7. Track B — Grid Beam Search, the deployable method](#7-track-b--grid-beam-search-the-deployable-method)
  - [8. Negative constraints: keeping phrases out](#8-negative-constraints-keeping-phrases-out)
- [Results in detail](#results-in-detail)
- [Why lexical constraints matter in practice](#why-lexical-constraints-matter-in-practice)
- [Tech stack](#tech-stack)
- [Limitations and future work](#limitations-and-future-work)
- [Author and references](#author-and-references)

---

## The problem

A translation model's decoder is usually a black box: you call `.generate()` and trust whatever beam search hands back. Most of the time that is fine. But real work often comes with *hard requirements* — a medical term that must not be paraphrased, a product name that must stay in its official form, a glossary the output must comply with, a word the style guide bans. Plain beam search offers no lever for any of this: an output can be fluent and high-probability while still dropping the one phrase that mattered.

This project builds that lever — and because it lives at the decoding layer, *below* the model, it is more general than the translation task used to demonstrate it. The decoder manipulates the search over tokens, not the weights that produce them; it reasons only about token IDs and per-step scores from a forward pass. So nothing about it is specific to Russian, or even to translation. If phrase inclusion can be **guaranteed** here, the same machinery enforces constraints in any language pair, and more broadly imposes lexical constraints on *any* autoregressive generation — summarization, structured or schema-bound output, controlled text — wherever the result must provably contain, or exclude, certain spans. EN→RU is the concrete testbed; **constraint satisfaction over a decoder** is the general method.

The path runs from the model's raw forward pass, through a from-scratch beam search verified against the library, to two constrained decoders — one that cheats to establish the ceiling, and one, Grid Beam Search that could actually ship.

## Why lexical constraints matter in practice

Constrained decoding turns translation from "hope the model preserves it" into "the decoder must include it." Concrete use cases include:

- **Domain-specific jargon (legal, medical, technical)** — preserve terms whose mistranslation changes meaning or creates safety issues.
    Translating *"The patient presented to the emergency department with bilateral lower extremity edema and dyspnea…"*, requiring «двусторонний отек нижних конечностей» (bilateral lower extremity edema) and «одышка» (dyspnea) holds the clinical terms in place, where an unconstrained model tends to soften them to laymen terms slike «отек обеих ног» (swelling of both legs) or «затрудненное дыхание» (difficulty breathing) for purpose of fluency.
- **Named entities** — pin people and places to one standard transliteration.
    Translating *"Jacinda Ardern delivered a speech to some officials from Wellington,"* requiring «Джасинда Ардерн» (Jacinda Ardern) and «Веллингтон» (Wellington) fixes the name and city to one spelling instead of variants like «Ясинда Ардерн» (Yasinda Ardern, wrong first letter) or «Уэллингтон» (Wellington, variant spelling).
- **Brand and product consistency** — keep brand and product names in their official Latin-script form.
    Translating *"Apple introduced new spatial computing features for Vision Pro…"*, requiring «Apple» (Apple) and «Vision Pro» (Vision Pro) blocks a Cyrillic transliteration like «Эппл» (Apple) / «Вижн Про» (Vision Pro) — or the literal «Яблоко» (apple, the fruit).
- **Glossary compliance** — force the approved glossary term when the source word has several plausible translations.
    Translating *"The merchant received a chargeback after the customer disputed the card transaction,"* requiring the fintech term «чарджбэк» (chargeback) stops the model reaching for a generic paraphrase like «возврат платежа» (payment refund).
- **Content sanitization** — the negative direction: keep harsh or unwanted wording out of the output.
    Translating *"The customer called the delayed rollout a stupid mess and said the support team was useless,"* blocking «тупой» (stupid) and «бесполезный» (useless) makes the decoder fall back on milder wording such as «неудачный» (unsuccessful) or «неэффективный» (ineffective).

## Background: from greedy decoding to beam search

A sequence-to-sequence model does not emit a translation in one go. The source sentence flows through the encoder once; the decoder then builds the output token by token, and at each step the model scores every vocabulary token as a possible continuation. The *decoding algorithm* decides which partial translations live on.

- **Greedy decoding** keeps exactly one candidate: take the top-scoring token at each step and never look back. Fast, but a locally best token is not always part of the best complete sentence — good paths get discarded before they can pay off.
- **Exhaustive search** would score every possible sequence: `|V|^T` candidates for vocabulary size `|V|` and length `T`. At 50,000 tokens and 20 steps that is `50,000^20` sequences — never practical.
- **Beam search** is the working compromise: keep the top `k` partial hypotheses after every step, expanding roughly `T × k × |V|` candidates in total. It recovers from locally imperfect choices without attempting the impossible.

Beam search is what `.generate()` runs under the hood. But it still has no notion of *requirements*: nothing in its scoring protects a low-probability-but-mandatory phrase from being pruned, and nothing bans an unwanted one. Adding those controls means owning the loop — which is where this project starts.

## The journey, end to end

The full narrative, with code and annotated commentary, lives in [Final Research Draft.ipynb](Final%20Research%20Draft.ipynb). This is the map.

### 1. Data and model selection

The dataset is [WMT16](https://huggingface.co/datasets/wmt/wmt16) EN→RU: ~1.5M training pairs and 2,998 validation pairs, loaded via `datasets`.

The first modeling attempt was fine-tuning `google/mt5-base` — a multilingual model that is not a translator out of the box. After hours of training it still trailed a purpose-built translation model, and the redirect was cheap: everything this project investigates concerns *decoding*, not the model producing the logits. The winner was **`Helsinki-NLP/opus-mt-en-ru` (MarianMT)** — small (~300 MB), purpose-built for the pair, strong with no training at all. The mt5 pipeline stays in the notebook behind a flag as a documented dead end.

### 2. A baseline to beat

Before replacing the decoder, establish the number it must reproduce: the library's own beam search (`model.generate()`, 4 beams) on the first 100 validation sentences scores **BLEU 24.23** under sacreBLEU. Every later experiment is read as a delta from this run.

### 3. The tokenizer detail that decides everything

`MarianTokenizer` bundles **two** SentencePiece models: `source.spm` (English) and `target.spm` (Russian). A bare `tokenizer(...)` call always segments with the *source* one. Segment Russian text that way and you get character shrapnel or `<unk>` pieces — token sequences the decoder never actually produces.

This silently broke the project's first constrained-decoding implementation: references, mined phrases, and hypotheses were all encoded with the English segmenter, so every constraint pushed into the decoder was expressed in a vocabulary it never emits. Constraints never landed, the same phrase was re-mined cycle after cycle, and BLEU *fell* when constraints were applied. The fix is one rule enforced everywhere — **every target-side string goes through `encode_target()`**, which routes text through the target SentencePiece model — plus a sanity-check cell that prints both segmentations side by side so the difference can never go invisible again. This was the project's key debugging finding.

### 4. Rebuilding beam search from scratch

`scratch_beam_search` replaces `.generate()` by driving the model's raw forward pass one token at a time. The flow of a decode:

1. **Fan out the input.** Each source sentence's `input_ids` and `attention_mask` are expanded with `repeat_interleave(num_beams)`, one row per hypothesis, all attached to the same source. `decoder_input_ids` starts as one `decoder_start_token_id` per hypothesis.
2. **Seed the beams.** A forward pass returns logits — raw scores over the vocabulary for the next position. `log_softmax` converts them to log-probabilities that can be summed across steps without underflow. Since every hypothesis starts identically, the first expansion simply takes the sentence's top `num_beams` tokens as the initial beam.
3. **Expand.** At each later step, every one of the surviving `num_beams` hypotheses proposes its top `num_beams` continuations (next tokens), chosen from among its own `|V|` candidates by each continuation's log-probability alone — a ranking local to that hypothesis. That puts a `num_beams × num_beams` pool of candidates on the table per source sentence.
4. **Select and bookkeep.** From that pool, candidates compete only against candidates from the same source sentence, now ranked by *running score + continuation log-probability*; the best `num_beams` survive into the next step, each mapped back to the hypothesis it extends. Finished beams are frozen — extended with pad at zero cost — so nothing is generated past EOS.
5. **Terminate.** The loop stops at `max_length`, or earlier with `early_stopping` when the best new token is EOS. The top beam per sentence is decoded back to text.

(Model anatomy worth knowing when reading the code: for the Helsinki model `eos = 0`, `unk = 1`, and `pad = decoder_start = 62517` — the pad ID doubles as the decoder's start token.)

The point is trust, then surgery: the scratch decoder scores **BLEU 25.45**, faithfully reproducing (here slightly exceeding) the library's 24.23 — which makes it a safe foundation. Both constrained decoders below start from this exact loop.

### 5. Mining constraints: the pick–revise loop

With a trustworthy decoder, the experimental question becomes: can we *guarantee* that specific reference phrases appear in the translation? The workflow is **pick–revise**: compare the current translation against the reference → *pick* the longest reference n-gram (up to 3 words) missing from it → make that phrase a hard constraint → *revise* by re-decoding with all constraints accumulated so far → repeat for `CYCLES` rounds, one new constraint per sentence per round.

The mining functions (`mine_word` by default, `mine_token` as an alternative granularity) encode three rules, each of which exists because its absence was a real failure mode in the first implementation:

- **Longest-first (3-gram → 2-gram → 1-gram)** — a longer contiguous phrase carries more information per constraint.
- **Span tracking** — reference spans already used in earlier cycles are skipped; without this, overlapping constraints get re-mined and force duplicated text into the output.
- **Target-side tokenization** — every mined Russian phrase goes through `encode_target()` (see [above](#3-the-tokenizer-detail-that-decides-everything)).

Mining returns each phrase's token IDs *and* its position in the reference. Only stamping consumes the position; Grid Beam Search deliberately ignores it — that distinction is the whole experiment.

### 6. Track A — Stamping, the oracle ceiling

`stamping_beam_search` is the scratch decoder of [section 4](#4-rebuilding-beam-search-from-scratch) with a constraint *schedule* grafted onto its loop — nothing about scoring, beam competition, or frozen finished beams changes, so whatever the BLEU curve does is attributable to the constraints alone. The flow of a decode:

1. **Queue the schedule.** The constraints enter as a queue sorted by their reference position — the second thing mining returned in [section 5](#5-mining-constraints-the-pickrevise-loop), and the piece of information only this track consumes.
2. **Decode normally until a constraint is due.** Each step runs the usual loop — forward pass, `log_softmax`, expand, select — until the step counter reaches the position of the first pending constraint (`step >= position`).
3. **Stamp one token per step.** A due constraint takes over the step: its next token is appended to *every* beam outright, bypassing the top-`num_beams` competition entirely. Emitting one token per step (rather than a whole phrase at once) means constraints with colliding positions queue up behind each other instead of overwriting each other.
4. **Score the forced token anyway.** The stamped token was still assigned a log-probability in the forward pass, and that value joins each beam's running sum — so beams remain comparable with each other, and with the free-running steps around them, on the same scale. (Grid Beam Search applies the same forced-but-still-scored principle to its **start**/**continue** moves.)
5. **Guard the end.** While any constraint is pending, the EOS token's log-probability is masked to `−inf` in every free expansion — a sentence cannot end before its constraints are placed. Once the queue is empty, the normal termination rules take over (`max_length`, or `early_stopping` when the best beam finishes), and the top beam per sentence is decoded back to text.

Its BLEU climbs to 81.70 in four cycles — and that is precisely its flaw. The placement position is **oracle information**: it exists only because the reference translation is sitting there to copy from. In every real use — glossary enforcement, protecting a product name, interactive post-editing — the user knows *which* phrase must appear, never *where* it belongs in a sentence that has not been generated yet. Stamping is kept as a learning exercise and as the upper bound: the score you reach when placement comes for free.

### 7. Track B — Grid Beam Search, the deployable method

`grid_beam_search` implements [Hokamp & Liu (2017)](https://arxiv.org/abs/1704.07138). The search lives on a grid indexed by *(timestep, coverage)*: the horizontal axis is the normal decoding timeline, and the vertical axis is constraint progress — coverage being the number of constraint tokens emitted so far.

![Grid Beam Search search graph](assets/grid-beam-search.svg)

A **cell** `(t, c)` is one search state: *t tokens generated, c of them constraint tokens*. Each cell holds its own beam of up to `num_beams` hypotheses. They share the timestep and the coverage count but differ in everything else — their actual tokens, their scores, even *which* constraints they have covered. Being in the same cell, again, means only sharing a coverage *count*: each hypothesis carries its own record of the specific constraints it has covered and, if it is mid-phrase, which constraint is currently open.

The flow of a decode:

1. **Start in the corner.** The encoder runs exactly once; its states are re-used at every step. The search begins with a single hypothesis in cell `(0, 0)` — nothing generated, nothing covered.

2. **Expand every hypothesis.** At each time step (column), all live hypotheses across every coverage level (row) of that time step are batched through the decoder in **one forward pass**. The successors a hypothesis produces are decided by its state — in Hokamp & Liu's terminology, a hypothesis can be **open**, when it is not in the middle of a multi-token constraint, or **closed**, when it is mid-way through emitting one:
   - **Open hypotheses** take **generate** and **start** together, fanning out into `num_beams + U` successors, where `U` is the number of uncovered constraints:
     - **generate** — top `num_beams` continuations (next tokens), chosen from among its own `|V|` candidates; results in a *horizontal* move to `(t+1, c)`. Skipped in one edge case: when the remaining steps exactly match the remaining constraint tokens, there is no slack left for free tokens.
     - **start** — the first token of an uncovered constraint, one successor per such constraint; results in a *diagonal* move to `(t+1, c+1)`.
   - **Closed hypotheses** take exactly one move:
     - **continue** — the next token of the constraint currently mid-emission; also a *diagonal* move to `(t+1, c+1)`. A closed hypothesis can only continue its phrase — no free tokens, no starting another constraint — until the phrase's last token re-opens it.

    Remember that a forced token (emitted by a **start** or **continue** move) is exempt from the top-`num_beams` cutoff — it reaches the diagonal cell no matter how the model ranked it — but not from scoring: the log-probability it was assigned in the forward pass still joins the hypothesis's running sum.

3. **Re-bucket and prune per cell.** All successors are pooled and re-bucketed by their *new* coverage count into the appropriate cell. A cell at `(t+1, c)` therefore receives arrivals from two directions at once: *horizontal* ones from `(t, c)` (free token from generate move) and *diagonal* ones from `(t, c−1)` (forced token from start or continue move). The hypotheses in each cell then compete on total score — the parent's running score plus the new token's log-probability — and the top `num_beams` *distinct* hypotheses survive for that cell. This **per-cell** pruning is what keeps the contest fair: every hypothesis is judged only against others that have paid for the same number of constraint tokens. A hypothesis mid-way through an improbable required phrase never faces the fluent, free-running hypotheses at lower coverage — the exact competition that would kill it in plain beam search.
4. **Guard the search.** Two important rules make sure a sentence cannot end while any constraint is unplaced.

    First, EOS is masked for every hypothesis below full coverage of constraints — finishing early is simply not a legal move. Masking means that in the hypothesis's generate expansion, the EOS token's log-probability is set to `−inf` before the top-`num_beams` cut, so it can never be selected, however much the model wants to end the sentence.

    Second, a hypothesis is dropped the moment its remaining steps are less than the remaining constraint tokens.

5. **Finish at the top row.** Only in the top row, where every constraint is covered, is EOS unmasked and hypotheses are finally allowed to end. They do not all end at once: whenever a top-row hypothesis emits EOS, it leaves the grid for a *completed pool*, its score frozen, while its siblings keep decoding and may finish later at other lengths in successive time steps.

    When the search ends, that pool is ranked once, globally — the only comparison in the algorithm that crosses timesteps and lengths, which is why length normalization applies exactly here: each hypothesis's score is divided by its length, turning the sum into an average log-probability per token, so a shorter sentence is not rewarded merely for collecting fewer negative terms (log-probabilities). The best normalized hypothesis is the returned translation: a complete sentence that provably covers every constraint, with placement chosen by the model's own scores rather than by any oracle. (And if nothing ever completes — the step budget runs out, or the constraints prove unsatisfiable — the decoder returns the best partial hypothesis from the highest coverage row rather than failing.)

The cost: with `c` constraint-coverage states, roughly `T × c × k × |V|` expansions instead of beam search's `T × k × |V|` — more expensive, still nowhere near exhaustive search's `|V|^T`. The property that matters: constraints enter as **bare phrases** — no positions anywhere in the interface. Exactly the information a real user could supply.

### 8. Negative constraints: keeping phrases out

The mirror question arrives immediately in any real use: how do we keep a phrase *out*? Profanity that should give way to a euphemism, a competitor's name, an anglicism the style guide forbids.

The same `grid_beam_search` accepts a `negative_constraints` argument, and the mechanism needs **no new grid dimension**. The coverage grid of [section 7](#7-track-b--grid-beam-search-the-deployable-method) exists to *protect* improbable-but-required progress from beam pruning; a ban protects nothing — it only removes options — so it lives entirely inside the existing loop as one more masking rule, at negligible cost. The flow of a ban:

1. **Encode the blocklist.** Each banned phrase goes through `encode_target()` (as always — [section 3](#3-the-tokenizer-detail-that-decides-everything)) into a banned token-ID sequence. Matching is exact at the token level, so each surface form to block — case, inflection, capitalization — needs its own entry.
2. **Watch every hypothesis's tail.** At each **generate** expansion, the decoder computes which next tokens would *complete* a banned sequence given that hypothesis's current tail: a token is blocked when the hypothesis's most recent tokens already spell out all of a banned sequence except its last token, and this token is that last piece. (A single-token ban has an empty prefix, so it is blocked everywhere.)
3. **Mask before the cut.** Each blocked token's log-probability is set to `−inf` before the top-`num_beams` selection — the same mechanism section 7 uses to mask EOS — so it can never be chosen. The probability mass the model gave the banned continuation falls to its runner-up, and because the top of the distribution at that point is dominated by alternative wordings of the same source meaning, the runner-up for a harsh word is usually a milder synonym.
4. **Guard the forced moves too.** A **start** or **continue** move bypasses the top-k, so it gets its own check: if the forced token would complete a banned sequence, that successor is dropped rather than emitted. A required phrase can never smuggle a banned one into the output.

Negative constraints sit outside the BLEU experiments by design: pick–revise mines phrases to *require* from the reference, and a reference offers nothing to *ban* — banning reference phrases would push BLEU down by construction, measuring nothing. Section 8 of the notebook instead demonstrates them the way they would actually be used — ban the anglicism «Мистер», let the decoder fall back to its next-best honorific, optionally pair the ban with a positive constraint («Господин Моди») to *direct* the replacement instead of trusting the model's ranking, and assert at the token level that the banned sequence is truly absent from the output.

Three honest caveats: the match is exact, so each surface form of a banned word needs its own blocklist entry; banning selects *against* a phrase, never *for* its replacement — if the runner-up is also unwanted, the fix is composing the ban with a positive constraint, as above; and a positive constraint that *contains* a banned sequence makes the two unsatisfiable together, leaving the search only its fallback hypothesis — don't require what you ban.

## Results in detail

First 100 sentences of the WMT16 EN→RU validation split, `NUM_BEAMS = 4`, word-level mining, 4 pick–revise cycles, scored with sacreBLEU. Both tracks start from the custom decoder's 25.45 baseline; each cycle mines one new missing reference phrase per sentence from that track's own previous output and re-decodes with all constraints so far:

| Cycle | Stamping BLEU | GBS BLEU | Constraints satisfied (both tracks) |
|-------|---------------|----------|-------------------------------------|
| baseline | 25.45 | 25.45 | — |
| 1 | 38.68 | 31.81 | 100% |
| 2 | 56.20 | 38.41 | 100% |
| 3 | 71.95 | 40.40 | 100% |
| 4 | **81.70** | **40.59** | 100% |

(Constraints mined per cycle fall from 99 sentences in cycle 1 to ~60 by cycle 4, as translations progressively recover the reference.)

How to read the two curves:

- **Constraints actually land.** In every cycle of both tracks, 100% of forced phrases appear in the output, verified both at the token-ID level and in the detokenized text. The project's first, bugged implementation could not achieve this — see [the tokenizer section](#3-the-tokenizer-detail-that-decides-everything).
- **Both curves rise** — adding hard constraints *improves* translation quality rather than trading it away.
- **Stamping's curve is a ceiling, not a victory.** It consumes oracle positions copied from the reference, so its BLEU measures how much reference information was injected — not quality any user could obtain.
- **GBS's curve is the honest one.** It rises from 25.45 to 40.59 (**+15 BLEU**) using only the phrases themselves, while the model keeps their placement fluent. The gap between the curves is the price of not cheating.

**Cost.** Stamping runs at roughly the speed of unconstrained search (~0.4 s/sentence on a Colab T4). GBS grows with accumulated constraints — from ~1.9 s/sentence in cycle 1 to ~5.2 s in cycle 4 — because it maintains a beam per coverage level. GBS outputs also lengthen across cycles (mean ~32 → ~49 tokens), partly legitimate recovery of reference content, partly verbosity worth auditing.

## Tech stack

| Component | Choice |
| --- | --- |
| Translation model | `Helsinki-NLP/opus-mt-en-ru` (MarianMT, 6+6 layers, d=512, vocab 62,518) |
| ML framework | PyTorch + Hugging Face `transformers` / `accelerate` |
| Data | WMT16 `ru-en` via `datasets` |
| Evaluation | `sacrebleu` |
| Tokenization | `sentencepiece` — separate source/target models; all target text via `encode_target()` |
| Historical | `google/mt5-base` fine-tuning attempt, preserved behind a flag |

## Limitations and future work

- **Constraints are mined from gold references** — the research loop answers "does constrained decoding work?", not a deployment scenario. [DEMO.ipynb](DEMO.ipynb) closes part of the gap by taking user-supplied phrases; a glossary-driven pipeline is the natural next step.
- **GBS is slow, and slows as constraints accumulate** — ~1.9 s/sentence with one constraint rising to ~5.2 s with four. Batching multiple sentences through GBS at once is unimplemented.
- **Outputs lengthen under GBS** — mean length grows ~32 → ~49 tokens across four cycles; some is legitimate recovery of reference content, some may be verbosity worth auditing.
- **Modest evaluation scale** — 100 sentences, 4 beams (the paper used 10), and only the EN→RU pair is measured; the decoder is language- and task-agnostic (see [The problem](#the-problem)), but that generality is argued, not yet benchmarked across pairs. Scale and beam width are single-knob changes (`N_EVAL`, `NUM_BEAMS`) trading runtime for fidelity.
- **Exact-match negative constraints** — each surface form (case, inflection) of a banned word needs its own entry; production use would expand lemmas or add a detokenized substring check.

## Author and references

**Anupreet Singh** — sole author. Designed and built the project end to end: the from-scratch decoder verified against `.generate()` (25.45 vs 24.23 BLEU), both constrained decoders (stamping and Grid Beam Search with coverage-bucketed beams, feasibility pruning, and length-normalized selection — 25.45 → 40.59 BLEU at 100% constraint satisfaction), the negative-constraint extension, the pick–revise mining loop, the diagnosis of the dual-SentencePiece tokenizer bug, the WMT16 evaluation pipeline, the mt5 comparison, the interactive demo, and the annotated write-up.

**Reference:** Chris Hokamp and Qun Liu. 2017. [*Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*](https://arxiv.org/abs/1704.07138). ACL 2017.
