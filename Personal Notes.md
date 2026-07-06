# Personal Notes

## Bugs

### The dual-SentencePiece tokenizer bug (constraints never landed)

**What went wrong.**
The first constrained-decoding implementation failed silently. Constraints were pushed into the decoder, the code ran without errors, translations came out — but:

- Forced phrases never actually appeared in the output.
- The same phrase was re-mined cycle after cycle (it was always still "missing").
- BLEU went *down* when constraints were added — the opposite of the expected effect.

**Root cause.**
For a constraint to work, the pipeline is:

```
Russian constraint → tokenizer → token IDs → decoder forces those IDs → output text
```

The decoder produces token IDs that correspond to russian token so the constraint's token IDs are expected to be same tokens russian origin token IDs that the decoder naturally produces and the ones we are supposed to intercept in intermittent time steps to apply the contraints.

SentencePiece Model(SPM) is a tokenizer model that learns how to split the text into tokens.

`MarianTokenizer` bundles **two** SentencePiece models:

- `source.spm` — trained on English, meant to be used for corresponding English text.
- `target.spm` — trained on Russian, meant to be used for corresponding Russian text.

A bare `tokenizer("одышка")` call silently uses the **English** SPM. An English tokenizer shreds Russian Cyrillic constraints into character shrapnel and `<unk>` pieces — e.g. `[812, 4, 4, 199, ...]` — so decoder machinery responsible for implementing constraints try to plug in `[812, 4, 4, 199, ...]` instead of the required Russian token IDs `[2341, 87]` of the contraint.

So the constraint machinery was faithfully doing its job — but `[812, 4, 4, 199]` is most probably a sequnece that the decoder **can never even emit** because it is a bunch of shrapenl and `<unk>` and not russian tokens part of the models vocabulary.

**The fix.**

1. **One chokepoint:** a single helper, `encode_target()`, that explicitly routes contsraint phrase through the target (Russian) SPM.
2. **A visible check:** a sanity-check cell in the notebook that tokenizes the same Russian phrase both ways and prints the two segmentations side by side, so the failure mode can never go invisible again.

**Impact of the fix.**
Constraint satisfaction went from effectively 0% to **100% in every experiment** (verified at both the token-ID level and in the detokenized text), turning a BLEU regression into the +15 BLEU gain (25.45 → 40.59).
