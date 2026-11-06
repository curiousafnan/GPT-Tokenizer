# Approach — Day 48: GPT-2 Tokenizer

## The Algorithm

### Byte-Level Start
Every string is UTF-8 encoded into bytes, so the base vocabulary is fixed at 256 ids (0–255). Any character in any language, including emoji, is just a sequence of these bytes, so there is no "unknown token" case.

### Pair Counting and Merging
`get_stats` counts every adjacent pair in a list of ids. The most frequent pair across the whole corpus gets a new id (`256 + i`), and `merge` replaces every occurrence of that pair with the new id. Repeat `num_merges` times. Because merged ids can themselves be merged later, tokens grow: `" t"` (256) + `"he"` (257) → `" the"` (263).

### Regex Pre-Tokenization
Before BPE, text is split with the GPT-2 pattern:
```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
Pairs are only counted inside each chunk. This stops merges from mixing letters with punctuation or joining the end of one word with the start of the next.

### Encoding Order
When encoding, a chunk may contain several pairs that have merges. The pair with the lowest merge id (learned earliest) is applied first, by taking `min` over pairs using `gpt_merges.get(p, inf)`. This reproduces the same order the merges were created in training. When the best pair has no merge (`inf`), encoding of that chunk stops.

### Decoding
Each id maps to its bytes through the vocab (`vocab[idx] = vocab[p0] + vocab[p1]`), bytes are joined and decoded as UTF-8 with `errors="replace"` so an invalid byte sequence can't crash decoding.

## Design Decisions

### Special Tokens Split First
A second regex splits on special tokens before the GPT-2 pattern runs. Otherwise `<|endoftext|>` would be broken into `<`, `|`, `endoftext`, `|>`. During training each special token becomes a single-id piece, which is skipped by the `len(piece) < 2` check, so it never takes part in merges.

### `allowed_special` Flag
Encoding treats special strings as special only when `allowed_special=True`. With it off, the same text is encoded as normal bytes. This matters when user text happens to contain `<|endoftext|>` and should not be able to inject a control token.

### Module Layout
Notebook functions moved into `src/` unchanged. The only change is that the notebook globals (`gpt_merges`, `gpt2_vocab`) are passed in as arguments to `gpt2_encode` / `gpt2_decode`, and the training cell is wrapped in `train()` which returns them.
