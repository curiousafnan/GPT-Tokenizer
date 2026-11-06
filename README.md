# Day 48 — GPT-2 Tokenizer (Byte-Level BPE)

## Key Insight
A tokenizer is just a compression algorithm learned from data. Starting from 256 raw bytes means nothing is ever "unknown": English, Urdu, Pashto and emoji all round-trip perfectly. But merges only help the text they were trained on. After 14 merges on English books, English compresses a little, while Urdu and Pashto stay at roughly 2 tokens per character because no merges were ever learned for their byte pairs.

## Quick Start
```python
from src.train import train, build_vocab
from src.tokenizer import gpt2_encode, gpt2_decode

gpt_merges, _ = train(training_text)
gpt2_vocab = build_vocab(gpt_merges)

ids = gpt2_encode("Hello world!<|endoftext|>", gpt_merges, allowed_special=True)
text = gpt2_decode(ids, gpt2_vocab)
```

Run the full experiment (downloads 4 Gutenberg books, trains, tests round-trips):
```
python -m experiments.experiment
```

## How It Works
```
text
 → split on special tokens          (<|endoftext|> etc. kept whole)
 → GPT-2 regex pre-tokenize         ('s|'t| ?\p{L}+| ?\p{N}+| ...)
 → UTF-8 bytes per chunk            (ids 0–255)
 → train: count pairs across all chunks, merge most frequent → new id 256, 257 ...
 → encode: keep applying the earliest-learned merge in the chunk until none apply
 → decode: id → bytes → UTF-8 string
```
Merges never cross chunk boundaries, so a token can't swallow "dog." or join two words.

## Experiment Results — vocab_size 270 (14 merges), 4 Gutenberg books
| Metric | Value |
|---|---|
| Training characters | 1,886,725 |
| Training UTF-8 bytes | 1,918,390 |
| Token ids after merges | 1,602,710 |
| Compression ratio | 1.18x |

First merges learned:
```
(32, 116)   → 256   " t"
(104, 101)  → 257   "he"
(32, 97)    → 258   " a"
(105, 110)  → 259   "in"
(256, 257)  → 263   " the"   ← merge built on two earlier merges
(226, 128)  → 267   shared prefix of ’ “ ” — in the books
```

## Round-Trip Check
| Input | Decoded correctly |
|---|---|
| `Hello world! I'm learning GPT-2. <\|endofprompt\|>` | ✓ |
| `ہم کو تم سے پیار ہے` (Urdu) | ✓ |
| `زه له تاسره مینه لرم` (Pashto) | ✓ |
| `Hello 👋 World 🌍` | ✓ |

## Special Tokens
| Token | ID |
|---|---|
| `<\|endoftext\|>` | 270 |
| `<\|endofprompt\|>` | 271 |
| `<\|pad\|>` | 272 |
| `<\|startoftext\|>` | 273 |

`allowed_special=True` → special string becomes one id. `allowed_special=False` → it's encoded as normal text bytes.

## Structure
```
src/
  config.py        vocab size, regex pattern, special tokens
  bpe.py           get_stats, merge
  pretokenize.py   pre_tokenize
  train.py         train, build_vocab
  tokenizer.py     gpt2_encode, gpt2_decode
  data.py          load_training_text (Gutenberg)
experiments/
  experiment.py    compression, round-trip and special token experiments
tests/
  test.py
exp_gpt2_full_working.ipynb   original working notebook
```

## Tests
```
pip install -r requirements.txt
python -m pytest tests/test.py
```
