# Problem Statement — Day 48: GPT-2 Tokenizer

## Problem
Build a GPT-2 style byte-level Byte Pair Encoding (BPE) tokenizer from scratch: train merges on real text, encode any string into token ids, decode them back losslessly, and handle special tokens.

## Core Questions
- Why start from bytes instead of characters or words?
- Why does GPT-2 split text with a regex before running BPE?
- How should encoding pick which merge to apply when several pairs in a chunk are mergeable?
- How do special tokens like `<|endoftext|>` stay as one id without being broken apart by the regex or BPE?

## Requirements
- `get_stats` to count adjacent pairs and `merge` to replace a pair with a new id
- GPT-2 regex pre-tokenization with special tokens split out first
- Training loop that counts pairs across all chunks (never across chunk boundaries) and records merges in order
- Vocab built from merges, `gpt2_encode` and `gpt2_decode`
- `allowed_special` flag controlling whether special strings map to their reserved ids
- Train on 4 Project Gutenberg books and report compression ratio

## Success Criteria
- Exact round-trip on English, Urdu, Pashto and emoji text
- Special token encodes to a single id when allowed, and as plain bytes when not
- Merges learned in a sensible order (common English pairs like `" t"`, `"he"`, `" the"` first)
