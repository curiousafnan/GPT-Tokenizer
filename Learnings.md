# Learnings — Day 48: GPT-2 Tokenizer

## Key Concepts Learned

### Bytes Make the Tokenizer Lossless
Because the base vocab is all 256 byte values, any text can be encoded. Urdu, Pashto and emoji round-trip exactly even though the training data was English books.

### Merges Only Help What They Were Trained On
English sentences got shorter after training, but Urdu and Pashto stayed close to 2 ids per character. Their byte pairs never showed up in the training text, so no merges exist for them. The training corpus decides which languages get compressed.

### Merges Build on Merges
`" the"` (263) is a merge of `" t"` (256) and `"he"` (257). The vocab grows into longer and longer pieces by combining earlier tokens, not by learning whole words directly.

### The Corpus Leaks Into the Vocab
Merge 267 is `(226, 128)`, the shared first two UTF-8 bytes of curly quotes `’ “ ”` and the em dash `—`. Gutenberg books use these heavily, so the tokenizer spent one of only 14 merges on them.

### Encoding Must Follow Training Order
Encoding applies the earliest-learned merge first, not the most frequent pair in the input. Using a different order would produce ids the model never saw during training.

### Special Tokens Need Their Own Path
Special tokens have to be split out before the regex, or they get chopped into pieces. And the `allowed_special` flag decides if a literal `<|endoftext|>` in user text becomes the control token or plain text.

## Results
14 merges on ~1.9M characters gave a 1.18x compression ratio. Small, but enough to see the mechanics; real GPT-2 uses 50,000 merges.
