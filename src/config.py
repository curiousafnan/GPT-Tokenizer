import regex as re

vocab_size = 270
current_size = 256
num_merges = vocab_size - current_size

pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

special_tokens = {
    "<|endoftext|>": 270,
    "<|endofprompt|>": 271,
    "<|pad|>": 272,
    "<|startoftext|>": 273,
}

special_pattern = re.compile(
    "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
)

id_to_special = {
    idx: token for token, idx in special_tokens.items()
}
