from src.bpe import get_stats, merge
from src.config import num_merges as default_num_merges, special_tokens
from src.pretokenize import pre_tokenize


def train(training_text, num_merges=default_num_merges, verbose=True):
    ids_gpt2 = []
    chunks = pre_tokenize(training_text)
    for chunk in chunks:
        if chunk in special_tokens:
            ids_gpt2.append([special_tokens[chunk]])
            continue
        ids_gpt2.append(list(chunk.encode("utf-8")))

    gpt_merges = {}
    for i in range(num_merges):
        stats = {}
        for piece in ids_gpt2:
            if len(piece) < 2:
                continue
            piece_stats, _ = get_stats(piece)
            for pair, count in piece_stats.items():
                stats[pair] = stats.get(pair, 0) + count
        if not stats:
            break
        top_pair = max(stats, key=stats.get)
        idx = 256 + i
        if verbose:
            print(f"merging {top_pair} into new token {idx}")
        ids_gpt2 = [merge(piece, top_pair, idx) for piece in ids_gpt2]
        gpt_merges[top_pair] = idx

    return gpt_merges, ids_gpt2


def build_vocab(gpt_merges):
    gpt2_vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in gpt_merges.items():
        gpt2_vocab[idx] = gpt2_vocab[p0] + gpt2_vocab[p1]
    return gpt2_vocab
