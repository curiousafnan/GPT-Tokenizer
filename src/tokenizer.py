from src.bpe import get_stats, merge
from src.config import id_to_special, special_tokens
from src.pretokenize import pre_tokenize


def gpt2_encode(text, gpt_merges, allowed_special=False):
    chunks = pre_tokenize(text)
    all_ids = []
    for chunk in chunks:
        if chunk in special_tokens and allowed_special:
            all_ids.append(special_tokens[chunk])
            continue
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats, _ = get_stats(ids)
            pair = min(stats, key=lambda p: gpt_merges.get(p, float("inf")))
            if pair not in gpt_merges:
                break
            idx = gpt_merges[pair]
            ids = merge(ids, pair, idx)
        all_ids.extend(ids)
    return all_ids


def gpt2_decode(ids, gpt2_vocab):
    result = bytearray()
    for idx in ids:
        if idx in id_to_special:
            result.extend(id_to_special[idx].encode("utf-8"))
        else:
            result.extend(gpt2_vocab[idx])

    return result.decode("utf-8", errors="replace")
