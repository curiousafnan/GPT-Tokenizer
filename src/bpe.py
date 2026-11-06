def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    top_pair = max(counts, key=counts.get)
    return counts, top_pair


def merge(ids, pair, idx):
    new_id = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_id.append(idx)
            i += 2
        else:
            new_id.append(ids[i])
            i += 1
    return new_id
