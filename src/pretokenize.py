from src.config import pattern, special_pattern, special_tokens


def pre_tokenize(text):
    parts = special_pattern.split(text)
    chunks = []
    for part in parts:
        if part in special_tokens:
            chunks.append(part)
        else:
            chunks.extend(pattern.findall(part))
    return chunks
