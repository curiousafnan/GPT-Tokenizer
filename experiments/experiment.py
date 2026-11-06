from src.data import load_training_text
from src.train import train, build_vocab
from src.tokenizer import gpt2_encode, gpt2_decode


def measure_compression(training_text, ids_gpt2):
    """Characters in the training text vs token ids after merging."""
    total_ids = sum(len(piece) for piece in ids_gpt2)
    return len(training_text), total_ids, len(training_text) / total_ids


def run_roundtrip(gpt_merges, gpt2_vocab, texts):
    """Encode then decode each text, check it comes back unchanged."""
    results = []
    for text in texts:
        ids = gpt2_encode(text, gpt_merges, allowed_special=True)
        restored = gpt2_decode(ids, gpt2_vocab)
        results.append({"text": text, "ids": ids, "ok": restored == text})
    return results


def run_special_token_check(gpt_merges, gpt2_vocab, text="hi<|endoftext|>"):
    """Same text with allowed_special on and off."""
    allowed = gpt2_encode(text, gpt_merges, allowed_special=True)
    not_allowed = gpt2_encode(text, gpt_merges, allowed_special=False)
    return {
        "allowed": allowed,
        "not_allowed": not_allowed,
        "roundtrip_ok": gpt2_decode(not_allowed, gpt2_vocab) == text,
    }


if __name__ == "__main__":
    training_text = load_training_text()
    print("Total characters:", len(training_text))
    print("Total UTF-8 bytes:", len(training_text.encode("utf-8")))

    gpt_merges, ids_gpt2 = train(training_text)
    gpt2_vocab = build_vocab(gpt_merges)

    chars, total_ids, ratio = measure_compression(training_text, ids_gpt2)
    print("text length:", chars)
    print("ids length:", total_ids)
    print(f"compression ratio: {ratio:.2f}X")
    print()

    for r in run_roundtrip(gpt_merges, gpt2_vocab, [
        "Hello world! I'm learning GPT-2. <|endofprompt|>",
        "ہم کو تم سے پیار ہے",
        "زه له تاسره مینه لرم",
        "Hello 👋 World 🌍",
    ]):
        print("Text:", r["text"])
        print("IDs:", r["ids"])
        print("Decoded correctly:", r["ok"])
        print()

    example = "Hello world! سلام دنیا"
    print("Regex BPE token count:", len(gpt2_encode(example, gpt_merges)))
    print(run_special_token_check(gpt_merges, gpt2_vocab))
