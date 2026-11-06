import pytest

from src.pretokenize import pre_tokenize
from src.config import special_tokens
from src.train import train, build_vocab
from src.tokenizer import gpt2_encode, gpt2_decode

# Small local corpus so tests run offline and fast
CORPUS = (
    "the cat sat on the mat. the other cat sat there too. "
    "she said they're here and we'll see what they've done.\n\n"
) * 50 + "<|endoftext|>"


@pytest.fixture(scope="module")
def trained():
    gpt_merges, _ = train(CORPUS, verbose=False)
    return gpt_merges, build_vocab(gpt_merges)


def test_pre_tokenize():
    text = "Hello world!<|endoftext|>How are you?"
    assert pre_tokenize(text) == [
        "Hello", " world", "!", "<|endoftext|>", "How", " are", " you", "?"
    ]


@pytest.mark.parametrize("text", [
    "Hello world! I'm learning GPT-2. <|endofprompt|>",
    "ہم کو تم سے پیار ہے",
    "زه له تاسره مینه لرم",
    "Hello 👋 World 🌍",
])
def test_roundtrip(trained, text):
    gpt_merges, gpt2_vocab = trained
    ids = gpt2_encode(text, gpt_merges, allowed_special=True)
    assert gpt2_decode(ids, gpt2_vocab) == text


def test_special_token_allowed(trained):
    gpt_merges, _ = trained
    ids = gpt2_encode("hi<|endoftext|>", gpt_merges, allowed_special=True)
    assert ids[-1] == special_tokens["<|endoftext|>"]


def test_special_token_not_allowed(trained):
    gpt_merges, gpt2_vocab = trained
    text = "hi<|endoftext|>"
    ids = gpt2_encode(text, gpt_merges, allowed_special=False)
    assert special_tokens["<|endoftext|>"] not in ids
    assert gpt2_decode(ids, gpt2_vocab) == text


def test_merges_compress(trained):
    gpt_merges, _ = trained
    text = "the cat sat on the mat"
    assert len(gpt2_encode(text, gpt_merges)) < len(text.encode("utf-8"))
