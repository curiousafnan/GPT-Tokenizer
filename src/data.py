import ssl
import urllib.request

import certifi

urls = [
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
]


def load_training_text(urls=urls):
    context = ssl.create_default_context(cafile=certifi.where())
    texts = []
    for url in urls:
        response = urllib.request.urlopen(url, context=context, timeout=30)
        text = response.read().decode("utf-8-sig")
        texts.append(text)
    training_text = "\n\n".join(texts) + "<|endoftext|>"
    return training_text
