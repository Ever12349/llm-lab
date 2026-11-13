#!/usr/bin/env python3
"""演示 BPE encode/decode：展示多字 token，并做往返校验。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "checkpoints" / "tokenizer_bpe.json",
    )
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    print(f"loaded: {args.tokenizer}")
    print(f"vocab_size: {tok.get_vocab_size()}")
    print()

    samples = [
        "化物语是轻小说。",
        "今天天气真好，我们去散步吧。",
        "阿丽思掉进了兔子洞。",
        "The quick brown fox jumps over the lazy dog.",
        "Alice was beginning to get very tired of sitting by her sister.",
        "今天 we go to the library 一起看书。",
    ]
    for text in samples:
        enc = tok.encode(text)
        pieces = enc.tokens
        ids = enc.ids
        decoded = tok.decode(ids)
        multi = [p for p in pieces if len(p) > 1 and not (p.startswith("[") and p.endswith("]"))]
        print(f"text:    {text!r}")
        print(f"tokens:  {pieces}")
        print(f"ids:     {ids}")
        print(f"multi:   {multi}")  # 长度>1 的子词
        print(f"decoded: {decoded!r}")
        print(f"roundtrip_ok: {decoded == text}")
        print()


if __name__ == "__main__":
    main()
