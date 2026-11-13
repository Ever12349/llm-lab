#!/usr/bin/env python3
"""演示默认 HF 词表（Qwen/Qwen2.5-1.5B）的 encode / decode。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TrainConfig
from tokenizer import HFTokenizer


def main() -> int:
    cfg = TrainConfig()
    local = Path(cfg.hf_cache_dir) / "Qwen2.5-1.5B-tokenizer"
    src = str(local) if local.is_dir() else cfg.hf_tokenizer_id
    print(f"loading: {src}")
    tok = HFTokenizer.load(src, cache_dir=cfg.hf_cache_dir)
    print(f"vocab_size: {tok.vocab_size}")
    for label, text in [
        ("ZH", "今天天气真好，我想去公园散步。"),
        ("JA", "今日はいい天気ですね。"),
        ("EN", "The weather is nice today."),
    ]:
        ids = tok.encode(text)
        print(f"{label}: {len(ids)} tok | {ids[:16]}… | {tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
