#!/usr/bin/env python3
"""流式下载高质量中文语料（P0/P1）到 data/seed_v2_zh_hq_raw/。

按「字符预算」截断，避免下载时反复跑 tokenizer 抢 CPU。
支持 --skip-docs 续拉（跳过已下载的 HF 样本）。
默认串行；可用 HF_ENDPOINT=https://hf-mirror.com。
不覆盖训练用 seed_v2_zh_clean_*.bin。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "key": "wiki_pretrain_zh",
        "outfile": "wiki_pretrain_zh.txt",
        "hf_id": "yuhuanstudio/wikipedia-pretrain-zh",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": "P0",
        "default_char_budget": 120_000_000,
    },
    {
        "key": "stories_zh",
        "outfile": "stories_zh.txt",
        "hf_id": "richarddzh/chinese-small-lm-corpus",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": "P0",
        "default_char_budget": 60_000_000,
    },
    {
        "key": "skypile_zh",
        "outfile": "skypile_zh.txt",
        "hf_id": "Skywork/SkyPile-150B",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": "P1",
        "default_char_budget": 60_000_000,
    },
    {
        "key": "wiki_zh_wikimedia",
        "outfile": "wiki_zh_wikimedia.txt",
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "text_field": "text",
        "priority": "P0",
        "default_char_budget": 80_000_000,
    },
]


@dataclass
class Stats:
    key: str
    hf_id: str
    outfile: str
    char_budget: int
    skip_docs: int = 0
    chars: int = 0
    docs: int = 0
    skipped: int = 0
    seconds: float = 0.0
    status: str = "pending"
    error: str = ""
    est_tokens: int = 0


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").strip()
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def _iter_docs(
    hf_id: str, config: Optional[str], split: str, text_field: str
) -> Iterator[str]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(hf_id, config, split=split, streaming=True)
    else:
        ds = load_dataset(hf_id, split=split, streaming=True)
    for ex in ds:
        raw = ex.get(text_field) or ""
        if not isinstance(raw, str):
            continue
        text = _clean_text(raw)
        if len(text) < 40:
            continue
        yield text


def download_one(
    src: dict,
    out_path: Path,
    char_budget: int,
    *,
    skip_docs: int = 0,
    min_chars: int = 80,
) -> Stats:
    st = Stats(
        key=src["key"],
        hf_id=src["hf_id"],
        outfile=src["outfile"],
        char_budget=char_budget,
        skip_docs=skip_docs,
    )
    if char_budget <= 0:
        st.status = "skipped"
        return st
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    print(
        f"→ {src['key']}: {src['hf_id']} skip={skip_docs:,} "
        f"budget≈{char_budget:,} chars → {out_path}",
        flush=True,
    )
    t0 = time.perf_counter()
    seen = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for text in _iter_docs(
                src["hf_id"], src.get("config"), src["split"], src["text_field"]
            ):
                if len(text) < min_chars:
                    continue
                seen += 1
                if seen <= skip_docs:
                    st.skipped = seen
                    if seen % 5000 == 0:
                        print(
                            f"\r   {src['key']}: skipping {seen:,}/{skip_docs:,}",
                            end="",
                            flush=True,
                        )
                    continue
                remain = char_budget - st.chars
                if remain <= 0:
                    break
                if len(text) > remain:
                    text = text[:remain].rsplit("\n", 1)[0] or text[:remain]
                    if len(text) < min_chars:
                        break
                f.write(text)
                f.write("\n\n")
                st.chars += len(text)
                st.docs += 1
                if st.docs % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = st.chars / elapsed if elapsed else 0
                    print(
                        f"\r   {src['key']}: docs={st.docs:,} chars={st.chars:,}/"
                        f"{char_budget:,} ({100 * st.chars / char_budget:5.1f}%) "
                        f"{rate:,.0f} ch/s skip_done={st.skipped:,}",
                        end="",
                        flush=True,
                    )
                if st.chars >= char_budget:
                    break
        print()
        st.status = "ok"
    except Exception as e:
        print()
        st.status = "error"
        st.error = f"{type(e).__name__}: {e}"
        print(f"   [错误] {src['key']}: {st.error}", file=sys.stderr, flush=True)
    st.seconds = round(time.perf_counter() - t0, 1)
    st.est_tokens = int(st.chars / 1.4)
    print(
        f"   done {src['key']}: {st.chars:,} chars / ~{st.est_tokens:,} tok / "
        f"{st.docs:,} docs / skipped={st.skipped:,} / {st.seconds}s [{st.status}]",
        flush=True,
    )
    return st


def _parse_kv_ints(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not s.strip():
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--skip-docs 需要 KEY=N 格式，收到: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip().replace("_", ""))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "seed_v2_zh_hq_raw")
    p.add_argument(
        "--sources",
        default="wiki_pretrain_zh,stories_zh",
        help="逗号分隔 key；默认 P0。加 skypile_zh / wiki_zh_wikimedia",
    )
    p.add_argument("--wiki-chars", type=int, default=120_000_000)
    p.add_argument("--stories-chars", type=int, default=60_000_000)
    p.add_argument("--skypile-chars", type=int, default=60_000_000)
    p.add_argument("--wikimedia-chars", type=int, default=80_000_000)
    p.add_argument(
        "--skip-docs",
        default="",
        help="按源跳过已拉 HF 样本，如 wiki_pretrain_zh=29479,skypile_zh=60527",
    )
    p.add_argument(
        "--outfile-suffix",
        default="",
        help="写到 outfile 名插入后缀，如 _more → wiki_pretrain_zh_more.txt",
    )
    p.add_argument("--smoke", action="store_true", help="冒烟：每源 200_000 字符")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    wanted = [x.strip() for x in args.sources.split(",") if x.strip()]
    skip_map = _parse_kv_ints(args.skip_docs)
    budgets = {
        "wiki_pretrain_zh": 200_000 if args.smoke else args.wiki_chars,
        "stories_zh": 200_000 if args.smoke else args.stories_chars,
        "skypile_zh": 200_000 if args.smoke else args.skypile_chars,
        "wiki_zh_wikimedia": 200_000 if args.smoke else args.wikimedia_chars,
    }

    print("=== download_hq_zh_corpus ===")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default hub)')}")
    print(f"out_dir={args.out_dir}")
    print(f"sources={wanted}")
    print(f"skip_docs={skip_map}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[Stats] = []
    t_all = time.perf_counter()
    by_key = {s["key"]: s for s in SOURCES}
    for key in wanted:
        if key not in by_key:
            print(f"[警告] 未知源 {key}，跳过", flush=True)
            continue
        src = dict(by_key[key])  # copy
        if args.outfile_suffix:
            stem = Path(src["outfile"]).stem
            src["outfile"] = f"{stem}{args.outfile_suffix}.txt"
        out_path = args.out_dir / src["outfile"]
        budget = budgets.get(key, src["default_char_budget"])
        skip_docs = skip_map.get(key, 0)
        if (
            not args.force
            and out_path.is_file()
            and out_path.stat().st_size >= int(budget * 0.9)
            and skip_docs == 0
        ):
            print(f"↷ skip {key}: 已有 {out_path.stat().st_size:,} bytes", flush=True)
            st = Stats(
                key=key,
                hf_id=src["hf_id"],
                outfile=src["outfile"],
                char_budget=budget,
                chars=out_path.stat().st_size,
                status="skipped_existing",
                est_tokens=int(out_path.stat().st_size / 2.5 / 1.4),
            )
            all_stats.append(st)
            continue
        st = download_one(src, out_path, budget, skip_docs=skip_docs)
        all_stats.append(st)

    elapsed = round(time.perf_counter() - t_all, 1)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": str(args.out_dir),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "skip_docs": skip_map,
        "outfile_suffix": args.outfile_suffix,
        "sources": [asdict(s) for s in all_stats],
        "elapsed_seconds": elapsed,
        "est_chars_per_token": 1.4,
        "next": (
            "python scripts/clean_zh_corpus.py "
            "--in-dir data/seed_v2_zh_hq_raw_v2 --out-dir data/seed_v2_zh_hq_new_v2"
        ),
    }
    meta_path = args.out_dir / "download_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 汇总 ===")
    total_chars = sum(s.chars for s in all_stats if s.status == "ok")
    total_tok = sum(s.est_tokens for s in all_stats if s.status == "ok")
    print(f"chars={total_chars:,} est_tok≈{total_tok:,} | {elapsed}s")
    print(f"meta: {meta_path}")
    errors = [s for s in all_stats if s.status == "error"]
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    import os as _os

    _os._exit(code)
