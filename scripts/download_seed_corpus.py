#!/usr/bin/env python3
"""流式拉取种子语料：维基（中日英）+ FineWeb 系网页截断，写入 data/seed/*.txt。

默认预算约 60M Qwen token（中日英大致均分）；--smoke 只拉约 1M 做冒烟。
不启动训练。下载完后用 encode_corpus.py --corpus-dir data/seed 编码。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TrainConfig  # noqa: E402
from tokenizer import HFTokenizer  # noqa: E402

# --- 数据源定义（均为 streaming 友好）---
# 网页：英用 fineweb-edu；中用 fineweb-2；日用日语 edu 子集（比 raw jpn_Jpan 更干净）。
SOURCES = [
    {
        "key": "wiki_zh",
        "lang": "zh",
        "kind": "wiki",
        "outfile": "wiki_zh.txt",
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "text_field": "text",
    },
    {
        "key": "wiki_ja",
        "lang": "ja",
        "kind": "wiki",
        "outfile": "wiki_ja.txt",
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.ja",
        "split": "train",
        "text_field": "text",
    },
    {
        "key": "wiki_en",
        "lang": "en",
        "kind": "wiki",
        "outfile": "wiki_en.txt",
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
    },
    {
        "key": "web_zh",
        "lang": "zh",
        "kind": "web",
        "outfile": "web_zh.txt",
        "hf_id": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
        "split": "train",
        "text_field": "text",
    },
    {
        "key": "web_ja",
        "lang": "ja",
        "kind": "web",
        "outfile": "web_ja.txt",
        "hf_id": "hotchpotch/fineweb-2-edu-japanese",
        "config": "sample_10BT",
        "split": "train",
        "text_field": "text",
    },
    {
        "key": "web_en",
        "lang": "en",
        "kind": "web",
        "outfile": "web_en.txt",
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
    },
]


@dataclass
class SourceStats:
    key: str
    lang: str
    kind: str
    outfile: str
    hf_id: str
    config: str
    target_tokens: int
    tokens: int = 0
    chars: int = 0
    docs: int = 0
    seconds: float = 0.0
    status: str = "pending"
    error: str = ""


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").strip()
    if not text:
        return ""
    # 压缩过多空行，避免 wiki 模板噪音占满预算
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


def _count_tokens(tok: HFTokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tok.encode(text))


def _iter_docs(hf_id: str, config: str, split: str, text_field: str) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, config, split=split, streaming=True)
    for ex in ds:
        raw = ex.get(text_field) or ""
        if not isinstance(raw, str):
            continue
        text = _clean_text(raw)
        if len(text) < 80:
            continue
        yield text


def download_one(
    src: dict,
    out_path: Path,
    target_tokens: int,
    tok: HFTokenizer,
    *,
    append: bool = False,
    progress_every: int = 200,
) -> SourceStats:
    stats = SourceStats(
        key=src["key"],
        lang=src["lang"],
        kind=src["kind"],
        outfile=src["outfile"],
        hf_id=src["hf_id"],
        config=src["config"],
        target_tokens=target_tokens,
    )
    if target_tokens <= 0:
        stats.status = "skipped"
        return stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out_path.exists() else "w"
    if mode == "w" and out_path.exists():
        out_path.unlink()

    print(
        f"→ {src['key']}: {src['hf_id']} ({src['config']}) "
        f"target≈{target_tokens:,} tok → {out_path.name}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        with out_path.open(mode, encoding="utf-8") as f:
            for text in _iter_docs(
                src["hf_id"], src["config"], src["split"], src["text_field"]
            ):
                n = _count_tokens(tok, text)
                if n <= 0:
                    continue
                # 超出预算时截断本篇（尽量写完整句前缀）
                remain = target_tokens - stats.tokens
                if remain <= 0:
                    break
                if n > remain:
                    # 粗按字符比例截；再精确数 token
                    ratio = max(remain / n, 0.05)
                    cut = max(200, int(len(text) * ratio))
                    text = text[:cut].rsplit("\n", 1)[0] or text[:cut]
                    n = _count_tokens(tok, text)
                    if n <= 0:
                        break
                    if n > remain:
                        # 再狠砍一次
                        text = text[: max(100, int(len(text) * remain / max(n, 1)))]
                        n = _count_tokens(tok, text)
                        if n <= 0 or n > remain * 1.2:
                            break
                f.write(text)
                f.write("\n\n")
                stats.tokens += n
                stats.chars += len(text)
                stats.docs += 1
                if stats.docs % progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = stats.tokens / elapsed if elapsed > 0 else 0
                    print(
                        f"\r   {src['key']}: docs={stats.docs:,} "
                        f"tok={stats.tokens:,}/{target_tokens:,} "
                        f"({100.0 * stats.tokens / target_tokens:5.1f}%) "
                        f"{rate:,.0f} tok/s",
                        end="",
                        flush=True,
                    )
                if stats.tokens >= target_tokens:
                    break
        print()
        stats.status = "ok"
    except Exception as e:
        print()
        stats.status = "error"
        stats.error = f"{type(e).__name__}: {e}"
        print(f"   [错误] {src['key']}: {stats.error}", file=sys.stderr, flush=True)
    stats.seconds = round(time.perf_counter() - t0, 1)
    print(
        f"   done {src['key']}: {stats.tokens:,} tok / {stats.chars:,} chars / "
        f"{stats.docs:,} docs / {stats.seconds}s [{stats.status}]",
        flush=True,
    )
    return stats


def allocate_budgets(
    total: int,
    langs: list[str],
    wiki_ratio: float,
    kinds: set[str],
) -> dict[str, int]:
    """按语言均分，语言内再按 wiki_ratio 分给 wiki / web。"""
    langs = [x for x in langs if x in ("zh", "ja", "en")]
    if not langs:
        raise ValueError("至少指定一种语言：zh / ja / en")
    wiki_ratio = min(max(wiki_ratio, 0.0), 1.0)
    per_lang = total // len(langs)
    rem = total - per_lang * len(langs)
    out: dict[str, int] = {}
    for i, lang in enumerate(langs):
        budget = per_lang + (1 if i < rem else 0)
        has_wiki = "wiki" in kinds
        has_web = "web" in kinds
        if has_wiki and has_web:
            wiki_budget = int(budget * wiki_ratio)
            web_budget = budget - wiki_budget
        elif has_wiki:
            wiki_budget, web_budget = budget, 0
        elif has_web:
            wiki_budget, web_budget = 0, budget
        else:
            wiki_budget, web_budget = 0, 0
        out[f"wiki_{lang}"] = wiki_budget
        out[f"web_{lang}"] = web_budget
    return out


def main() -> int:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(
        description="Stream-download seed corpus (Wikipedia + FineWeb-style) into data/seed/"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "seed",
        help="输出目录（默认 data/seed）",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=60_000_000,
        help="总 token 预算（Qwen 词表计数，默认 60M）",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟：总预算改为 1_000_000（约 33万/语种）",
    )
    p.add_argument(
        "--wiki-ratio",
        type=float,
        default=0.30,
        help="每种语言里维基占比（默认 0.30，其余给网页）",
    )
    p.add_argument(
        "--langs",
        default="zh,ja,en",
        help="语言列表，逗号分隔（默认 zh,ja,en）",
    )
    p.add_argument(
        "--wiki-only",
        action="store_true",
        help="只下维基（把预算全给 wiki）",
    )
    p.add_argument(
        "--web-only",
        action="store_true",
        help="只下网页 FineWeb 系",
    )
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help=f"HF tokenizer id（默认 {cfg.hf_tokenizer_id}）",
    )
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help=f"HF 缓存目录（默认 {cfg.hf_cache_dir}）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有输出文件（默认跳过已达预算的文件）",
    )
    args = p.parse_args()

    if args.wiki_only and args.web_only:
        print("[错误] --wiki-only 与 --web-only 不能同时用", file=sys.stderr)
        return 1

    total = 1_000_000 if args.smoke else args.budget
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    kinds: set[str] = {"wiki", "web"}
    if args.wiki_only:
        kinds = {"wiki"}
    if args.web_only:
        kinds = {"web"}

    budgets = allocate_budgets(total, langs, args.wiki_ratio, kinds)
    print("=== 种子语料下载 ===")
    print(f"out_dir: {args.out_dir}")
    print(f"total budget: {total:,} tokens (Qwen) | langs={langs} | kinds={sorted(kinds)}")
    print(f"per-source targets: { {k: v for k, v in budgets.items() if v} }")

    cache = args.hf_cache_dir or (ROOT / cfg.hf_cache_dir)
    hf_id = args.hf_tokenizer or cfg.hf_tokenizer_id
    print(f"加载 tokenizer: {hf_id} …", flush=True)
    tok = HFTokenizer.load(hf_id, cache_dir=cache)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[SourceStats] = []
    t_all = time.perf_counter()

    for src in SOURCES:
        if src["lang"] not in langs:
            continue
        if src["kind"] not in kinds:
            continue
        target = budgets.get(src["key"], 0)
        out_path = args.out_dir / src["outfile"]
        if (
            not args.force
            and out_path.is_file()
            and out_path.stat().st_size > 0
            and target > 0
        ):
            # 粗估已有 token（按字符/语种经验系数），够了就跳过
            existing_chars = out_path.stat().st_size
            # 快速抽检：读最多 200KB 估 ratio
            sample = out_path.read_text(encoding="utf-8", errors="ignore")[:200_000]
            if sample:
                s_tok = _count_tokens(tok, sample)
                ratio = s_tok / max(len(sample), 1)
                est = int(existing_chars * ratio)
            else:
                est = 0
            if est >= int(target * 0.95):
                print(
                    f"↷ skip {src['key']}: 已有约 {est:,} tok ≥ 目标 {target:,}",
                    flush=True,
                )
                st = SourceStats(
                    key=src["key"],
                    lang=src["lang"],
                    kind=src["kind"],
                    outfile=src["outfile"],
                    hf_id=src["hf_id"],
                    config=src["config"],
                    target_tokens=target,
                    tokens=est,
                    chars=existing_chars,
                    status="skipped_existing",
                )
                all_stats.append(st)
                continue

        st = download_one(src, out_path, target, tok)
        all_stats.append(st)

    elapsed = round(time.perf_counter() - t_all, 1)
    total_tok = sum(s.tokens for s in all_stats)
    total_chars = sum(s.chars for s in all_stats)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_budget": total,
        "wiki_ratio": args.wiki_ratio,
        "langs": langs,
        "kinds": sorted(kinds),
        "smoke": bool(args.smoke),
        "tokenizer": hf_id,
        "total_tokens_approx": total_tok,
        "total_chars": total_chars,
        "elapsed_seconds": elapsed,
        "sources": [asdict(s) for s in all_stats],
        "encode_command": (
            "python scripts/encode_corpus.py --corpus-dir data/seed "
            "--train-bin data/seed_train.bin --val-bin data/seed_val.bin"
        ),
        "note": (
            "token 数为下载时用 Qwen 词表统计的近似值；"
            "正式训练以 encode_corpus 生成的 tokens_meta.json 为准。"
            "本地小说请继续放 data/corpus/，种子与小说分开；"
            "混训时可用 --corpus-dir data/seed 再额外加 --corpus-glob。"
        ),
    }
    meta_path = args.out_dir / "seed_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 汇总 ===")
    print(f"tokens≈{total_tok:,} | chars={total_chars:,} | {elapsed}s")
    print(f"meta: {meta_path}")
    print("下一步编码（不覆盖旧小说 bin，写入 seed_*）：")
    print(f"  {meta['encode_command']}")
    errors = [s for s in all_stats if s.status == "error"]
    if errors:
        print(f"[警告] {len(errors)} 个源失败：{[s.key for s in errors]}")
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    # datasets streaming 常留下后台线程，导致进程挂起不退出
    import os

    os._exit(code)
