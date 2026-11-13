#!/usr/bin/env python3
"""阶段 C 同款中文语料清洗：过短 / 赌球 / 广告 / URL / 乱码 / 外文混排 / 碎片。

把「双换行分段」的 txt 读入，过滤后写出。不碰正在训练用的 bin。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAMBLING_PAT = re.compile(
    r"(赌球|赌博|赌场|博彩|六合彩|时时彩|快三|赛车投注|百家乐|德州扑克|"
    r"威尼斯人|太阳城|澳门赌场|皇冠现金|bet365|体育投注|线上赌场|在线博彩|"
    r"彩票平台|开奖结果|赔率|盘口|注单|娱乐城|电子游艺|老虎机|"
    r"真人荷官|AG真人|BBIN|开元棋牌|炸金花|牛牛游戏|澳门新葡京)",
    re.I,
)
ADS_PAT = re.compile(
    r"(点击领取|免费试用|加微信|加QQ|代开发票|办证|贷款秒下|信用卡套现|"
    r"刷单|兼职日结|私聊我|扫码关注|限时特价|马上注册|立即下载|"
    r"官网入口|优惠活动|代理加盟|招商加盟|微商代理)",
    re.I,
)
URL_PAT = re.compile(r"https?://|www\.[a-z0-9\-]+\.[a-z]{2,}", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
# 常见乱码 / 替换符
GARBLED_PAT = re.compile(r"(?:�|\ufeff|ï»¿)")
# 无标点、过碎的「词语堆」
MASH_PAT = re.compile(r"^[\u4e00-\u9fff\s\|/·•,]{0,200}$")


def normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
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


def classify(text: str, *, min_chars: int, min_cjk: int) -> str:
    n = len(text)
    if n < min_chars:
        return "too_short"
    # 单遍计数，避免多次 findall
    cjk = latin = digits = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        elif (65 <= o <= 90) or (97 <= o <= 122):
            latin += 1
        elif 48 <= o <= 57:
            digits += 1
    if cjk < min_cjk:
        return "low_cjk"
    if latin / n > 0.45 and cjk / n < 0.35:
        return "latin_heavy"
    if digits / n > 0.35:
        return "digit_heavy"
    urls = len(URL_PAT.findall(text))
    if urls >= 4 or (urls >= 2 and urls / max(cjk, 1) > 0.02):
        return "url_heavy"
    if GAMBLING_PAT.search(text):
        return "gambling"
    if ADS_PAT.search(text) and (urls >= 1 or "微信" in text or "QQ" in text):
        return "ads"
    if ADS_PAT.search(text) and n < 400:
        return "ads"
    repl = text.count("\ufffd") + text.count("\ufeff")
    if repl >= 3:
        return "garbled"
    if GARBLED_PAT.search(text) and cjk / n < 0.5:
        return "garbled"
    # 碎片仅对短文本检查，避免长文正则回溯
    if n < 240:
        punct = sum(text.count(p) for p in "。！？；，、,.!?;")
        if punct == 0 and cjk < 60:
            return "mash_fragment"
        flat = text.replace("\n", " ")
        if MASH_PAT.match(flat):
            return "mash_fragment"
    return "keep"


def iter_docs(path: Path):
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "" and buf:
                doc = normalize("".join(buf))
                buf = []
                if doc:
                    yield doc
            else:
                buf.append(line)
        if buf:
            doc = normalize("".join(buf))
            if doc:
                yield doc


def clean_file(
    in_path: Path,
    out_path: Path,
    *,
    min_chars: int,
    min_cjk: int,
    progress_every: int = 2000,
) -> dict:
    drop = Counter()
    out_docs = 0
    out_chars = 0
    in_docs = 0
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for doc in iter_docs(in_path):
            in_docs += 1
            tag = classify(doc, min_chars=min_chars, min_cjk=min_cjk)
            if tag != "keep":
                drop[tag] += 1
                drop["dropped_docs"] += 1
                continue
            out.write(doc)
            out.write("\n\n")
            out_docs += 1
            out_chars += len(doc)
            if in_docs % progress_every == 0:
                print(
                    f"\r   {in_path.name}: in={in_docs:,} keep={out_docs:,} "
                    f"drop={drop['dropped_docs']:,}",
                    end="",
                    flush=True,
                )
    print()
    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "in_docs": in_docs,
        "out_docs": out_docs,
        "out_chars": out_chars,
        "out_bytes": out_path.stat().st_size if out_path.exists() else 0,
        "drop": dict(drop),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Clean Chinese corpus with phase-C rules")
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--min-cjk", type=int, default=40)
    p.add_argument("--meta", type=Path, default=None)
    args = p.parse_args()

    files = sorted(args.in_dir.glob(args.glob))
    if not files:
        print(f"[错误] 无文件: {args.in_dir}/{args.glob}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"=== clean_zh_corpus ===")
    print(f"in={args.in_dir} out={args.out_dir} files={len(files)}")
    for fp in files:
        out_fp = args.out_dir / fp.name
        print(f"→ {fp.name}", flush=True)
        results[fp.name] = clean_file(
            fp, out_fp, min_chars=args.min_chars, min_cjk=args.min_cjk
        )
        r = results[fp.name]
        print(
            f"   keep {r['out_docs']:,}/{r['in_docs']:,} docs | "
            f"{r['out_chars']:,} chars | {r['out_bytes']:,} bytes | {r['seconds']}s",
            flush=True,
        )

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(args.in_dir),
        "out_dir": str(args.out_dir),
        "rules": {
            "min_chars": args.min_chars,
            "min_cjk": args.min_cjk,
            "drop": [
                "gambling",
                "ads",
                "latin_heavy",
                "digit_heavy",
                "url_heavy",
                "garbled",
                "low_cjk",
                "mash_fragment",
                "too_short",
            ],
        },
        "files": results,
        "note": "阶段 C 同款规则；未覆盖训练用 clean bin。",
    }
    meta_path = args.meta or (args.out_dir / "clean_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
