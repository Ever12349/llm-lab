#!/usr/bin/env python3
"""Extract plain Chinese text from EPUB novels (spine order).

Pipeline:
  container.xml -> OPF spine -> HTML body text -> strip tags ->
  drop front/back matter chapters -> normalize whitespace ->
  traditional -> simplified (zhconv zh-cn).

Does NOT merge into data/input.txt.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

try:
    from zhconv import convert as zh_convert
except ImportError:  # pragma: no cover
    print("zhconv required: pip install zhconv", file=sys.stderr)
    raise

# Spine item filenames / stems that are almost always non-body.
SKIP_NAME_RE = re.compile(
    r"(?i)("
    r"cover|titlepage|title|logo|message|contents|toc|nav|ncx|"
    r"summary|colophon|copyright|backcover|back.?cover|"
    r"afterword|preface|foreword|appendix|"
    r"作者简介|作者介绍|制作信息|版权|目录"
    r")"
)

# Titles that mark front/back matter chapters (matched after HTML title extract).
SKIP_TITLE_RE = re.compile(
    r"("
    r"目录|目次|制作信息|录入信息|錄入信息|简介|内容简介|内容提要|"
    r"封面|前页|版权|版权页|出版|献辞|献词|"
    r"序言|序章|前言|序$|"
    r"后记|後記|译后记|譯後記|编者后记|编后记|"
    r"附录|附錄|作者介绍|作者简介|译者介绍|插画|插畫|"
    r"人物介绍|登场人物|登場人物|卷末|广告"
    r")"
)

# Trailing whole-text cut only when a late heading clearly starts notes
# (safety net; prefer skipping by spine title when possible).
TRAILING_CUT_RE = re.compile(
    r"(?m)^\s*(后记|後記|译后记|譯後記|附录|附錄|作者介绍|作者简介)\s*$"
)

MIN_CHAPTER_CHARS = 40  # skip image-only / near-empty spine items


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in {"script", "style"}:
            self._skip_depth += 1
            return
        if t == "title":
            self._in_title = True
        if t in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if t == "title":
            self._in_title = False
        if t in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_xml(data: bytes) -> ET.Element:
    # Drop xmlns decls and element/attr prefixes so plain tag names work.
    cleaned = re.sub(rb'\sxmlns(:\w+)?="[^"]*"', b"", data)
    cleaned = re.sub(rb'(</?)([A-Za-z_][\w.-]*):', rb'\1', cleaned)
    cleaned = re.sub(rb'([\s])([A-Za-z_][\w.-]*):([A-Za-z_][\w.-]*=)', rb'\1\3', cleaned)
    return ET.fromstring(cleaned)


def find_opf_path(zf: zipfile.ZipFile) -> str:
    with zf.open("META-INF/container.xml") as f:
        root = _parse_xml(f.read())
    for node in root.iter():
        if _local(node.tag) == "rootfile":
            path = node.attrib.get("full-path")
            if path:
                return path
    raise FileNotFoundError("No rootfile in META-INF/container.xml")


def spine_html_paths(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    with zf.open(opf_path) as f:
        opf = _parse_xml(f.read())
    manifest: dict[str, str] = {}
    for node in opf.iter():
        if _local(node.tag) == "item":
            iid = node.attrib.get("id")
            href = node.attrib.get("href")
            if iid and href:
                manifest[iid] = href
    opf_dir = posixpath.dirname(opf_path)
    paths: list[str] = []
    for node in opf.iter():
        if _local(node.tag) != "itemref":
            continue
        idref = node.attrib.get("idref")
        if not idref or idref not in manifest:
            continue
        href = manifest[idref]
        media = ""
        # Prefer xhtml/html only.
        full = posixpath.normpath(posixpath.join(opf_dir, href) if opf_dir else href)
        if not re.search(r"\.(x?html?|htm)$", full, re.I):
            continue
        paths.append(full)
    return paths


def html_to_text(raw: bytes) -> tuple[str, str]:
    # Decode with fallbacks common in CN light-novel EPUBs.
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            s = raw.decode(enc)
            break
        except UnicodeDecodeError:
            s = None
    else:
        s = raw.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    try:
        parser.feed(s)
        parser.close()
    except Exception:
        # Extremely broken markup: crude strip.
        title_m = re.search(r"<title[^>]*>(.*?)</title>", s, re.I | re.S)
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
        body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", s)
        body = re.sub(r"<[^>]+>", "\n", body)
        return title, body
    return parser.title.strip(), parser.text()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", "　")
    # Collapse spaces within lines but keep CJK spacing feel.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        lines.append(line)
    # Collapse 3+ blank lines to a single blank line (keep paragraph breaks).
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue
        blank_run = 0
        out.append(line)
    return "\n".join(out).strip() + "\n"


def should_skip_chapter(href: str, title: str, body: str) -> str | None:
    """Return skip reason or None to keep."""
    name = posixpath.basename(href)
    stem = Path(name).stem
    if SKIP_NAME_RE.search(stem) or SKIP_NAME_RE.search(name):
        # Section0xx numbered chapters must NOT be skipped by name alone.
        if not re.match(r"(?i)section\d+", stem):
            return f"name:{stem}"
    # Explicit front/back names even if called Section…
    if title and SKIP_TITLE_RE.search(title.strip()):
        # Avoid skipping a main chapter whose title merely contains 序 as a word
        # in a longer phrase we already anchored; SKIP_TITLE_RE uses 序$ carefully.
        return f"title:{title.strip()[:40]}"
    cleaned = body.strip()
    if len(cleaned) < MIN_CHAPTER_CHARS:
        return f"too_short:{len(cleaned)}"
    # Short production / staff pages with generic titles (chapterN.html etc.).
    head = cleaned[:200]
    if len(cleaned) < 800 and (
        "制作信息" in head
        or "录入信息" in head
        or "錄入信息" in head
        or ("轻之国度" in cleaned and "仅供" in cleaned)
        or ("輕之國度" in cleaned and "僅供" in cleaned)
    ):
        return "content:production_page"
    return None


def cut_trailing_matter(full: str) -> str:
    """Only cut from a late standalone heading (后记/附录/…); never mid-book ※."""
    m = TRAILING_CUT_RE.search(full)
    if not m:
        return full
    # Only cut if the match is in the last 15% of the book (trailing).
    if m.start() < int(len(full) * 0.85):
        return full
    return full[: m.start()].rstrip() + "\n"


def extract_epub(epub_path: Path) -> tuple[str, list[dict]]:
    """Return (simplified_text, chapter_log)."""
    log: list[dict] = []
    chapters: list[str] = []
    with zipfile.ZipFile(epub_path, "r") as zf:
        opf = find_opf_path(zf)
        hrefs = spine_html_paths(zf, opf)
        for href in hrefs:
            try:
                raw = zf.read(href)
            except KeyError:
                # Some EPUBs use different path casing.
                alt = None
                for n in zf.namelist():
                    if n.lower() == href.lower():
                        alt = n
                        break
                if alt is None:
                    log.append({"href": href, "status": "missing"})
                    continue
                raw = zf.read(alt)
                href = alt
            title, body = html_to_text(raw)
            body = normalize_whitespace(body)
            reason = should_skip_chapter(href, title, body)
            if reason:
                log.append(
                    {
                        "href": href,
                        "title": title,
                        "status": "skipped",
                        "reason": reason,
                        "chars": len(body.strip()),
                    }
                )
                continue
            chapters.append(body.strip())
            log.append(
                {
                    "href": href,
                    "title": title,
                    "status": "kept",
                    "chars": len(body.strip()),
                }
            )
    joined = "\n\n".join(chapters)
    joined = cut_trailing_matter(joined)
    joined = normalize_whitespace(joined)
    simplified = zh_convert(joined, "zh-cn")
    if not simplified.endswith("\n"):
        simplified += "\n"
    return simplified, log


DEFAULT_BOOKS = [
    ("monogatari_01.txt", "01.化物语（上）.epub"),
    ("monogatari_02.txt", "02.化物语（下）.epub"),
    ("monogatari_03.txt", "03.伤物语.epub"),
    ("monogatari_04.txt", "04.伪物语（上）.epub"),
    ("monogatari_05.txt", "05.伪物语（下）.epub"),
    ("monogatari_06.txt", "06.猫物语（黑）.epub"),
    ("monogatari_07.txt", "07.猫物语（白）.epub"),
    ("monogatari_08.txt", "08.倾物语.epub"),
    ("monogatari_09.txt", "09.花物语.epub"),
    ("monogatari_10.txt", "10.囮物语.epub"),
]


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="EPUB 源目录（需本机自行指定，不写死路径）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "raw_epub",
    )
    ap.add_argument("--limit", type=int, default=10, help="Max books to process")
    args = ap.parse_args(list(argv) if argv is not None else None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total = 0

    for out_name, src_name in DEFAULT_BOOKS[: args.limit]:
        src = args.source_dir / src_name
        out = args.out_dir / out_name
        entry: dict = {
            "title": Path(src_name).stem,
            "source": str(src),
            "output": str(out),
        }
        if not src.is_file():
            entry["status"] = "error"
            entry["error"] = "source missing"
            entry["char_count"] = 0
            manifest.append(entry)
            print(f"MISSING {src}", file=sys.stderr)
            continue
        try:
            text, log = extract_epub(src)
            out.write_text(text, encoding="utf-8")
            n = len(text)
            total += n
            entry["status"] = "ok"
            entry["char_count"] = n
            entry["chapters_kept"] = sum(1 for c in log if c.get("status") == "kept")
            entry["chapters_skipped"] = sum(
                1 for c in log if c.get("status") == "skipped"
            )
            # Keep a short chapter log for debugging order/skips.
            entry["chapter_log"] = log
            print(
                f"OK {src_name} -> {out_name}  chars={n}  "
                f"kept={entry['chapters_kept']} skipped={entry['chapters_skipped']}"
            )
        except Exception as e:  # noqa: BLE001 — continue batch
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["char_count"] = 0
            print(f"FAIL {src_name}: {e}", file=sys.stderr)
        manifest.append(entry)

    summary = {
        "books": manifest,
        "total_chars": total,
        "ok_count": sum(1 for m in manifest if m.get("status") == "ok"),
        "cleaning": {
            "spine_order": True,
            "skip_front_back": True,
            "skip_image_only": True,
            "trailing_afterword_cut": True,
            "mid_book_notes_kept": True,
            "zhconv": "zh-cn",
            "whitespace": "collapse 3+ blank lines to 1; keep paragraphs",
        },
    }
    man_path = args.out_dir / "manifest.json"
    man_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {man_path}")
    print(f"Total chars: {total}  ok={summary['ok_count']}/{len(manifest)}")
    return 0 if summary["ok_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
