"""语料路径解析：支持单文件、目录、glob，默认兼容 data/input.txt。

学习要点：训练脚本不应写死「只有一个 input.txt」；把「有哪些源文件」
集中在这里，encode / 训词表 / 字符版读文本都走同一套规则。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "input.txt"
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"


def _sorted_txt_in_dir(directory: Path) -> list[Path]:
    """目录及子目录下所有 .txt（不含隐藏名），按相对路径排序，保证可复现。"""
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.rglob("*.txt")
        if p.is_file()
        and not p.name.startswith(".")
        and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    ]
    return sorted(files, key=lambda p: p.relative_to(directory).as_posix())


def _expand_glob(pattern: str, base: Path | None = None) -> list[Path]:
    """相对路径相对项目根做 glob；绝对路径按其父目录 glob。"""
    base = base or ROOT
    raw = Path(pattern)
    if raw.is_absolute():
        matches = sorted(raw.parent.glob(raw.name))
    else:
        matches = sorted(base.glob(pattern))
    return [
        p.resolve()
        for p in matches
        if p.is_file() and p.suffix.lower() == ".txt"
    ]


def resolve_corpus_files(
    *,
    corpus: Path | None = None,
    corpus_dir: Path | None = None,
    corpus_glob: str | None = None,
    extra_files: Sequence[Path] | None = None,
    default_input: Path = DEFAULT_INPUT,
    default_corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> list[Path]:
    """解析出最终要读的 .txt 列表（已 resolve、去重、保序）。

    显式参数（任一给出）时，按「corpus → corpus_dir → glob → extra」拼接。

    未给任何显式参数时（向后兼容）：
      1. 若 data/corpus/ 存在且含 .txt（含子目录）→ 用该目录（推荐多文件）
      2. 否则若 data/input.txt 存在 → 仅用该文件
      3. 否则报错
    """
    explicit = any(
        [
            corpus is not None,
            corpus_dir is not None,
            bool(corpus_glob),
            bool(extra_files),
        ]
    )

    collected: list[Path] = []

    if explicit:
        if corpus is not None:
            p = Path(corpus)
            if not p.is_file():
                raise FileNotFoundError(f"语料文件不存在: {p.resolve()}")
            collected.append(p.resolve())
        if corpus_dir is not None:
            d = Path(corpus_dir)
            if not d.is_dir():
                raise FileNotFoundError(f"语料目录不存在: {d.resolve()}")
            found = _sorted_txt_in_dir(d)
            if not found:
                raise FileNotFoundError(f"语料目录下没有 .txt: {d.resolve()}")
            collected.extend(p.resolve() for p in found)
        if corpus_glob:
            found = _expand_glob(corpus_glob)
            if not found:
                raise FileNotFoundError(f"glob 未匹配到 .txt: {corpus_glob}")
            collected.extend(found)
        if extra_files:
            for f in extra_files:
                p = Path(f)
                if not p.is_file():
                    raise FileNotFoundError(f"语料文件不存在: {p.resolve()}")
                collected.append(p.resolve())
    else:
        dir_files = _sorted_txt_in_dir(default_corpus_dir)
        if dir_files:
            collected = [p.resolve() for p in dir_files]
        elif default_input.is_file():
            collected = [default_input.resolve()]
        else:
            raise FileNotFoundError(
                "未找到语料。请任选其一：\n"
                f"  - 放入多个 UTF-8 文本：{default_corpus_dir}/（可分子目录）某书.txt\n"
                f"  - 或沿用单文件：{default_input}\n"
                "  - 或传 --corpus / --corpus-dir / --corpus-glob"
            )

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        raise FileNotFoundError("语料列表为空。")
    return unique


def add_corpus_cli_args(parser) -> None:
    """给 ArgumentParser 挂上统一的语料相关参数（encode / train_bpe 共用）。"""
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="单个语料 .txt（兼容旧用法）",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="语料目录：递归读取其中全部 .txt（含子目录，按相对路径排序）。推荐 data/corpus/",
    )
    parser.add_argument(
        "--corpus-glob",
        type=str,
        default=None,
        help='相对项目根的 glob，例如 "data/raw_epub/monogatari_*.txt"',
    )
    parser.add_argument(
        "corpus_files",
        nargs="*",
        type=Path,
        help="额外的语料文件（可多个，写在命令末尾）",
    )


def resolve_from_args(args) -> list[Path]:
    """从 argparse Namespace 解析语料列表。"""
    extra = list(getattr(args, "corpus_files", None) or [])
    return resolve_corpus_files(
        corpus=getattr(args, "corpus", None),
        corpus_dir=getattr(args, "corpus_dir", None),
        corpus_glob=getattr(args, "corpus_glob", None),
        extra_files=extra or None,
    )


def format_corpus_summary(files: Iterable[Path], *, max_list: int = 40) -> str:
    """汇总语料列表；文件很多时只打印头尾，避免刷屏。"""
    files = list(files)
    total = sum(p.stat().st_size for p in files)
    lines = [
        f"共 {len(files)} 个语料文件，合计 {total:,} bytes"
        f" ({total / (1024 ** 3):.2f} GiB)"
    ]

    def _rel(p: Path) -> Path | str:
        try:
            return p.relative_to(ROOT)
        except ValueError:
            return p

    if len(files) <= max_list:
        for i, p in enumerate(files, 1):
            lines.append(f"  [{i:03d}] {_rel(p)} ({p.stat().st_size:,} bytes)")
    else:
        head = max_list // 2
        tail = max_list - head
        for i, p in enumerate(files[:head], 1):
            lines.append(f"  [{i:03d}] {_rel(p)} ({p.stat().st_size:,} bytes)")
        omitted = len(files) - head - tail
        lines.append(f"  ... 省略 {omitted} 个文件 ...")
        for i, p in enumerate(files[-tail:], len(files) - tail + 1):
            lines.append(f"  [{i:03d}] {_rel(p)} ({p.stat().st_size:,} bytes)")
    return "\n".join(lines)


def read_corpus_text(files: Sequence[Path], joiner: str = "\n\n") -> str:
    """把多个小文件拼成一个大字符串（仅适合字符版 / 小语料；大语料请走 encode 流式）。"""
    parts: list[str] = []
    for p in files:
        text = Path(p).read_text(encoding="utf-8")
        if text.strip():
            parts.append(text)
    if not parts:
        raise ValueError("所有语料文件都为空。")
    return joiner.join(parts)
