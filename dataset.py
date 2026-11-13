"""语料加载：字符级全量张量，或预编码 memmap token 文件。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch


def _torch() -> Any:
    """惰性导入，避免「仅检查语料」时因未装 torch 直接崩溃。"""
    try:
        import torch as _t
    except ImportError as e:
        raise ImportError(
            "未安装 PyTorch。请先: pip install -r requirements.txt"
        ) from e
    return _t


def resolve_token_dtype(path: str | Path, meta_path: str | Path | None = None) -> np.dtype:
    """从 tokens_meta.json 读 dtype；没有则默认 uint16（旧 BPE）。"""
    path = Path(path)
    candidates = []
    if meta_path is not None:
        candidates.append(Path(meta_path))
    candidates.append(path.parent / "tokens_meta.json")
    for meta in candidates:
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                name = data.get("dtype", "uint16")
                return np.dtype(name)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return np.dtype(np.uint16)


class CharDataset:
    """把整份语料编码成一长串 token，再随机切 block（适合小语料 / 字符版）。"""

    def __init__(self, data: "torch.Tensor", block_size: int):
        self.data = data
        self.block_size = block_size
        if len(data) < block_size + 1:
            raise ValueError(
                f"语料编码后长度 {len(data)} 太短，至少需要 block_size+1="
                f"{block_size + 1}。请换更长的语料（data/corpus/ 或 data/input.txt）。"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str) -> Tuple[Any, Any]:
        """随机采 batch：x 是输入，y 是右移一位的目标。"""
        torch = _torch()
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


class MemmapDataset:
    """从 memmap（train.bin / val.bin）随机切 block，不把全文载入 RAM。

    dtype 默认跟 tokens_meta.json（HF 大词表用 uint32；旧 BPE 用 uint16）。
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        *,
        dtype: np.dtype | str | None = None,
        meta_path: str | Path | None = None,
    ):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 token 文件: {path.resolve()}\n"
                "请先运行: python scripts/encode_corpus.py"
            )
        self.path = path
        self.block_size = block_size
        self.dtype = np.dtype(dtype) if dtype is not None else resolve_token_dtype(path, meta_path)
        self._mmap = np.memmap(path, dtype=self.dtype, mode="r")
        if len(self._mmap) < block_size + 1:
            raise ValueError(
                f"{path} 只有 {len(self._mmap)} 个 token，"
                f"至少需要 block_size+1={block_size + 1}。"
            )

    def __len__(self) -> int:
        return len(self._mmap) - self.block_size

    def get_batch(self, batch_size: int, device: str) -> Tuple[Any, Any]:
        torch = _torch()

        ix = torch.randint(len(self), (batch_size,))
        xs, ys = [], []
        for i in ix.tolist():
            window = np.array(self._mmap[i : i + self.block_size + 1], dtype=np.int64)
            xs.append(torch.from_numpy(window[:-1]))
            ys.append(torch.from_numpy(window[1:]))
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x.to(device), y.to(device)


def load_text_corpus(
    path: str | Path | None = None,
    *,
    corpus_dir: str | Path | None = None,
    corpus_glob: str | None = None,
    extra_files: list[str | Path] | None = None,
) -> str:
    """读取 UTF-8 语料（可多文件拼接）。

    - 仅传 path 且为普通文件：行为与旧版一致（字符版 / 小语料）。
    - path 为目录、或给了 corpus_dir / corpus_glob / extra_files：走 corpus.resolve。
    - 全部省略：自动选 data/corpus/*.txt 或 data/input.txt。
    大语料请用 scripts/encode_corpus.py，不要把整份拼进内存。
    """
    from corpus import read_corpus_text, resolve_corpus_files

    # 兼容：旧调用 load_text_corpus("data/input.txt")
    if (
        path is not None
        and corpus_dir is None
        and not corpus_glob
        and not extra_files
        and Path(path).is_file()
    ):
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"语料文件为空: {p}。请写入非空文本后再训练。")
        return text

    corpus_arg = None
    dir_arg = Path(corpus_dir) if corpus_dir is not None else None
    if path is not None:
        p = Path(path)
        if p.is_dir():
            dir_arg = p if dir_arg is None else dir_arg
        elif p.is_file():
            corpus_arg = p
        else:
            raise FileNotFoundError(
                f"找不到语料: {p.resolve()}\n"
                "请放入 data/corpus/*.txt 或 data/input.txt。详见 data/README.md。"
            )

    try:
        files = resolve_corpus_files(
            corpus=corpus_arg,
            corpus_dir=dir_arg,
            corpus_glob=corpus_glob,
            extra_files=[Path(x) for x in extra_files] if extra_files else None,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\n详见 data/README.md。") from e
    return read_corpus_text(files)


def build_datasets(
    text: str,
    tokenizer,
    block_size: int,
    train_ratio: float = 0.9,
) -> Tuple[CharDataset, CharDataset]:
    """编码全文并按比例切成 train / val（字符版 / 小语料）。"""
    torch = _torch()
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_ratio * len(ids))
    n = max(n, min(len(ids) - 1, block_size + 1))
    train_ids = ids[:n]
    val_ids = ids[n:] if n < len(ids) else ids[-min(len(ids), block_size + 2) :]
    return CharDataset(train_ids, block_size), CharDataset(val_ids, block_size)


def load_memmap_datasets(
    train_bin: str | Path,
    val_bin: str | Path,
    block_size: int,
    *,
    dtype: np.dtype | str | None = None,
    meta_path: str | Path | None = None,
) -> Tuple[MemmapDataset, MemmapDataset]:
    """加载预先 encode 好的 train.bin / val.bin。"""
    return (
        MemmapDataset(train_bin, block_size, dtype=dtype, meta_path=meta_path),
        MemmapDataset(val_bin, block_size, dtype=dtype, meta_path=meta_path),
    )
