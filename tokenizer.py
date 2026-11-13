"""Tokenizer：字符级 / 自训 BPE / Hugging Face AutoTokenizer。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Union


class Tokenizer(Protocol):
    """训练 / 生成共用的最小接口。"""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class CharTokenizer:
    """按「出现过的字符」建 vocab，简单直观，适合教学对照。"""

    def __init__(self, stoi: Dict[str, int] | None = None, itos: Dict[int, str] | None = None):
        self.stoi: Dict[str, int] = dict(stoi or {})
        self.itos: Dict[int, str] = {int(k): v for k, v in (itos or {}).items()}
        if self.stoi and not self.itos:
            self.itos = {i: ch for ch, i in self.stoi.items()}
        if self.itos and not self.stoi:
            self.stoi = {ch: i for i, ch in self.itos.items()}

    @classmethod
    def build_from_text(cls, text: str) -> "CharTokenizer":
        """从整段文本提取去重字符，按排序得到稳定词表。"""
        chars = sorted(set(text))
        stoi = {ch: i for ch, i in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            uniq = "".join(sorted(set(unknown)))
            raise KeyError(f"文本含词表外字符: {uniq!r}。请用同一语料重建 tokenizer。")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = {int(k): v for k, v in data["itos"].items()}
        return cls(stoi=data["stoi"], itos=itos)


class BPETokenizer:
    """本地自训 BPE（HuggingFace tokenizers JSON，如 tokenizer_bpe.json）。"""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFFastTokenizer

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 BPE 词表: {path.resolve()}")
        return cls(HFFastTokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids])

    def save(self, path: str | Path) -> None:
        """BPE 词表通常已由 scripts/train_bpe.py 写好；此处仅按需另存。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))


class HFTokenizer:
    """transformers.AutoTokenizer 包装（默认 Qwen/Qwen2.5-1.5B）。

    只加载分词器文件，不下载整模权重。encode/decode 与 Char/BPE 对齐：
    不加 special tokens，decode 时跳过 special tokens。
    """

    def __init__(self, tokenizer, name_or_path: str):
        self._tok = tokenizer
        self.name_or_path = name_or_path

    @classmethod
    def load(
        cls,
        name_or_path: str | Path = "Qwen/Qwen2.5-1.5B",
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "HFTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "未安装 transformers。请先: pip install -r requirements.txt"
            ) from e

        name = str(name_or_path)
        kwargs: dict = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(cache_dir)
        tok = AutoTokenizer.from_pretrained(name, **kwargs)
        return cls(tok, name)

    @property
    def vocab_size(self) -> int:
        # len(tokenizer) 含已扩展的特殊符；比 vocab_size 属性更贴近实际 embedding 行数
        return int(len(self._tok))

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        """把 tokenizer 文件落到本地目录，便于离线复用。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save_pretrained(str(path))


def load_tokenizer(path: str | Path):
    """按本地文件内容自动识别字符版或自训 BPE。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if '"model"' in raw[:2000] or path.name.endswith("tokenizer_bpe.json"):
        return BPETokenizer.load(path)
    data = json.loads(raw)
    if "stoi" in data and "itos" in data:
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def load_tokenizer_backend(
    backend: str = "hf",
    *,
    hf_id: str = "Qwen/Qwen2.5-1.5B",
    hf_cache_dir: str | Path | None = None,
    local_path: str | Path | None = None,
) -> Union[HFTokenizer, BPETokenizer, CharTokenizer]:
    """统一入口：hf / bpe / char。

    - hf: Hugging Face 模型 id 或本地 tokenizer 目录
    - bpe: 本地 tokenizer_bpe.json
    - char: 本地 tokenizer.json（字符版）
    """
    backend = (backend or "hf").lower().strip()
    if backend == "hf":
        return HFTokenizer.load(hf_id, cache_dir=hf_cache_dir)
    if backend == "bpe":
        if local_path is None:
            raise ValueError("bpe 后端需要 local_path（如 checkpoints/tokenizer_bpe.json）")
        return BPETokenizer.load(local_path)
    if backend == "char":
        if local_path is None:
            raise ValueError("char 后端需要 local_path（如 checkpoints/tokenizer.json）")
        return CharTokenizer.load(local_path)
    raise ValueError(f"未知 tokenizer 后端: {backend!r}（可选 hf / bpe / char）")
