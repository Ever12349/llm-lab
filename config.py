"""超参数集中配置（可按机器与语料规模自行调整）。"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def pick_device(override: str | None = None) -> str:
    """自动选择设备：cuda > mps > cpu；可用字符串覆盖。"""
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(
    it: int,
    *,
    max_iters: int,
    learning_rate: float,
    warmup_iters: int = 0,
    min_lr: Optional[float] = None,
) -> float:
    """当前 iter 的学习率：线性 warmup，再余弦衰减到 min_lr。

    - warmup_iters<=0：不预热，第一步就从峰值（或余弦起点）开始。
    - min_lr 为 None 时等于 learning_rate：warmup 之后保持恒定（mid100 默认如此）。
    - 若本轮 max_iters 短于设定的 warmup（例如冒烟只跑几百步），
      把 warmup 压缩到本轮长度，避免短跑全程停在接近 0 的 lr。
    - it 与 max_iters 都按「全局 iter」理解：max_iters 应传本轮结束时的全局步号
      （从头训即 cfg.max_iters；续训则为 start_iter + cfg.max_iters）。
    """
    peak = float(learning_rate)
    floor = peak if min_lr is None else float(min_lr)
    warmup = max(0, int(warmup_iters))
    end = max(1, int(max_iters))
    step = max(1, int(it))
    if warmup > 0:
        warmup = min(warmup, end)
        if step < warmup:
            return peak * step / warmup
    if end <= warmup:
        return peak
    progress = (step - warmup) / (end - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


# 模型档位：默认 seed20（~20M，旧实验）；mid100 ~100M；mid200 ~200M
# mid100 / seed20 只改形状，不改学习率等优化默认，避免旧 checkpoint 续训被悄悄改掉。
MODEL_TIERS = {
    "seed20": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 64,
        "batch_size": 32,
        "checkpoint_name_hf": "ckpt_hf.pt",
        "summary": "微型种子 ~20M（旧默认）",
    },
    "mid100": {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "block_size": 512,
        "batch_size": 4,  # 训练时建议再降到 2–4 限性能
        "checkpoint_name_hf": "ckpt_hf_mid100.pt",
        "summary": "方案 A ~100–120M（d=512, L=8；MLP=4d→d_ff=2048）",
    },
    "mid200": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "block_size": 512,
        "batch_size": 1,  # M1 统一内存紧张；训练时配合 grad-accum
        "checkpoint_name_hf": "ckpt_hf_mid200_v2.pt",
        "learning_rate": 1e-4,
        "min_lr": 1e-5,
        "warmup_iters": 2000,
        "adam_eps": 1e-6,
        "decay_only_2d": True,
        "dropout": 0.0,
        "summary": "方案 mid200 ~200M（d=768, L=12, H=12；MLP=4d→d_ff=3072）",
    },
}

# 档位可选覆盖的优化/正则字段（缺省则沿用 TrainConfig 默认，mid100 不受影响）
_TIER_OPTIONAL = (
    "learning_rate",
    "min_lr",
    "warmup_iters",
    "adam_eps",
    "decay_only_2d",
    "dropout",
)


def apply_model_tier(cfg: "TrainConfig", tier: str) -> "TrainConfig":
    """按档位覆盖模型宽深与默认 HF checkpoint 名；未知档位抛 ValueError。"""
    if tier not in MODEL_TIERS:
        raise ValueError(f"未知档位 {tier!r}，可选: {', '.join(MODEL_TIERS)}")
    t = MODEL_TIERS[tier]
    cfg.n_layer = int(t["n_layer"])
    cfg.n_head = int(t["n_head"])
    cfg.n_embd = int(t["n_embd"])
    cfg.block_size = int(t["block_size"])
    cfg.batch_size = int(t["batch_size"])
    for key in _TIER_OPTIONAL:
        if key in t:
            setattr(cfg, key, t[key])
    # 仅改 HF 默认 ckpt 名；--bpe/--char 仍用各自路径
    if getattr(cfg, "tokenizer_backend", "hf") == "hf":
        cfg.checkpoint_name = str(t["checkpoint_name_hf"])
    return cfg


@dataclass
class TrainConfig:
    """微型因果语言模型的训练与模型超参。

    默认走 Hugging Face 词表（Qwen/Qwen2.5-1.5B，仅 tokenizer）+ memmap 语料。
    自训 BPE / 字符版仍可用 --bpe / --char 对照。
    默认刻意保持很小，方便 CPU / Apple MPS 上快速跑通、读懂代码。
    调大 n_layer / n_head / n_embd / block_size 可提升容量，但会更慢、更吃显存。
    """

    # --- 序列与数据 ---
    block_size: int = 64  # 上下文长度（token 数）
    batch_size: int = 32
    train_ratio: float = 0.9

    # --- 模型（微型 GPT）---
    # vocab_size 由 tokenizer 决定，训练时再写入
    vocab_size: int = 0
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True  # Linear / LayerNorm 是否使用 bias

    # --- 优化 ---
    # mid100 / 默认：恒定 3e-4（warmup=0、min_lr=None）。mid200 档位会覆盖为
    # 1e-4、warmup 2000、余弦衰减到 1e-5，见 MODEL_TIERS 与 get_lr()。
    learning_rate: float = 3e-4
    min_lr: Optional[float] = None  # None=不衰减，保持为 learning_rate
    warmup_iters: int = 0
    weight_decay: float = 0.1
    adam_eps: float = 1e-8
    decay_only_2d: bool = False  # True 时只对 2D 权重做 weight decay（mid200）
    max_iters: int = 50000
    eval_interval: int = 500
    eval_iters: int = 50
    grad_clip: float = 1.0

    # --- 词表后端：hf（默认）/ bpe / char ---
    tokenizer_backend: str = "hf"
    hf_tokenizer_id: str = "Qwen/Qwen2.5-1.5B"
    # 项目内 HF 缓存目录（只存 tokenizer，不下整模）
    hf_cache_dir: str = "checkpoints/hf"

    # --- 路径与设备 ---
    # data_path：字符版默认单文件；也可用 data/corpus/ 多文件（见 corpus.py）
    data_path: str = "data/input.txt"
    corpus_dir: str = "data/corpus"  # 推荐：每本书一个 .txt；空则回退 data_path
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    # 种子语料编码产物（download_seed_corpus → encode --corpus-dir data/seed）：
    # data/seed_train.bin / data/seed_val.bin
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ckpt_hf.pt"
    # 本地文件名（仅 bpe / char 后端使用）
    tokenizer_name: str = "tokenizer_bpe.json"
    # 对照路径
    bpe_checkpoint_name: str = "ckpt_bpe.pt"
    bpe_tokenizer_name: str = "tokenizer_bpe.json"
    char_checkpoint_name: str = "ckpt_char.pt"
    char_tokenizer_name: str = "tokenizer.json"
    device: str = field(default_factory=pick_device)
    seed: int = 1337

    def checkpoint_path(self) -> str:
        return f"{self.checkpoint_dir}/{self.checkpoint_name}"

    def best_checkpoint_path(self) -> str:
        """与主 ckpt 同目录的 best-val 路径：ckpt_foo.pt → ckpt_foo_best.pt。"""
        p = self.checkpoint_name
        if p.endswith(".pt"):
            return f"{self.checkpoint_dir}/{p[:-3]}_best.pt"
        return f"{self.checkpoint_dir}/{p}_best.pt"

    def tokenizer_path(self) -> str:
        return f"{self.checkpoint_dir}/{self.tokenizer_name}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
