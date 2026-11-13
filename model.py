"""微型 GPT：因果自注意力 + MLP 的字符级语言模型。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：只看当前位置及之前的 token。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 一次算出 Q、K、V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 下三角 mask：阻止看到未来 token
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """两层前馈网络：扩展 → GELU → 压回。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """标准 Transformer 解码块：LN → Attn → 残差，再 LN → MLP → 残差。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """微型因果语言模型。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size > 0
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置 embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重共享：embedding 与输出头共用，常见做法且省参数
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # 残差投影再缩一次：层数越多，初始残差流越大。按 GPT-2 / nanoGPT
        # 用 1/sqrt(2L) 压低注意力输出（c_proj）和 MLP 输出（proj），减轻后期膨胀。
        # 其余 Linear / Embedding 仍保持 N(0, 0.02)。
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("mlp.proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=resid_std)
        n_params = sum(p.numel() for p in self.parameters())
        # 打印放在外部；这里只算好供调用方使用
        self._n_params = n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) token ids
        返回 logits (B, T, vocab)，若给了 targets 则额外返回标量 loss。
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"序列长度 {T} 超过 block_size={self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> torch.Tensor:
        """自回归采样：每次根据已有上下文预测下一个 token。

        若传入 token_callback，每采样出一个新 token id 就立刻调用一次，
        便于流式打印（不必等整段结束）。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if token_callback is not None:
                token_callback(int(next_id[0, 0].item()))
        return idx

    def num_params(self) -> int:
        return self._n_params
