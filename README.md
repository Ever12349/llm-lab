# llm-lab · 学习型微型 Transformer

从零实现的**因果语言模型**（nanoGPT 教学风格），用 Python + PyTorch 把「分词 → 模型 → 训练 → 生成」串成一个可跑通的小项目。目标是**读懂代码**，不是追求规模或效果。

当前默认路径是 **Hugging Face 开源词表（`Qwen/Qwen2.5-1.5B`，只下 tokenizer）+ memmap 语料**。  
旧的**自训 BPE（约 32k）**与**字符级**模型仍保留在磁盘上，用 `--bpe` / `--char` 对照。

## 目录结构

```
llm-lab/
  README.md
  requirements.txt
  config.py          # 超参与设备选择（默认 HF 词表）
  corpus.py          # 多文件语料路径解析（单文件 / 目录 / glob）
  tokenizer.py       # CharTokenizer + BPETokenizer + HFTokenizer
  model.py           # 微型 GPT
  dataset.py         # memmap / 字符 batch 采样（uint16 / uint32）
  train.py           # 训练入口（默认 HF）
  generate.py        # 采样生成入口（流式输出）
  scripts/
    train_bpe.py     # （可选）自训 BPE 词表
    encode_corpus.py # 语料 → data/train.bin + val.bin
    download_seed_corpus.py  # 流式拉维基+FineWeb 种子 → data/seed/
    demo_bpe.py      # BPE encode/decode 演示
  data/
    corpus/          # 可选：本地小说等多文件 .txt
    seed/            # 种子语料（维基+网页截断）
    input.txt        # 兼容：单文件大语料（corpus/ 为空时自动使用）
    train.bin        # 预编码 token memmap（encode 后生成）
    val.bin
  checkpoints/
    hf/              # HF tokenizer 缓存 / 本地副本
    tokenizer_bpe.json   # 旧自训 BPE（对照，非默认）
    ckpt_bpe.pt          # 旧 BPE 权重（对照）
    tokenizer.json       # 字符级词表（对照）
    ckpt_char.pt         # 字符级权重（对照）
    ckpt_hf.pt           # HF 词表训练出的权重（默认训练目标）
```

## 产物对照

| 路径 | 含义 |
|------|------|
| `Qwen/Qwen2.5-1.5B`（HF） | **默认词表**（中日英友好，约 15 万+） |
| `checkpoints/hf/` | 本地 HF tokenizer 缓存 |
| `checkpoints/ckpt_hf.pt` | HF 词表 + seed20（~20M）权重 |
| `checkpoints/ckpt_hf_mid100.pt` | HF 词表 + mid100（方案 A ~100M）权重 |
| `checkpoints/tokenizer_bpe.json` | 旧自训 BPE（仍保留，非默认） |
| `checkpoints/ckpt_bpe.pt` | 旧 BPE 权重（对照） |
| `checkpoints/tokenizer.json` / `ckpt_char.pt` | 字符级对照 |
| `data/train.bin` / `val.bin` | 预编码 token（HF 大词表用 **uint32**） |

## 环境准备

```bash
cd /Users/qianxiao/worker/llm-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

需要：`torch`、`transformers`、`huggingface_hub`、`tokenizers`、`numpy`。

设备：自动选 `cuda` > `mps` > `cpu`，也可用 `--device` 强制指定。

## 使用 Qwen 词表（默认）

首次会从 Hugging Face 下载 **tokenizer 文件**（不是 1.5B 整模权重），缓存到 `checkpoints/hf/`。

```bash
# 1）用 Qwen 词表编码语料 → train.bin / val.bin
python scripts/encode_corpus.py
# 或指定 id：
python scripts/encode_corpus.py --hf-tokenizer Qwen/Qwen2.5-1.5B

# 快速冒烟（只编一小段，不覆盖正式 bin）
python scripts/encode_corpus.py --smoke

# 2）从头训练微型 GPT（vocab_size 自动跟 Qwen）
python train.py
# 续训：
python train.py --resume

# 3）生成（流式）
python generate.py --prompt "今天" --max-new-tokens 100
```

换别的 HF 词表：

```bash
python scripts/encode_corpus.py --hf-tokenizer Qwen/Qwen2.5-0.5B
python train.py --hf-tokenizer Qwen/Qwen2.5-0.5B
```

说明：Qwen 词表很大（约 15 万+），embedding 会占不少参数；本项目默认仍用微型 GPT（`--tier seed20`：约 4 层 × 128 维）；也可用 `--tier mid100` 拉到约 100M。**只是借用 Qwen 的分词器**，不会加载 Qwen 的模型权重。

## 旧自训 BPE / 字符对照

```bash
# BPE（需已有 tokenizer_bpe.json）
python scripts/encode_corpus.py --bpe
python train.py --bpe
python generate.py --bpe --prompt "今天"

# 字符级
python train.py --char --max-iters 500
python generate.py --char --prompt "春"
```

自训 BPE 词表（可选，非默认流程）：

```bash
python scripts/train_bpe.py --corpus-dir data/corpus --vocab-size 32000
```

## 限制训练占用的硬件（可选）

```bash
python train.py --device cpu --cpu-threads 2 --batch-size 8 --max-iters 100
python train.py --device mps --batch-size 8
```

- `--cpu-threads N`：`N>0` 时限制线程；`0`=不限制。
- **MPS / CUDA 很难「只用一部分 GPU」**；想明显省资源用 `--device cpu --cpu-threads 2`。


## 模型档位（`--tier`）

| 档位 | 规模 | 配置 | 默认 checkpoint |
|------|------|------|-----------------|
| `seed20`（默认） | ~20M | L=4, H=4, d=128, block=64 | `ckpt_hf.pt` |
| `mid100` | ~100–120M | L=8, H=8, d=512, block=512（MLP 4d→d_ff=2048） | `ckpt_hf_mid100.pt` |

```bash
# 方案 A：mid100 + 种子语料，MPS 小 batch 限性能
python train.py --tier mid100 \
  --train-bin data/seed_train.bin --val-bin data/seed_val.bin \
  --device mps --batch-size 2 --max-iters 50000

# 生成
python generate.py --tier mid100 --prompt "今天" --device mps
```

形状与旧 seed20 不同，**从头训**，不会覆盖 `ckpt_hf.pt`。

## 语料怎么放（多文件）

**推荐**：`data/corpus/` 下的 `.txt`（**支持子目录递归**）。

```bash
python scripts/encode_corpus.py --list-only
python scripts/encode_corpus.py --corpus-dir data/corpus
python scripts/encode_corpus.py --corpus data/input.txt --corpus-dir data/corpus
```

未传语料参数时：有 `data/corpus/**/*.txt` 用目录，否则回退 `data/input.txt`。

## 种子语料（开训底座）

目标：先训一个多语**种子 LLM**，再叠指令/工具等能力。默认预算约 **60M token**（中日英大致均分；维基约 30% + FineWeb 系网页约 70%）。本地小说仍放 `data/corpus/`，建议混入 ≤20%。

```bash
# 依赖（含 datasets）
pip install -r requirements.txt

# 冒烟：约 1M token → data/seed/*.txt
python scripts/download_seed_corpus.py --smoke

# 正式：默认 60M（可改 --budget 30000000）
python scripts/download_seed_corpus.py --budget 60000000

# 编码（写入 seed_*，不覆盖旧小说 train.bin）
python scripts/encode_corpus.py \
  --corpus-dir data/seed \
  --train-bin data/seed_train.bin \
  --val-bin data/seed_val.bin

# 训练
python train.py --train-bin data/seed_train.bin --val-bin data/seed_val.bin
```

详情与数据源表见 [`data/seed/README.md`](data/seed/README.md)。

## 推荐阅读顺序

1. `config.py` — 超参与默认 HF 词表
2. `corpus.py` — 多文件语料如何解析
3. `tokenizer.py` — Char / BPE / HF 如何变成 id
4. `model.py` — 注意力与 Transformer 块
5. `scripts/encode_corpus.py` → `dataset.py` → `train.py` → `generate.py`
