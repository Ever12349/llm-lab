# 语料说明

推荐把训练文本拆成多个 UTF-8 `.txt`，放在 **`data/corpus/`**（一书一文，方便增删）。

仍兼容单文件 **`data/input.txt`**：当 `data/corpus/` 下没有 `.txt` 时，编码 / 训词表会自动用它。

## 常用命令

```bash
# 列出将要使用的语料（不编码）
python scripts/encode_corpus.py --list-only

# 编码（默认：corpus/ 有文件则用目录，否则 input.txt）
python scripts/encode_corpus.py

# 指定目录 / glob / 单文件
python scripts/encode_corpus.py --corpus-dir data/corpus
python scripts/encode_corpus.py --corpus-glob "data/raw_epub/monogatari_*.txt"
python scripts/encode_corpus.py --corpus data/input.txt

# 训 BPE 词表（同一套语料参数）
python scripts/train_bpe.py --corpus-dir data/corpus --vocab-size 32000
```

字符级小实验仍可用：`python train.py --char`（语料过大会占满内存，大语料请走 BPE + memmap）。

路径解析逻辑见项目根目录 `corpus.py`。
