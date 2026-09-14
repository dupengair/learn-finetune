# 大模型微调（PEFT）学习实验仓库

本仓库是大模型参数高效微调（PEFT）的学习与实验仓库，围绕两种典型架构、两种任务形态，覆盖从数据准备、微调训练到部署的完整流程。

## 两条实验线

| | **Qwen3-0.6B** | **BERT-base-uncased** |
|---|---|---|
| 架构 | Decoder-Only（`AutoModelForCausalLM`） | Encoder-Only（`AutoModelForSequenceClassification`） |
| 任务类型 | 生成（CAUSAL_LM） | 判别（SEQ_CLS，二分类） |
| 数据集 | zhihu-kol（INSTRUCTION/RESPONSE，约 100 万条） | GLUE MRPC（句对语义等价判断） |
| 实验方法 | 全量 SFT、LoRA、Prefix-Tuning、vLLM 部署、4bit NF4 量化 | 全量微调、LoRA、Prompt 系列 |

## 运行环境

- **硬件**：RTX 4050 Laptop（6GB 显存）+ WSL2
- **Conda 环境**：`ai-gpu`
- **关键版本**：torch 2.8.0 / transformers 4.55.0 / peft 0.18.1 / datasets 4.5.0 / evaluate 0.4.6 / vllm 0.10.2 / bitsandbytes 0.50.2

> 脚本中的 `batch_size=1`、`gradient_accumulation_steps=4`、`gradient_checkpointing=True` 等参数是为 6GB 显存刻意设置的 OOM 规避方案；所有模型/数据集加载均使用 `local_files_only=True`（离线环境）。

## 目录与文件结构

```
.
├── test_*.py                    # 自包含实验/训练脚本（顶层顺序执行，非 pytest 测试）
├── CLAUDE.md                    # AI 助手项目配置（含架构约定与踩坑记录，亦可作项目说明阅读）
├── docs/
│   ├── 大模型微调学习方案.md      # 实验总体规划
│   └── LoRA微调requires_grad错误分析与修复.md   # 踩坑笔记
├── model/                       # ❌ 不进 git：本地 HF 模型权重（约 5G）
├── datasets/
│   ├── evaluate/                # ✅ 进 git：vendored 的 HF evaluate 库源码（脚本依赖其本地路径加载指标）
│   ├── zhihu-kol/               # ❌ 不进 git：SFT 原始数据 1.4G + 分词缓存 53G（仅提交 README 数据卡片）
│   ├── glue/                    # ❌ 不进 git：GLUE 各子集 161M + 分词缓存（仅提交 README）
│   └── .cache/                  # ❌ 不进 git：下载缓存
└── training/
    ├── <模型>_<方法>/
    │   ├── output/              # ❌ 不进 git：训练 checkpoint（体积大、可再生）
    │   ├── logs/                # ❌ 不进 git：TensorBoard 日志
    │   └── lora_adapter/        # ❌ 不进 git：LoRA 适配器（最终成果，仅保留在本地）
    ├── qwen3-0.6b_SFT / qwen3-0.6b_Lora
    └── bert-base-uncased_SFT / bert-base-uncased_Lora
```

### 脚本清单

`test_` 前缀**不是 pytest 测试**，是自包含的实验脚本（无 main 函数，顶层顺序执行）。运行单个实验即 `python` 执行对应脚本：

| 脚本 | 内容 |
|---|---|
| `test_qwen3-0.6b.py` / `test_Bert.py` | 模型加载与结构探测（chat template、generate） |
| `test_transformers_qwen3-0.6b.py` / `test_transformers_bert.py` | 推理基础；Qwen 侧含 4bit NF4 量化 |
| `test_dataset_qwen3-0.6b.py` | zhihu-kol 数据加载/切分/分词/labels mask 流程 |
| `test_SFT_qwen3-0.6b.py` / `test_SFT_bert.py` | 全量微调（Trainer） |
| `test_Lora_qwen3-0.6b.py` / `test_Lora_bert.py` | LoRA 微调（peft），先跑训练前基线评估 |
| `test_Lora-load_bert.py` | 从 `training/.../lora_adapter` 加载适配器推理 |
| `test_Prefix_qwen3-0.6b.py` | Prefix-Tuning 实验 |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务 |

### 运行方式

所有脚本必须**从仓库根目录运行**（脚本内均为相对路径）：

```bash
conda activate ai-gpu
python test_Lora_qwen3-0.6b.py
```

## 克隆后需要自行准备的资源

以下大体积资源不进 git，克隆后需按下方说明放置到对应路径，脚本才能运行：

| 资源 | 放置路径 | 获取方式 |
|---|---|---|
| Qwen3-0.6B 模型 | `model/Qwen3-0.6B/` | HuggingFace / ModelScope 下载后放本地 |
| bert-base-uncased 模型 | `model/bert-base-uncased/` | 同上 |
| zhihu-kol 数据集 | `datasets/zhihu-kol/data/` | HuggingFace 下载 parquet 原始数据 |
| GLUE mrpc 数据 | `datasets/glue/mrpc/` | `datasets` 库下载 GLUE 各子集 |

> 训练产物（`training/**/output/`、`logs/`、`lora_adapter/`）由本地训练脚本自动生成，无需下载。

## Git 使用说明

### 不提交远程的目录（见 `.gitignore`）

| 目录 | 原因 |
|---|---|
| `model/` | 模型权重约 5G，可从官方渠道重新下载 |
| `datasets/zhihu-kol/`、`datasets/glue/`（README 除外） | 原始数据 + 分词缓存共约 55G，可再生 |
| `training/*` | checkpoint、日志体积大且可再生；适配器成果仅保留本地 |
| `.codegraph/`、`__pycache__/`、`tmp_trainer/` | 本地索引、编译缓存、临时产物 |

### 日常提交流程

```bash
git add -A                      # 暂存所有变更（新增/修改/删除），幂等可重复执行
git commit -m "feat: 描述本次改动"
git push                        # 首次推送用 git push -u origin main
```

> `git add` 是幂等覆盖：暂存后若又修改了文件，再次 `git add -A` 即可让暂存区拿到最新版本。
> 只想撤销暂存（不动磁盘文件）：`git restore --staged <文件>`；只从暂存区移除已跟踪文件：`git rm -r --cached <路径>`。

### 首次关联远程仓库（已完成，留档备查）

```bash
git init
git remote add origin https://github.com/dupengair/learn-finetune.git
git push -u origin main         # -u 建立跟踪关系，之后可省略 origin main
```

## 更多文档

- 实验总体规划：[docs/大模型微调学习方案.md](docs/大模型微调学习方案.md)
- 踩坑记录：[docs/LoRA微调requires_grad错误分析与修复.md](docs/LoRA微调requires_grad错误分析与修复.md)
