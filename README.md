# 大模型微调（PEFT）学习实验仓库

本仓库是大模型参数高效微调（PEFT）的学习与实验仓库，围绕两种典型架构、两种任务形态，覆盖从数据准备、微调训练到部署的完整流程。

## 两条实验线

| | **Qwen3-0.6B** | **BERT-base-uncased** |
|---|---|---|
| 架构 | Decoder-Only（`AutoModelForCausalLM`） | Encoder-Only（`AutoModelForSequenceClassification`） |
| 任务类型 | 生成（CAUSAL_LM） | 判别（SEQ_CLS，二分类） |
| 数据集 | zhihu-kol（INSTRUCTION/RESPONSE，约 100 万条） | GLUE MRPC（句对语义等价判断） |
| 实验方法 | 全量 SFT、LoRA、AdaLora、QLoRA（4bit NF4）、Prefix-Tuning、推理量化、vLLM 部署 | 全量微调、LoRA、AdaLora、Prompt 系列 |

## 实验成果速览

> 数字均为本地实测（80 条 zhihu 采样 / MRPC 全量），详细分析与踩坑过程见 docs 对应文档。

| 实验线 | 方法 | 状态 / 关键结论 | 详见 |
|---|---|---|---|
| Qwen | SFT | 已完成 | — |
| Qwen | LoRA | eval_loss 4.8659 → 4.5211（**-7.1%**，3 epochs） | [LoRA 分析](docs/LoRA微调requires_grad错误分析与修复.md) |
| Qwen | AdaLora | 调参后 4.8659 → 4.6509（-4.4%，10/10 验证样本逐条改善）；核心教训：**adapter 训练必须显式设 lr**（默认 5e-5 会"没学动"），秩预算需回调驱动 `update_and_allocate` | [AdaLora 审查报告](docs/AdaLora微调代码审查报告.md) |
| Qwen | QLoRA（4bit NF4） | 已跑通；权重显存 1.12→0.82 GiB；**QLoRA 线 eval_loss 与 LoRA 线不可横比**（量化底座不同） | [QLoRA 改造方案](docs/QLoRA微调改造方案.md) |
| Qwen | 4bit NF4 推理 | `test_transformers_qwen3-0.6b.py` 已验证 | — |
| Qwen | vLLM 部署 | 脚本就绪（OpenAI 兼容服务） | — |
| BERT | SFT / LoRA | 已完成 | [LoRA 分析](docs/LoRA微调requires_grad错误分析与修复.md) |
| BERT | AdaLora | 2751 步训练完成（末轮 macro_f1≈0.80）；核心教训：`load_best_model_at_end` 与 AdaLora 的 rank_pattern 演化**不兼容**（size mismatch） | [AdaLora BERT 改造方案](docs/AdaLora微调BERT改造方案.md) |
| BERT | Prompt 系列（Prefix 等） | 未开始（`PrefixTuningConfig` 仅支持 Seq2Seq 的兼容性限制，见方案文档第四节） | [学习方案](docs/大模型微调学习方案.md) |

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
├── docs/                        # 规划 + 各实验线的踩坑/方案文档（见文末索引）
├── model/                       # ❌ 不进 git：本地 HF 模型权重（约 5G）
├── datasets/
│   ├── evaluate/                # ✅ 进 git：vendored 的 HF evaluate 库源码（脚本依赖其本地路径加载指标）
│   ├── zhihu-kol/               # ❌ 不进 git：SFT 原始数据 1.4G + 分词缓存（仅提交 README 数据卡片）
│   ├── glue/                    # ❌ 不进 git：GLUE 各子集 + 分词缓存（仅提交 README）
│   └── .cache/                  # ❌ 不进 git：下载缓存
└── training/                    # ❌ 不进 git：训练产物（脚本自动生成）
    ├── qwen3-0.6b_SFT / qwen3-0.6b_Lora / qwen3-0.6b_AdaLora / qwen3-0.6b_QLora
    └── bert-base-uncased_SFT / bert-base-uncased_Lora / bert-base-uncased_AdaLora
    （每个目录下：output/=checkpoint、logs/=TensorBoard、lora_adapter/=适配器成果）
```

### 脚本清单

`test_` 前缀**不是 pytest 测试**，是自包含的实验脚本（无 main 函数，顶层顺序执行）。运行单个实验即 `python` 执行对应脚本：

| 脚本 | 内容 |
|---|---|
| `test_qwen3-0.6b.py` / `test_Bert.py` | 模型加载与结构探测（chat template、generate） |
| `test_transformers_qwen3-0.6b.py` / `test_transformers_bert.py` | 推理基础；Qwen 侧含 4bit NF4 量化推理 |
| `test_dataset_qwen3-0.6b.py` | zhihu-kol 数据加载/切分/分词/labels mask 流程 |
| `test_SFT_qwen3-0.6b.py` / `test_SFT_bert.py` | 全量微调（Trainer） |
| `test_Lora_qwen3-0.6b.py` / `test_Lora_bert.py` | LoRA 微调（peft）；含"复用已训练成果"双模式（has_adapter 判断）与包装前基线评估 |
| `test_Lora-load_bert.py` | 从 `training/.../lora_adapter` 加载适配器推理 |
| `test_AdaLora_qwen3-0.6b.py` / `test_AdaLora_bert.py` | AdaLora 微调：秩预算回调（`update_and_allocate`）、三阶段调度、`is_trainable=True` 加载 |
| `test_QLora_qwen3-0.6b.py` | QLoRA 微调：BitsAndBytes 4bit NF4 加载 + `prepare_model_for_kbit_training` + paged 优化器 |
| `test_Prefix_qwen3-0.6b.py` | Prefix-Tuning 实验 |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务 |

### 运行方式

所有脚本必须**从仓库根目录运行**（脚本内均为相对路径）：

```bash
conda activate ai-gpu
python test_Lora_qwen3-0.6b.py
```

> LoRA/AdaLora 脚本均有双模式行为：检测到 `training/<线>/lora_adapter/adapter_config.json` 则加载已训练权重跳过训练；删除该目录（或改名）即强制重训。

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
| `datasets/zhihu-kol/`、`datasets/glue/`（README 除外） | 原始数据 + 分词缓存，可再生 |
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
- LoRA 线踩坑全集（requires_grad / collator / 评估方法 / 复用模式基线陷阱）：[docs/LoRA微调requires_grad错误分析与修复.md](docs/LoRA微调requires_grad错误分析与修复.md)
- AdaLora(Qwen) 代码审查与三轮实战复盘（调度回调 / 加载崩溃 / eval_loss 口径 / 效果评估）：[docs/AdaLora微调代码审查报告.md](docs/AdaLora微调代码审查报告.md)
- AdaLora(BERT) 改造方案与 size mismatch 复盘：[docs/AdaLora微调BERT改造方案.md](docs/AdaLora微调BERT改造方案.md)
- QLoRA 改造方案（bnb 配置检查 / 显存实测 / 量化损失）：[docs/QLoRA微调改造方案.md](docs/QLoRA微调改造方案.md)
