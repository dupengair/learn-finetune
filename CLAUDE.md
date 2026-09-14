# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

大模型微调（PEFT）学习/实验仓库，非 git 仓库、无测试框架、无 lint 配置。按 `大模型微调学习方案.md` 规划分两条实验线：

- **Qwen3-0.6B**（Decoder-Only / CausalLM）：全量 SFT、LoRA、QLoRA/量化、vLLM 部署 — 生成任务
- **BERT-base-uncased**（Encoder-Only / SequenceClassification）：全量微调、LoRA、Prompt 系列算法 — GLUE MRPC 判别任务

## 运行环境与命令

- 命令约束：一切分析代码的活动都要使用codegraph工具
- Conda 环境：`ai-gpu`（`/home/dupengair/shared/conda/anaconda3/envs/ai-gpu`）。关键版本：torch 2.8.0 / transformers 4.55.0 / peft 0.18.1 / datasets 4.5.0 / evaluate 0.4.6 / vllm 0.10.2 / bitsandbytes 0.50.2
- **所有脚本必须从仓库根目录运行**，脚本内全部是相对路径（`./model/`、`./datasets/`、`./training/`）：
  ```bash
  python test_Lora_qwen3-0.6b.py
  ```
- 显卡为 RTX 4050 Laptop（6GB 显存）+ WSL2。脚本里 batch_size=1、gradient_accumulation_steps=4、gradient_checkpointing=True、fp16/bf16、adamw_torch_fused 等 OOM 规避参数是**刻意设置**，不要当 bug 改掉。
- 所有模型/数据集加载均带 `local_files_only=True`（离线环境），不要改成联网加载。

## 脚本命名约定

`test_<方法>_<模型>.py` —— **`test_` 前缀不是 pytest 测试**，是自包含的实验/训练脚本（顶层顺序执行，无 main 函数、无函数封装）。运行单个"实验"就是 `python` 执行对应脚本。

| 脚本 | 内容 |
|---|---|
| `test_qwen3-0.6b.py` / `test_Bert.py` | 模型加载与结构探测（chat template、generate） |
| `test_transformers_qwen3-0.6b.py` / `test_transformers_bert.py` | 推理基础；Qwen 侧含 bitsandbytes 4bit NF4 量化 |
| `test_dataset_qwen3-0.6b.py` | zhihu-kol 数据加载/切分/分词/labels mask 流程 |
| `test_SFT_qwen3-0.6b.py` / `test_SFT_bert.py` | 全量微调（Trainer） |
| `test_Lora_qwen3-0.6b.py` / `test_Lora_bert.py` | LoRA 微调（peft），先跑训练前基线评估 |
| `test_Lora-load_bert.py` | 从 `training/.../lora_adapter` 加载适配器推理 |
| `test_Prefix_qwen3-0.6b.py` | Prefix-Tuning 实验（注意兼容性坑，见下） |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务（`VLLM_USE_V1=0`，`build_app` 入参是 Namespace） |

## 目录约定

- `model/`：本地 HF 模型（`Qwen3-0.6B`、`bert-base-uncased`、`bert-base-uncased-finetuned`）
- `datasets/zhihu-kol/data/`：SFT 数据（INSTRUCTION/RESPONSE 字段，约 100 万条）；`cache/` 存分词后的 arrow 缓存
- `datasets/glue/`：GLUE 各子集本地数据（BERT 用 mrpc）；`cache-mrpc/` 为分词缓存
- `datasets/evaluate/`：**vendored 的 HF evaluate 库完整源码**（只为用 `./datasets/evaluate/metrics/glue/glue.py` 加载 MRPC 指标）。注意：CodeGraph 索引会命中这里的库代码而非根目录实验脚本
- `training/<模型>_<方法>/`：`output/`（checkpoint）、`logs/`（TensorBoard）、`lora_adapter/`（peft 适配器）

## 两条实验线的核心差异（不要混用）

| | Qwen3-0.6B | BERT |
|---|---|---|
| 模型类 | `AutoModelForCausalLM` | `AutoModelForSequenceClassification`（不能用 CausalLM） |
| 任务/数据 | CAUSAL_LM，zhihu-kol 生成数据 | SEQ_CLS，GLUE mrpc（`num_labels=2`） |
| collator | `DataCollatorForLanguageModeling(mlm=False)` | `DataCollatorWithPadding` |
| LoRA target | `q/k/v/o/gate/up/down_proj` 全套 | `query/key/value` + **必须** `modules_to_save=["classifier"]` |
| 分词 | 手工拼 Qwen IM 模板，prompt 部分labels 置 -100 只对回答算 loss；`tokenizer.pad_token = eos_token` | 句对 pair 分词，max_length=128 |

SFT 数据流程是固定管线：加载 → `train_test_split` 两次切 80/10/10（seed=42）→ `map(tokenize_func, batched=True)` 逐 split 指定 `cache_file_name` → Trainer。新脚本通常复制现有脚本改配置，保持此结构。

## 关键坑（改代码前必读）

1. **分词缓存不自动失效**：`.map(cache_file_name=...)` 复用已有 arrow 缓存。修改 `tokenize_func` 或 max_length 后必须删除 `datasets/**/cache*/` 下对应 `.arrow` 文件，否则改动不生效。
2. **peft 的 `PrefixTuningConfig` ≠ 论文 P-Tuning-v2**：前者仅支持 Seq2Seq（T5/BART），不能直接用于 Qwen（CausalLM）或 BERT（SEQ_CLS）。BERT 侧 Prompt 系列实验需注意此兼容性限制（详见方案文档第四节）。
3. **`device_map="auto"` 与 Trainer 冲突**：全量 SFT 脚本里已注释掉（Trainer 自行迁移模型）；仅在纯推理脚本和 LoRA 脚本中保留。
4. **generate 前必须 `tokenizer.padding_side = "left"`**，且 PEFT 模型生成用 `model_lora.eval()` + `torch.no_grad()`。
5. **LoRA 保存**：`trainer.save_model()` 只存适配器不存主干；加载端（`test_Lora-load_bert.py`）通过 `PeftConfig` 读回 base model 路径，`num_labels` 必须与训练一致。
6. `eval_strategy`（新版）不要写成 `evaluation_strategy`。
7. 代码注释和文档均为中文，新增注释保持中文。
