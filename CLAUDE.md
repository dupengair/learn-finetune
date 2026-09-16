# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

大模型微调（PEFT）学习/实验仓库，git 仓库、无测试框架、无 lint 配置。按 `大模型微调学习方案.md` 规划的两条实验线**已全部完成**：

- **Qwen3-0.6B**（Decoder-Only / CausalLM）：全量 SFT、LoRA、AdaLora、QLoRA/量化、Prefix-Tuning、Prompt Tuning、P-Tuning v2（机制对照）、vLLM 部署 — 生成任务
- **BERT-base-uncased**（Encoder-Only / SequenceClassification）：全量微调、LoRA、AdaLora、P-Tuning v2 — GLUE MRPC 判别任务

## 运行环境与命令

- 命令约束：一切分析代码的活动都要使用codegraph工具
- Conda 环境：`ai-gpu`（`/home/dupengair/shared/conda/anaconda3/envs/ai-gpu`）。关键版本：torch 2.8.0 / transformers 4.55.0 / peft 0.18.1 / datasets 4.5.0 / evaluate 0.4.6 / vllm 0.10.2 / bitsandbytes 0.50.2 / tensorboard 2.20.0
- **所有脚本必须从仓库根目录运行**，脚本内全部是相对路径（`./model/`、`./datasets/`、`./training/`）：
  ```bash
  python test_Lora_qwen3-0.6b.py
  ```
- 显卡为 RTX 4050 Laptop（6GB 显存）+ WSL2。脚本里 batch_size=1、gradient_accumulation_steps=4、fp16/bf16、adamw_torch_fused 等 OOM 规避参数是**刻意设置**，不要当 bug 改掉。**gradient_checkpointing 例外**：SFT/LoRA/QLoRA/AdaLora 线开着，但 prompt 系（KV 注入）脚本必须 False（见坑 3），不要"统一开启"。
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
| `test_AdaLora_qwen3-0.6b.py` / `test_AdaLora_bert.py` | AdaLora：秩预算需回调驱动 `update_and_allocate`；加载必须 `is_trainable=True`（见坑 4） |
| `test_QLora_qwen3-0.6b.py` | QLoRA：4bit NF4 + `prepare_model_for_kbit_training` + paged 优化器 |
| `test_Prefix_qwen3-0.6b.py` | Prefix Tuning **原版论文形态**（`prefix_projection=True`，每层 KV 注入，效果线） |
| `test_Prompt_qwen3-0.6b.py` | Prompt Tuning（TEXT 初始化虚拟 embedding，lr=5e-3） |
| `test_PTuningV2_qwen3-0.6b.py` | P-Tuning v2 论文形态（`prefix_projection=False`，机制对照线，小数据下崩坏属预期） |
| `test_PTuningV2_bert.py` | BERT×MRPC 的 P-Tuning v2（SEQ_CLS，含全 1 退化解拯救方案） |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务（`VLLM_USE_V1=0`，`build_app` 入参是 Namespace） |

## 目录约定

- `model/`：本地 HF 模型（`Qwen3-0.6B`、`bert-base-uncased`、`bert-base-uncased-finetuned`）
- `datasets/zhihu-kol/data/`：SFT 数据（INSTRUCTION/RESPONSE 字段，约 100 万条）；`cache/` 存分词后的 arrow 缓存
- `datasets/glue/`：GLUE 各子集本地数据（BERT 用 mrpc）；`cache-mrpc/` 为分词缓存
- `datasets/evaluate/`：**vendored 的 HF evaluate 库完整源码**（只为用 `./datasets/evaluate/metrics/glue/glue.py` 加载 MRPC 指标）。注意：CodeGraph 索引会命中这里的库代码而非根目录实验脚本
- `training/<模型>_<方法>/`：`output/`（checkpoint）、`logs/`（TensorBoard）、`lora_adapter/`（peft 适配器）。注意 `qwen3-0.6b_PTuning-V2` / `bert-base-uncased_PTuning-V2` 用连字符，其余方法下划线

## 两条实验线的核心差异（不要混用）

| | Qwen3-0.6B | BERT |
|---|---|---|
| 模型类 | `AutoModelForCausalLM` | `AutoModelForSequenceClassification`（不能用 CausalLM） |
| 任务/数据 | CAUSAL_LM，zhihu-kol 生成数据 | SEQ_CLS，GLUE mrpc（`num_labels=2`） |
| collator | `DataCollatorForLanguageModeling(mlm=False)` | `DataCollatorWithPadding` |
| LoRA target | `q/k/v/o/gate/up/down_proj` 全套 | `query/key/value` + **必须** `modules_to_save=["classifier"]` |
| prompt 系 | `PrefixTuningConfig`（映射见坑 2）；gc 必须 False（坑 3） | 同左；SEQ_CLS 可用，peft **自动**把 classifier 加入 modules_to_save，无需手动传 |
| 分词 | 手工拼 Qwen IM 模板，prompt 部分labels 置 -100 只对回答算 loss；`tokenizer.pad_token = eos_token` | 句对 pair 分词，max_length=128 |

SFT 数据流程是固定管线：加载 → `train_test_split` 两次切 80/10/10（seed=42）→ `map(tokenize_func, batched=True)` 逐 split 指定 `cache_file_name` → Trainer。新脚本通常复制现有脚本改配置，保持此结构。

## 关键坑（改代码前必读）

1. **分词缓存不自动失效**：`.map(cache_file_name=...)` 复用已有 arrow 缓存。修改 `tokenize_func` 或 max_length 后必须删除 `datasets/**/cache*/` 下对应 `.arrow` 文件，否则改动不生效。
2. **【勘误】prompt 系方法与 peft 的映射及可用性**：旧说法"`PrefixTuningConfig` 仅支持 Seq2Seq，不能用于 Qwen/BERT"在 peft 0.18.1 **已过时**——Qwen（`PeftModelForCausalLM.forward` 有 PREFIX_TUNING 分支）与 BERT（SEQ_CLS，transformers 4.55 重构版 `BertSelfAttention` 已有 KV 路径）**均已实跑验证可用**。正确映射：P-Tuning v2 = `PrefixTuningConfig(prefix_projection=False)`（默认）；原版 Prefix Tuning = `prefix_projection=True`；P-Tuning v1 = `PromptEncoderConfig`。注意 `encoder_hidden_size` 等 `encoder_*` 参数只在 `prefix_projection=True` 时有效/存在，v2 形态传了会 TypeError。小数据（百条级）× 小模型下只有 `projection=True` + 大 lr 才稳定（v2 纯形态实测 +119% 崩坏，系论文前提不满足，非 bug）。
3. **gradient_checkpointing 与 KV 注入结构性冲突**：gc 把每层 `past_key_value` 置 None → prefix/PTuningV2 的可训练参数脱离计算图。Qwen 侧表现为 loss 恒定的"假跑"（仅刷警告），BERT 侧**连警告都没有**（实测 prompt embedding 梯度静默变 None）。KV 注入类方法必须 `gradient_checkpointing=False`；Prompt Tuning / P-Tuning v1（输入拼接）不受此限。
4. **加载适配器的 `is_trainable` 三分规则**：prompt learning 系（Prefix/Prompt/PTuningV2）**禁止 True**（peft 源码直接 raise ValueError）；LoRA/QLoRA 不传（默认 False，评估够用）；AdaLora **必须 True**（否则 evaluate 前向访问 `trainable_adapter_name` 崩 AttributeError）。
5. **prompt/embedding 系的 lr 用 1e-3~5e-3**：1e-4 是 adapter 系直觉，会导致"学不动"——BERT 侧卡在"全预测多数类"退化解（`len(set(preds))==1` 是硬检测信号），Qwen 侧训练前后生成完全不变。冻结越彻底，需要的步长越大。
6. **双模式脚本的 `training_args`/`trainer` 必须顶格定义**（if/else 分支外）：加载模式（has_adapter=True）也要走 evaluate/predict；缩进在 else 内会导致二次运行 `NameError: trainer is not defined`（Lora_bert/PTuningV2_bert 均踩过）。
7. **`device_map="auto"` 与 Trainer 冲突**：全量 SFT 脚本里已注释掉（Trainer 自行迁移模型）；仅在纯推理脚本和 LoRA 脚本中保留。
8. **generate 前必须 `tokenizer.padding_side = "left"`**，且 PEFT 模型生成用 `model_lora.eval()` + `torch.no_grad()`。
9. **LoRA 保存**：`trainer.save_model()` 只存适配器不存主干；加载端（`test_Lora-load_bert.py`）通过 `PeftConfig` 读回 base model 路径，`num_labels` 必须与训练一致。
10. `eval_strategy`（新版）不要写成 `evaluation_strategy`。
11. 代码注释和文档均为中文，新增注释保持中文。
