# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

大模型微调（PEFT）学习/实验仓库，git 仓库、无测试框架、无 lint 配置。按 `docs/大模型微调学习方案.md` 规划的两条实验线**已全部完成**（该文档现已刷新为「规划 + 复盘 + 35 条踩坑总集 + 排障工具箱」，是改动代码前最值得先读的一篇）：

- **Qwen3-0.6B**（Decoder-Only / CausalLM）：全量 SFT、LoRA、AdaLora、QLoRA/量化、Prefix-Tuning、Prompt Tuning、P-Tuning v2（机制对照）、vLLM 部署 — 生成任务
- **BERT-base-uncased**（Encoder-Only / SequenceClassification）：全量微调、LoRA、AdaLora、P-Tuning v2 — GLUE MRPC 判别任务

## 运行环境与命令

- 命令约束：分析代码优先用 codegraph（`codegraph explore`）。**但它对本仓库覆盖不完整**：索引主要命中 `datasets/evaluate/` 里 vendored 的 HF 库代码，根目录的 `test_*.py` 常常搜不到（会返回库里的近似符号，看似有结果实则无关）。**查根目录实验脚本时直接用 Read/Grep 更可靠。**
- Conda 环境：`ai-gpu`（`/home/dupengair/shared/conda/anaconda3/envs/ai-gpu`）。关键版本：torch 2.8.0 / transformers 4.55.0 / peft 0.18.1 / datasets 4.5.0 / evaluate 0.4.6 / vllm 0.10.2 / bitsandbytes 0.50.2 / tensorboard 2.20.0
- **所有脚本必须从仓库根目录运行**，脚本内全部是相对路径（`./model/`、`./datasets/`、`./training/`）：
  ```bash
  python test_Lora_qwen3-0.6b.py
  ```
- 显卡为 RTX 4050 Laptop（6GB 显存）+ WSL2。脚本里 batch_size=1、gradient_accumulation_steps=4、adamw_torch_fused 等 OOM 规避参数是**刻意设置**，不要当 bug 改掉。精度**全仓库统一 `bf16=True`**，**不要改成 `fp16=True`**（与 `torch_dtype="auto"` 拿回的 bf16 权重冲突，第一步就崩，见坑 12）。
- **`gradient_checkpointing` 逐脚本不同，不要"统一"**（见坑 3）：**True** = `SFT_qwen3-0.6b` / `SFT_bert` / `QLora_qwen` / `AdaLora_qwen` / `AdaLora_bert` / `Lora_bert`；**必须 False（结构性要求）** = prompt 系四个（`Prefix_qwen` / `Prompt_qwen` / `PTuningV2_qwen` / `PTuningV2_bert`）；**`Lora_qwen` 当前也是 False**（LoRA 开 gc 本身是安全的，它配了 `use_reentrant=False`，只是没开；脚本里那行"✅ 梯度检查点"注释与值相反，别照着改回 True）。
- 所有模型/数据集加载均带 `local_files_only=True`（离线环境），不要改成联网加载。

## 脚本命名约定

`test_<方法>_<模型>.py` —— **`test_` 前缀不是 pytest 测试**，是自包含的实验/训练脚本（顶层顺序执行，无 main 函数、无函数封装）。运行单个"实验"就是 `python` 执行对应脚本。

| 脚本 | 内容 |
|---|---|
| `test_qwen3-0.6b.py` / `test_Bert.py` | 模型加载与结构探测（chat template、generate） |
| `test_transformers_qwen3-0.6b.py` / `test_transformers_bert.py` | 推理基础；Qwen 侧含 bitsandbytes 4bit NF4 量化 |
| `test_dataset_qwen3-0.6b.py` | zhihu-kol 数据加载/切分/分词/labels mask 流程 |
| `test_SFT_qwen3-0.6b.py` / `test_SFT_bert.py` | 全量微调（Trainer）；双模式：`has_model` 判 `model.safetensors`，存在则重载**完整权重**跳过训练（见坑 13） |
| `test_Lora_qwen3-0.6b.py` / `test_Lora_bert.py` | LoRA 微调（peft），先跑训练前基线评估 |
| `test_Lora-load_bert.py` | 从 `training/.../lora_adapter` 加载适配器推理 |
| `test_AdaLora_qwen3-0.6b.py` / `test_AdaLora_bert.py` | AdaLora：秩预算需回调驱动 `update_and_allocate`；加载必须 `is_trainable=True`（见坑 4） |
| `test_QLora_qwen3-0.6b.py` | QLoRA：4bit NF4 + `prepare_model_for_kbit_training` + paged 优化器。量化缺失与基线 Trainer 报错两层问题均已解决（2026-09-17，见 docs/QLoRA微调改造方案.md 第六/八节）：**peft 包装先于基线评估，基线用 `with model_lora.disable_adapter():` 拿纯 4bit base 口径**（纯量化模型不能直接构造 Trainer，见坑 19）。实测 4.9813 → 4.6469（-6.7%） |
| `test_Prefix_qwen3-0.6b.py` | Prefix Tuning **原版论文形态**（`prefix_projection=True`，每层 KV 注入，效果线） |
| `test_Prompt_qwen3-0.6b.py` | Prompt Tuning（TEXT 初始化虚拟 embedding，lr=5e-3） |
| `test_PTuningV2_qwen3-0.6b.py` | P-Tuning v2 论文形态（`prefix_projection=False`，机制对照线，小数据下崩坏属预期） |
| `test_PTuningV2_bert.py` | BERT×MRPC 的 P-Tuning v2（SEQ_CLS，`lr=1e-3` + 5 epoch 保 v2 纯度，含全 1 退化解拯救方案） |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务（`VLLM_USE_V1=0`，`build_app` 入参是 Namespace） |

## 目录约定

- `model/`：本地 HF 模型（`Qwen3-0.6B`、`bert-base-uncased`）
- `datasets/zhihu-kol/data/`：SFT 数据（INSTRUCTION/RESPONSE 字段，约 100 万条）；`cache/` 存分词后的 arrow 缓存
- `datasets/glue/`：GLUE 各子集本地数据（BERT 用 mrpc）；`cache-mrpc/` 为分词缓存
- `datasets/evaluate/`：**vendored 的 HF evaluate 库完整源码**（只为用 `./datasets/evaluate/metrics/glue/glue.py` 加载 MRPC 指标）。注意：CodeGraph 索引会命中这里的库代码而非根目录实验脚本
- `training/<模型>_<方法>/`：`output/`（checkpoint）、`logs/`（TensorBoard）、`lora_adapter/`（训练成果）。注意 `qwen3-0.6b_PTuning-V2` / `bert-base-uncased_PTuning-V2` 用连字符，其余方法下划线
- ⚠️ **SFT 线的 `lora_adapter/` 名不副实**：里面是**完整权重** `model.safetensors`（Qwen 1.19GB / BERT 438MB）+ config + tokenizer，不是 peft 适配器。目录名沿用未改，别被名字误导

## 两条实验线的核心差异（不要混用）

| | Qwen3-0.6B | BERT |
|---|---|---|
| 模型类 | `AutoModelForCausalLM` | `AutoModelForSequenceClassification`（不能用 CausalLM） |
| 任务/数据 | CAUSAL_LM，zhihu-kol 生成数据 | SEQ_CLS，GLUE mrpc（`num_labels=2`） |
| collator | **自定义 `custom_collate_fn`**（全 Qwen 线统一，**不要**用 `DataCollatorForLanguageModeling`，见坑 11） | `DataCollatorWithPadding` |
| LoRA target | `q/k/v/o/gate/up/down_proj` 全套 | `query/key/value` + **必须** `modules_to_save=["classifier"]` |
| prompt 系 | `PrefixTuningConfig`（映射见坑 2）；gc 必须 False（坑 3） | 同左；SEQ_CLS 可用，peft **自动**把 classifier 加入 modules_to_save，无需手动传 |
| 分词 | 手工拼 Qwen IM 模板，prompt 部分labels 置 -100 只对回答算 loss；`tokenizer.pad_token = eos_token` | 句对 pair 分词，max_length=128 |
| lr | LoRA/QLoRA 脚本**未显式设**→ 吃默认 `5e-5`；AdaLora `2e-4`；prompt 系 `1e-3~5e-3`（坑 5） | LoRA `5e-4`；SFT `2e-5` |
| 复用已训练成果 | peft 线判 `adapter_config.json` → `PeftModel.from_pretrained`；**SFT 线**判 `model.safetensors` → `from_pretrained` 整个模型（坑 13） | 同左 |

SFT 数据流程是固定管线：加载 → **`.shuffle(seed=42).select(range(100))` 采样** → `train_test_split` 两次切 80/10/10（seed=42）→ `map(tokenize_func, batched=True)` 逐 split 指定 `cache_file_name` → Trainer。新脚本通常复制现有脚本改配置，保持此结构。

> 采样口径、`tokenize_func`、`max_length` 三者要与既有脚本**逐字一致**——它们共同决定分词缓存能否共享，也决定跨线数字能否横比（Qwen 线 80/10/10 是各线的公共口径）。

## 关键坑（改代码前必读）

1. **分词缓存不自动失效**：`.map(cache_file_name=...)` 复用已有 arrow 缓存（`arrow_dataset.py` 里就是 `if os.path.exists(cache_file_name): return Dataset.from_file(...)`，**不校验指纹**）。修改 `tokenize_func` 或 max_length 后必须删除 `datasets/**/cache*/` 下对应 `.arrow` 文件，否则改动不生效。
   > ⚠️ **真正的危险方向是反的**：不是"改了不生效"，而是**函数或采样口径变了、缓存还在 → 静默加载旧数据，日志毫无异常**。改数据管线后先删缓存，或把 `cache_file_name` 换个名字。附带实测：`shuffle().select()` 会改变数据集指纹，`map` 检测到变化时通常会自动重算覆盖——但**不要依赖这个行为**。
2. **【勘误】prompt 系方法与 peft 的映射及可用性**：旧说法"`PrefixTuningConfig` 仅支持 Seq2Seq，不能用于 Qwen/BERT"在 peft 0.18.1 **已过时**——Qwen（`PeftModelForCausalLM.forward` 有 PREFIX_TUNING 分支）与 BERT（SEQ_CLS，transformers 4.55 重构版 `BertSelfAttention` 已有 KV 路径）**均已实跑验证可用**。正确映射：P-Tuning v2 = `PrefixTuningConfig(prefix_projection=False)`（默认）；原版 Prefix Tuning = `prefix_projection=True`；P-Tuning v1 = `PromptEncoderConfig`。注意 `encoder_hidden_size` 等 `encoder_*` 参数只在 `prefix_projection=True` 时有效/存在，v2 形态传了会 TypeError。小数据（百条级）× 小模型下只有 `projection=True` + 大 lr 才稳定（v2 纯形态实测 +119% 崩坏，系论文前提不满足，非 bug）。
3. **gradient_checkpointing 与 KV 注入结构性冲突**：gc 把每层 `past_key_value` 置 None → prefix/PTuningV2 的可训练参数脱离计算图。Qwen 侧表现为 loss 恒定的"假跑"（仅刷警告），BERT 侧**连警告都没有**（实测 prompt embedding 梯度静默变 None）。KV 注入类方法必须 `gradient_checkpointing=False`；Prompt Tuning / P-Tuning v1（输入拼接）不受此限。
4. **加载适配器的 `is_trainable` 三分规则**：prompt learning 系（Prefix/Prompt/PTuningV2）**禁止 True**（peft 源码直接 raise ValueError）；LoRA/QLoRA 不传（默认 False，评估够用）；AdaLora **必须 True**（否则 evaluate 前向访问 `trainable_adapter_name` 崩 AttributeError）。
5. **prompt/embedding 系的 lr 用 1e-3~5e-3**：1e-4 是 adapter 系直觉，会导致"学不动"——BERT 侧卡在"全预测多数类"退化解（`len(set(preds))==1` 是硬检测信号），Qwen 侧训练前后生成完全不变。冻结越彻底，需要的步长越大。
6. **双模式脚本的 `training_args`/`trainer` 必须顶格定义**（if/else 分支外）：加载模式（has_adapter=True）也要走 evaluate/predict；缩进在 else 内会导致二次运行 `NameError: trainer is not defined`（Lora_bert/PTuningV2_bert 均踩过）。
7. **`device_map="auto"` 的现状（原"与 Trainer 冲突"说法已不适用）**：**17 个脚本全部在用**，含两个 SFT 脚本，实测均可用，不要主动删。唯一要注意的是 **SFT 复用模式**：重载后在 CPU，由 Trainer 构造时自动迁移，**不要手动 `.to("cuda")`**（见坑 13）。
8. **生成评估的三个实际约定**（现有脚本均未设 `tokenizer.padding_side`）：① 各脚本都是**单条 prompt 逐条生成**，没有 padding 才不会踩"右侧 padding 导致续写从 pad 后开始"的坑——**若要改成批量生成，必须先 `tokenizer.padding_side = "left"`**；② PEFT 模型生成前必须 `model.eval()`（Trainer 会把模型切回 train 模式，`lora_dropout=0.1` 会让 greedy 也不可复现）；③ 提示只拼**指令不含答案**、`do_sample=False`、切掉提示只解码新生成部分。
9. **LoRA 保存**：`trainer.save_model()` 只存适配器不存主干；加载端（`test_Lora-load_bert.py`）通过 `PeftConfig` 读回 base model 路径，`num_labels` 必须与训练一致。
10. `eval_strategy`（新版）不要写成 `evaluation_strategy`。
11. **Qwen 线的 collator 必须是自定义的 `custom_collate_fn`**：`DataCollatorForLanguageModeling` 在 instruction 微调上有**三个坑**——① 不 pad `labels`（eval 批量>1 直接崩）；② `mlm=False` 分支用 `input_ids` **无条件覆盖**自定义 labels，让 "prompt 置 -100、只对回答算 loss" 静默失效；③ `pad_token == eos == <|im_end|>` 时把停止符监督抹成 -100，模型永远学不会收尾。**换 collator 会改变 eval_loss 口径——前后对比必须用同一套代码重跑。**（脚本里那行直接用 `DataCollatorForLanguageModeling` 的赋值是死代码，后面会被 `data_collator = custom_collate_fn` 覆盖，但别误删覆盖行）
12. **精度匹配律：bf16 权重只能配 `bf16=True`**。`torch_dtype="auto"` 拿回 bf16 权重时再开 `fp16=True`（启用 GradScaler），会在**第一步 `clip_grad_norm_`** 崩 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`（bf16 本就不需要 loss scaling）。**traceback 里同时出现 `grad_scaler.py` 和 `BFloat16` 即可秒判。**
13. **SFT 的"复用已训练成果"与 peft 线是两套机制**：peft 线存/载"base + 增量"，走 `PeftModel.from_pretrained`；**SFT 全量微调覆盖了权重**，只能整模型 `from_pretrained` 重载。因此 ① 判断文件是 **`model.safetensors`** 不是 `adapter_config.json`；② **基线评估必须在重载之前完成**（无 `disable_adapter()` 可用，脚本里基线块在前是设计使然，别调整顺序）；③ 重载前必须 **`del trainer_baseline, model`** 释放旧引用，否则 6GB 卡上同时存在两份 0.6B 权重；④ 保存守卫/训练守卫用的是 `has_model`，改名时四处路径要同步。
14. **AdaLora 两个专属硬约束**：① `total_step` **必须显式填且等于真实训练总步数**（`(样本数/有效批量)×epochs`），改 `num_train_epochs` 要同步重算，否则 peft 直接 `ValueError`；② **`load_best_model_at_end=True` 与之不兼容**——秩 `rank_pattern` 随训练演化，回滚到中途 epoch 的 checkpoint 会 `size mismatch` 崩溃（`test_AdaLora_qwen3-0.6b.py` 已关；BERT 线若要最优 epoch 得用干净的 base 重新 `from_pretrained(..., is_trainable=True)`）。
15. **ROUGE 对中文无效，不要据此下结论**：vendored `rouge-score` 的分词器只认 `[a-z0-9]`，纯中文返回空列表（全 0、不报错）；"逐字加空格伪装英文词"的老 hack **对汉字同样无效**。此前各线的"ROUGE 提升"是拉丁乱码与参考的碰巧重合。**效果主指标用 `eval_loss`（NLL 不经过分词）**；非要用 ROUGE 就传字符级 tokenizer（`tokenizer=lambda s: list(s)`），且只作辅助参考。
16. **采样后全程只有几十~两百步，`steps` 类阈值会集体失效**：`eval_steps`/`save_steps` 按旧全量口径设的值可能一次都不触发（表现为**训练全程零 eval、零 checkpoint、TensorBoard 空白**），`logging_steps` 默认 500 更是一条 train_loss 都不记。**用 `eval_strategy="epoch"` + `save_strategy="epoch"` + 显式 `logging_steps=10`。**
17. **遗留的注释错误（改动这两个脚本时顺手清理）**：`test_Lora_bert.py:142-144` 与 `test_PTuningV2_bert.py:143-145` 的加载分支里，留着从 AdaLora 复制来的"★ `is_trainable=True` 必须…"注释，但两处代码都已不传该参数。**对 `test_PTuningV2_bert.py` 而言，照注释加回去会直接 `ValueError` 崩**（prompt learning 禁止 True，见坑 4）。
18. 代码注释和文档均为中文，新增注释保持中文。**注释要随代码同步改**——本仓库最贵的几个坑（坑 3、坑 7、坑 17）都源于"注释与代码相反"，对学习型仓库来说，错误注释比错误代码危害更大。
19. **纯量化模型不能直接构造 Trainer（QLoRA 专属，已踩实）**：`Trainer.__init__` 拦截"is_quantized + 未挂 adapter"的模型并 raise `ValueError: purely quantized models`，**不看 `do_train=False`**（只想做基线评估也被拦）。因此 QLoRA 线的 peft 包装必须**先于**基线评估，基线在 `with model_lora.disable_adapter():` 内 evaluate/generate（= 纯 4bit base 口径）。附带认知：这个报错出现说明量化已生效——它是"量化修复成功"的信号而非倒退。修复三步见 docs/QLoRA微调改造方案.md 第八节。
