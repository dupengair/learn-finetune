# Prompt Tuning 实现检查报告（test_Prompt_qwen3-0.6b.py）

> 检查对象：`test_Prompt_qwen3-0.6b.py`（由 `test_Prefix_qwen3-0.6b.py` 复制改造）
> 检查方式：① 通读 peft 0.18.1 源码（conda 环境 `ai-gpu` 的 site-packages）；② CPU 上实际运行 Prompt Tuning 的 forward/generate 链路做运行时验证（hook 抓取 peft 传给 base model 的真实张量形状）。
> 结论先行：**实现整体正确，没有会导致跑不起来或结果错误的 bug**。有 1 个强烈建议修改的超参、2 个必须理解的机制行为、2 个可选的小清理。

---

## 一、核对结论总览

| 检查项 | 结论 |
|---|---|
| `PromptTuningConfig` + `TaskType.CAUSAL_LM` 组合 | ✅ peft 0.18.1 官方支持 |
| labels 长度与拼接后序列对齐 | ✅ peft 自动处理，无需手动 pad |
| attention_mask 对齐 | ✅ peft 自动处理 |
| TEXT 初始化 + `tokenizer_name_or_path` | ✅ 必填项校验通过，本地路径不联网 |
| generate 路径（虚拟 token 注入） | ✅ peft 自动处理 KV cache 错位 |
| 加载/保存（`has_adapter` 分支） | ✅ 与 Prefix 版机制一致 |
| 数据管线 / collator / labels mask | ✅ 与 Prefix 版一致，合法复用既有 arrow 缓存 |
| `learning_rate=1e-4` | ⚠️ **对 Prompt Tuning 偏小，强烈建议调大**（见问题 1） |
| `prompt_tuning_init_text` 长度 | ⚠️ 会被循环复制，行为需理解（见机制点 A） |
| 目录名 `lora_adapter` / 变量名 `lora_save_path` | 💡 命名遗留，可选清理（见问题 3） |
| generate 时 `Position ids are not supported` 告警 | ✅ peft 防御性提示，数值无害，可选抑制（见第六节） |

---

## 二、peft 0.18.1 自动处理的事（写明了避免你手动去改错）

Prompt Tuning 的机制是把 `num_virtual_tokens` 个可学习的 embedding **拼在输入 embedding 前面**，序列长度从 `L` 变成 `L+16`。直觉上你会担心两个对齐问题，但源码确认 peft 在 `PeftModelForCausalLM.forward`（`peft/peft_model.py`）里全部自动处理了：

**① labels 自动前缀 pad 16 个 -100**

```python
# peft/peft_model.py  PeftModelForCausalLM.forward
if labels is not None:
    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
```

虚拟 prompt 位置不参与 loss，你的 `user_len` mask 逻辑完全不受影响。

**② attention_mask 自动前缀拼 16 个 1**

```python
if attention_mask is not None:
    prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens)...
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
```

**③ generate 时自动注入**：prefill 阶段 peft 重写的 `prepare_inputs_for_generation` 会把 prompt embedding 拼进 `inputs_embeds`（`input_ids` 置 None），自回归阶段自动截取最后一个 token，attention_mask 同步扩展。你的 `generate_answer()` 不需要任何修改。

> ⚠️ 注意：这三件事是 **peft ≥ 某版本**才有的行为。老教程里「Prompt Tuning 要手动 pad labels」的说法在本环境（0.18.1）不适用——**不要照抄老教程再加手动 pad，会 pad 两遍**。

### 运行时验证结果（CPU 实测，非纸面推断）

输入 5 个 token（其中前 2 个 label 为 -100），hook 抓取 peft 实际传给 Qwen3ForCausalLM 的张量：

```
[原始输入长度] 5
[peft传给base] inputs_embeds_len=21  labels_len=21  attention_mask_len=21   # 5+16，三者同步
[labels前16位] [-100 × 16]                                                    # 自动 pad
[attention_mask前16位] [1.0 × 16]                                             # 自动拼接
[loss] 5.8147                                                                 # 数值正常
trainable params: 16,384 || all params: 596,066,304 || trainable%: 0.0027
[generate] 正常
```

---

## 三、发现的问题与修改方案

### 问题 1（重要）：`learning_rate=1e-4` 对 Prompt Tuning 几乎学不动

**位置**：`training_args` 第 255 行

```python
learning_rate=1e-4,           # ② ★ 显式设置！默认5e-5对adapter训练太低（本节根因1）
```

**原因**：这条注释和数值是从 LoRA/Prefix 的经验继承来的，但 Prompt Tuning 的可训练参数形态完全不同：

| | Prefix Tuning | Prompt Tuning |
|---|---|---|
| 可训练参数 | 每层 KV 各一份重写矩阵，约 52 万+ | **只有 16,384 个**（16×1024，实测 0.0027%）|
| 参数形态 | 分布在全部 28 层的 past_key_values | 一整个 embedding 表，集中在输入端 |
| 论文/常用 lr | 1e-4 ~ 1e-3 | 论文（Lester et al. 2021）用 **0.1~1**；小模型常见 1e-3 ~ 1e-2 |

1e-4 配合 warmup_ratio=0.1、10 epochs（共约 200 步）去训 1.6 万个参数，大概率出现「loss 纹丝不动、微调前后生成完全一样」的假训练现象——这比报错更隐蔽。

**修改方案**（二选一，先小后大试）：

```python
learning_rate=5e-4,   # 保守起点；若 loss 与生成内容仍无变化，逐步加到 2e-3 / 5e-3
```

观察标准：训练分支跑完后看 `eval_loss` 是否明显下降（相对基线几个百分点以上）、微调后生成是否出现风格变化。若 5e-3 仍无变化再回来找我分析。

### 机制点 A（必须理解，不一定要改）：TEXT 初始化会「循环复制」填满 16 个虚拟 token

**位置**：`peft_config` 第 234-236 行

```python
prompt_tuning_init="TEXT",
prompt_tuning_init_text="下面是任务描述：",    # 用一段文本embedding初始化虚拟token
```

**行为**：这句话被 Qwen tokenizer 分成 8 个 token，而 `num_virtual_tokens=16`。peft 的 `PromptEmbedding` 会把这 8 个 token 的 embedding **复制 2 遍平铺**填满 16 个位置（已实测 `emb[:8] == emb[8:16]` 为 True）。

**影响**：不影响正确性，但意味着 16 个虚拟 token 里有一半是重复初始值。如果你希望每个虚拟 token 都由不同文字初始化，把 init_text 换成长度 ≥16 个 token 的任务描述，例如：

```python
prompt_tuning_init_text="你是一个知识渊博的知乎答主，请用中文认真、详细地回答用户提出的问题",  # >16 token，会被截到16
```

（超过 16 会截断，只取前 16 个 token 的 embedding。）

### 机制点 B（学习向，无需改代码）：Prompt Tuning 与 Prefix Tuning 的注入位置差异

这是两个实验最核心的对比点，也解释了两个脚本里 `gradient_checkpointing=False` 注释的差异：

| | Prefix Tuning | Prompt Tuning |
|---|---|---|
| 注入位置 | **每层** Transformer 的 KV cache（`past_key_values`） | **只拼在 embedding 层输出**（输入级） |
| 序列长度 | 不变（KV 变长） | 变长 +16 |
| 对 gradient_checkpointing 敏感 | 是（gc 把 past_key_values 置 None → 假跑） | 否（理论上可开 gc 省显存） |
| RoPE 位置 | 真实 token 从 0 开始 | 真实 token 从 16 开始（peft 置空 position_ids，模型自动重排） |

你的脚本对 Prompt Tuning 保留 `gradient_checkpointing=False` 是安全的（只是没占到显存便宜），**不算错误，可以不动**；若以后显存吃紧想开，Prompt Tuning 是三个 prompt-learning 方法里唯一能安全开的。

### 问题 2（边缘情况提示）：eval 中可能出现「全 -100 → loss=NaN」

**触发条件**：`max_length=1024` 截断时，如果某条样本的 user 部分（INSTRUCTION + IM 模板）本身就超过 1024 token，回答部分会被完全截掉 → labels 全为 -100 → `CrossEntropyLoss` 返回 NaN（本次运行时验证意外复现了这个现象：5 个 token 全 mask 后 loss=NaN）。

**影响**：zhihu-kol 数据极少触发，Prefix 实验已跑通也间接说明未触发；但知道这个机制有助于排查「某次 eval_loss 突然变成 nan」的问题。遇到时只需过滤超长样本或在 collator 里跳过全 -100 的样本，现在不用改。

### 问题 3（可选清理）：命名遗留

- `lora_save_path`（第 217 行）→ 实际存的是 prompt embedding（16×1024 fp32 张量，约 64KB），建议改名 `prompt_save_path`；目录 `training/qwen3-0.6b_Prompt/lora_adapter` → `training/qwen3-0.6b_Prompt/prompt_adapter`（**改目录名时第 219 行 `has_adapter` 的路径要同步改**，否则会误判为「未训练」重新训练一遍）。
- 第 256 行注释「AdaLora论文亦用warmup」是 Prefix 脚本复制来的，与本实验无关，可顺手删。

### 问题 4（文档勘误，不需要改代码）：关于 CLAUDE.md 坑 2

CLAUDE.md 关键坑 2 写「`PrefixTuningConfig` 仅支持 Seq2Seq（T5/BART），不能直接用于 Qwen（CausalLM）」——这在 peft 0.18.1 中**已过时**：`PeftModelForCausalLM.forward` 明确实现了 `PREFIX_TUNING` 分支（经 past_key_values 注入），且你的 Prefix 实验已在 Qwen3-0.6B 上跑通，佐证了这一点。同理 `PromptTuningConfig + CAUSAL_LM` 也是官方支持组合（本次已源码+运行时双重验证）。该坑仅对老版本 peft 成立。

---

## 四、验证过没有问题的部分（不用改）

1. `peft_type="PROMPT_TUNING"` 传字符串 OK——`PromptTuningConfig.__post_init__` 会强制覆盖为 `PeftType.PROMPT_TUNING`。
2. `inference_mode=False` 显式传 OK（默认值即 False；保存时 peft 自动改为 True 写入 adapter_config.json）。
3. TEXT 初始化的必填校验（`tokenizer_name_or_path`、`prompt_tuning_init_text` 缺失会直接 ValueError）你都已提供；`./model/Qwen3-0.6B` 是本地路径，`PromptEmbedding` 内部再开一个 tokenizer 也不会联网。
4. prompt 参数是 **fp32**（peft 刻意为之，`autocast_adapter_dtype` 机制，训练更稳），forward 时自动 `.to(inputs_embeds.dtype)` 转 bf16 参与计算——bf16 模型 + fp32 优化器状态是预期设计，不是 bug。
5. `model_trained.config.use_cache = False` 只影响训练 forward；`generate` 走独立的 `generation_config`（use_cache=true），配合 peft 的 KV 错位处理，生成正常。
6. `has_adapter` 用 `adapter_config.json` 判断保存完整性、`PeftModel.from_pretrained` 加载路径，对 prompt-learning 方法同样适用（保存的是 `prompt_embeddings` 张量，加载时重建）。
7. 分词缓存放心复用：`tokenize_func` 与 Prefix 版逐字相同，`cache/train.arrow` 等缓存合法复用；仅当你以后改 `tokenize_func` 或 `max_length` 时才需要删缓存（CLAUDE.md 坑 1）。
8. `generate_answer` 里 temperature/top_p/top_k 传参在 greedy 下无效、仍会打一行提示（`The following generation flags are not valid...`），无害，Prefix 版同样存在。

---

## 五、一句话行动建议

改一行：`learning_rate` 1e-4 → **5e-4**（不够再上 2e-3）；可选把 `prompt_tuning_init_text` 换成一句 ≥16 token 的任务描述。其余保持现状直接跑。

---

## 六、补充分析：generate 时的 `Position ids are not supported` 告警

> 背景：实跑结果 eval_loss 4.8659 → 4.3118（-11.4%）、ROUGE-L 0.0174 → 0.0586，训练已生效。运行日志在「微调后生成」阶段出现：
> `peft_model.py:2141: UserWarning: Position ids are not supported for parameter efficient tuning. Ignoring position ids.`

### 1. 告警从哪来：两处同文案警告，只有 generate 这处会触发

peft 0.18.1 的 `peft_model.py` 里有 8 处相同文案（分属各 PeftModel 类），与本实验相关的只有两处：

| 行号 | 所在方法 | 触发条件 | 本实验是否触发 |
|---|---|---|---|
| 1941 | `PeftModelForCausalLM.forward`（prompt-learning 分支） | forward 的 kwargs 里带 position_ids | ❌ Trainer 的 collator 不传 position_ids，训练/eval 静默通过 |
| 2141 | `PeftModelForCausalLM.prepare_inputs_for_generation` | PeftModel.generate 的每一步 | ✅ 就是日志里这条 |

### 2. 完整触发链路（为什么 generate 会「冒出」position_ids）

你并没有传 position_ids，它是 **transformers 自己生成的**。一次 generate 的 prefill 步：

1. `generate_answer(model_trained, ...)` → `PeftModelForCausalLM.generate` 把 `prepare_inputs_for_generation` 换成 peft 版本，再调 `Qwen3ForCausalLM.generate`；
2. peft 版本第一行先调用**原生** Qwen3 的 prepare：transformers 4.55 在这一步自动算好 `position_ids = cache_position.unsqueeze(0)`，长度按**原始 input_ids（L）**算——此时它不知道 peft 马上要在前面拼 16 个虚拟 token；
3. peft 拼完 prompt 后 `inputs_embeds` 长度变成 L+16，检查 model_kwargs 发现带着**长度 L** 的 position_ids → 形状对不上 → **警告 + `model_kwargs["position_ids"] = None`**；
4. `Qwen3Model.forward` 收到 `position_ids=None` 后按实际长度 L+16 自动重建位置编码：虚拟 token 占位置 0–15，真实 token 从 16 开始。

### 3. 有害吗：零，它恰恰是正确性保障的一部分

- 若 peft 不丢弃：长度 L 的 position_ids 配长度 L+16 的 inputs_embeds，RoPE 位置整体错位（甚至形状报错）；
- 丢弃后模型自动重建的结果，与**训练时 forward 完全一致**（训练时 Trainer 不传 position_ids，peft 同样置空让模型按拼接后长度自动生成，见第二节）；
- 训练与推理的位置语义一致 → 数值结果正确。这条警告只是 peft 提醒你「外部 position_ids 被忽略」，属防御性提示，唯一影响是日志噪音。

顺带解释日志里两个现象：

- **为什么基线生成没有这条告警**：`baseline_generations` 用的是未包装的原始 model，不经过 peft 拦截；只有微调后的 `generate_answer(model_trained, ...)` 走 peft 路径。
- **为什么 5 条样本只打印一次**：Python 的 `warnings` 对「同一位置 + 同一消息」默认去重，实际每步都触发，只是只显示一次。

### 4. 修复方案：正确性上无需修，想清爽日志可精准抑制

**不要**试图「传对 position_ids」来消除——它由原生 prepare 在内部生成，调用侧控制不了，这是 peft 与 transformers generate 协作的固有交互，不是脚本写错。

在脚本头部（`import torch, os` 附近）加一行**只针对这条消息**的过滤（前缀正则匹配，不会误杀其他警告）：

```python
import warnings
warnings.filterwarnings("ignore", message="Position ids are not supported for parameter efficient tuning")
```

不推荐：全局 `warnings.filterwarnings("ignore")`（会把其他有用警告一起吞掉）；手改 peft 源码（下次升级即丢）。

### 5. 顺带：本次结果解读

- eval_loss 相对基线下降 11.4%，且第二个 eval（微调后）比第一个 eval（基线）快得多（17.58 it/s vs 4.45 it/s）属正常——两次 eval 的模型和序列长度本质相同，速度差异主要来自首测时的初始化/编译开销，不影响数值对比；
- ROUGE-L 提升 3 倍以上，说明虚拟 prompt 确实在起作用（该指标是字符级粗指标，看相对变化即可）；
- 建议打开 TensorBoard（`training/qwen3-0.6b_Prompt/logs`）确认 eval_loss 曲线在 10 个 epoch 上单调下降、无回升，即无过拟合迹象。

