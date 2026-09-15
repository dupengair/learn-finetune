# PTuningV2 实现检查报告（test_PTuningV2_qwen3-0.6b.py）

> 检查对象：`test_PTuningV2_qwen3-0.6b.py`（由 `test_Prompt_qwen3-0.6b.py` 复制改造）
> 检查方式：① 通读 peft 0.18.1 的 `p_tuning/config.py`、`p_tuning/model.py`、`peft_model.py` 源码；② CPU 实际运行验证（peft_type 覆盖行为、gc 开关下梯度流动、假跑对照实验）。
> 结论先行：**脚本配置类用错了一代——`PromptEncoderConfig` 是 P-Tuning v1，不是 v2**；真正的 P-Tuning v2 在 peft 里就是 `PrefixTuningConfig`，即仓库已有的 `test_Prefix_qwen3-0.6b.py`。技术实现本身（v1 意义下）能正常训练、无假跑，另有 3 处注释/参数小错。
>
> **更新（用户已确认目标为 v2）**：正式修改方案见 **第五节**——把配置类换成 `PrefixTuningConfig` 并关闭 gradient_checkpointing，即 v2 的正确 peft 用法。

---

## 一、核心发现：v1 / v2 概念错位（先决策，再改代码）

| | P-Tuning v1 | P-Tuning v2 |
|---|---|---|
| 论文 | GPT Understands, Too (Liu et al. 2021) | P-Tuning v2 (Liu et al. 2022) |
| 核心机制 | **浅层**：LSTM/MLP 重编码 soft prompt，只拼在输入 embedding 层 | **深层**：每层 Transformer 注入可学习 past_key_values，去掉重编码器 |
| peft 配置类 | `PromptEncoderConfig`（内部类型 `P_TUNING`） | **`PrefixTuningConfig`**（内部类型 `PREFIX_TUNING`） |
| 仓库对应脚本 | ← 你这个脚本（实际跑的） | `test_Prefix_qwen3-0.6b.py`（已做过） |

**这意味着**：P-Tuning v2 实验你已经做过了（Prefix 脚本）。两条路二选一：

- **路线 A（推荐）**：把这个脚本**正名为 P-Tuning v1** 实验——保留 `PromptEncoderConfig`，改目录名/注释/打印文案。价值：与 Prefix(v2)、PromptTuning 形成完整的三代 prompt 系对照实验（v1 重编码浅层 / v2 深层无编码器 / v3 纯 embedding），这正是这类学习仓库最该有的横向对比。
- **路线 B**：坚持做 v2 → 直接复用 `test_Prefix_qwen3-0.6b.py`，本脚本没有独立价值（换成 PrefixTuningConfig 后与 Prefix 脚本只差目录名）。

下面的修改方案按路线 A 给出（问题 3 同时覆盖路线 B 的坑）。

---

## 二、问题清单与修改方案

### 问题 1（必改）：`peft_type="PROMPT_TUNING"` 是无效参数，会被静默覆盖

**位置**：`peft_config` 第 231 行

```python
peft_config = PromptEncoderConfig(
    peft_type="PROMPT_TUNING",    # LoRA/AdaLora/PrefixTuning/PromptTuning   ← 错
```

**原因**：`PromptEncoderConfig.__post_init__` 末尾强制执行 `self.peft_type = PeftType.P_TUNING`，你传什么都一样（CPU 实测：传入 `"PROMPT_TUNING"`，读回 `PeftType.P_TUNING`）。不报错、能跑，但注释会误导后续阅读者以为能通过这个参数切换算法。

**修改方案**（路线 A）：

```python
peft_config = PromptEncoderConfig(
    peft_type="P_TUNING",         # P-Tuning v1：PromptEncoder(LSTM/MLP重编码)；v2请用PrefixTuningConfig
    task_type=TaskType.CAUSAL_LM,
    ...
```

### 问题 2（必改）：`gradient_checkpointing=True` 与它自己的注释矛盾

**位置**：第 263-264 行

```python
gradient_checkpointing=True,         # ★ Prefix Tuning 必须关！gc 会把 past_key_value 置 None，
                                     #   prefix 参数被移出计算图 → 训练假跑（见文档第二节
```

**原因**：注释是从 Prefix 脚本复制来的，说的是**真 v2 (PrefixTuning)** 的坑，但值却在 Prompt 脚本（False）基础上改成了 True。两种情况的行为完全不同（均 CPU 实测）：

| 组合 | 实测结果 |
|---|---|
| `PromptEncoderConfig`(v1) + gc=True（你当前的组合） | ✅ 正常：loss=4.998，**7/7 可训练参数梯度全部非零**，无假跑。因为 v1 的 prompt 在输入端拼接，不依赖 past_key_values，梯度经拼接点自然回传 |
| `PrefixTuningConfig`(真 v2) + gc=True（若走路线 B 却不改 gc） | ❌ **第一步 backward 直接崩**：`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` |

**修改方案**：值可以保留 `True`（v1 开 gc 省显存，正确利用了 6G 卡条件），但注释必须改成事实：

```python
    gradient_checkpointing=True,         # ✅ v1(prompt拼输入端)可安全开gc；⚠️若换PrefixTuningConfig(v2)必须改False，
                                         #   否则backward直接RuntimeError（v2依赖past_key_values会被gc丢弃）
```

### 问题 3（建议改）：`learning_rate=8e-5` 对 v1 偏保守

**原因**：v1 的可训练参数 = prompt embedding + LSTM/MLP 编码器，实测 **296,192 个（0.0497%）**——是纯 PromptTuning(16,384) 的 18 倍，但仍是冻结全模型的极少参数。8e-5 大概率能学（v1 参数形态比 v3 健康），但参考仓库 AdaLora 用 2e-4、PromptEncoder 原论文量级，建议：

```python
learning_rate=2e-4,           # ② v1含编码器网络，8e-5偏保守；loss不动再上5e-4
```

先按 8e-5 跑也行，判据同 PromptTuning 实验：eval_loss 不降、生成无变化 → 升 lr。

### 问题 4（建议改，路线 A）：命名与文案正名

脚本通篇自称 P-TuningV2，若走路线 A 应统一正名为 v1：

| 位置 | 现状 | 建议 |
|---|---|---|
| 第 166/245/246 行目录 | `qwen3-0.6b_PTuning-V2/` | `qwen3-0.6b_PTuning/`（且注意仓库风格是下划线 `_`，不是连字符 `-`） |
| 第 217 行保存路径 | `.../PTuning-V2/lora_adapter` | `.../PTuning/lora_adapter`（**改目录名时 219 行 `has_adapter` 判断会跟着路径变，旧目录不要混用**） |
| 第 286/293/306/308 行打印 | "P-TuningV2 微调" 等 | "P-Tuning v1 微调"（避免以后回看日志时被误导） |
| `lora_save_path` 变量名 | LoRA 遗留命名 | 同前两份报告：可选改 `adapter_save_path`，不重要 |

### 已验证无需修改的部分

1. **TensorBoard 配置 ✅**：你从改后的 Prompt 脚本复制，`report_to="tensorboard"` + `logging_steps=10` 都带上了，且 `logging_dir` 已指向 `PTuning-V2/logs`（若改目录名记得三处同步）；
2. **数据管线 ✅**：`tokenize_func` 与 Prompt 脚本逐字相同，合法复用 `datasets/zhihu-kol/cache/*.arrow` 缓存；
3. **CausalLM 兼容 ✅**：`P_TUNING` 与 `PROMPT_TUNING` 走 `PeftModelForCausalLM.forward` 同一分支（拼输入端），labels 自动 pad 16 个 -100、attention_mask 自动扩展、generate 自动注入——上一份 Prompt 报告第二节的机制说明全部适用；
4. **`PromptEncoderConfig` 缺省参数 ✅**：`encoder_reparameterization_type` 未传，默认 `MLP`（v1 论文用 LSTM，peft 默认 MLP 也可，想贴论文可显式传 `"LSTM"`）；`encoder_num_layers=2`、`encoder_dropout=0.1` 合理；
5. **保存/加载链路 ✅**（学习点）：prompt 系方法保存的都是**编码后的最终 embedding**（`get_prompt_embedding_to_save`，key 为 `prompt_embeddings`）；`PromptEncoder.forward` 在 `inference_mode=True` 时**直接返回 embedding、跳过 LSTM/MLP**——所以加载端（`PeftModel.from_pretrained`，is_trainable=False → inference_mode=True）推理结果与训练时一致，编码器权重不保存也不影响推理。这是 peft 的刻意设计。

---

## 三、验证记录（CPU 运行时实测）

```
[验证1] 传入 peft_type='PROMPT_TUNING'，实际 peft_type = PeftType.P_TUNING
trainable params: 296,192 || all params: 596,346,112 || trainable%: 0.0497
[验证2] P_TUNING + gc=True: loss=4.9979, embedding.grad均值=8.074e-03,
        有非零梯度的可训练参数 7/7          ← 当前脚本组合可正常训练
[验证2] generate 正常: shape=(1, 17)
[验证3] PREFIX_TUNING(v2) + gc=True(包装后开): loss.backward() →
        RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
                                             ← 若走路线B必须 gc=False 的实证
```

---

## 四、一句话行动建议

先定方向：**想学 v2 → 直接看 `test_Prefix_qwen3-0.6b.py`（它就是 v2）；想保留本脚本 → 按路线 A 正名为 P-Tuning v1**，改 3 处：`peft_type="P_TUNING"`、gc 注释改为「v1 可开」、目录/文案正名；lr 可从 8e-5 提到 2e-4。

---

## 五、正式修改方案：让本脚本成为真正的 P-Tuning v2（已定方向）

### 5.1 先解开「三个方法是不是三个独立实现」的疑惑

三个方法的**机制差异**（你列的表格）完全正确，问题出在「论文名 → peft 配置类」的映射关系上：

| 论文方法 | 核心机制 | peft 配置类 | 对应关系 |
|---|---|---|---|
| Prefix Tuning (Li & Liang 2021) | 每层注入可学习 KV 前缀 **+ MLP 重参数化** | `PrefixTuningConfig(prefix_projection=True)` | 机制提出者 |
| P-Tuning v1 (Liu et al. 2021) | 输入层伪 token + **LSTM/MLP 重编码** | `PromptEncoderConfig`（内部 `P_TUNING`） | ← 你脚本现在跑的 |
| Prompt Tuning (Lester et al. 2021) | 输入层纯 embedding，无编码器 | `PromptTuningConfig` | ← 上一个实验 |
| **P-Tuning v2 (Liu et al. 2022)** | **每层注入可学习前缀，去掉重参数化** | **`PrefixTuningConfig(prefix_projection=False)`** | ← 你要做的 |

**为什么 v2 和 Prefix Tuning 共用同一个 peft 配置类**——不是 peft 偷懒，是 v2 论文自己声明的（[arXiv:2110.07602](https://arxiv.org/abs/2110.07602) 原文）：

> "Our method P-Tuning v2 is an implementation of **Deep Prompt Tuning**（Li & Liang 2021，即 Prefix Tuning 论文）**optimized and adapted for NLU**."

即 v2 = Prefix Tuning 提出的「每层注入」机制 + 两点改造（**去掉重参数化 MLP**、**面向分类任务去掉 verbalizer**）。你表格里写的 v2「每一层都加入可学习前缀（类似 Prefix Tuning）」就是这个意思。peft 只实现了一套「每层注入」的代码，配置类名叫 `PrefixTuningConfig`，用 `prefix_projection` 开关区分：

- `prefix_projection=False`（peft 默认）→ **P-Tuning v2** 的设计（去重参数化）；
- `prefix_projection=True` → 原版 **Prefix Tuning** 论文的设计（带 MLP 重参数化）。

所以「在 peft 里做 P-Tuning v2」= 用 `PrefixTuningConfig`（默认参数即 v2 形态）。这与你已有 Prefix 实验在 peft 配置层面等价——**不冲突**：仓库叙事上 Prefix 脚本可以视作「机制演示」，本脚本按 v2 正名跑 Qwen3 生成任务，见 5.3 的差异化建议。

### 5.2 逐行修改方案（共 4 处，按顺序改）

**① 第 1-6 行 import：换配置类**

```python
from peft import (
    PrefixTuningConfig,     # ← 原 PromptEncoderConfig（那是 P-Tuning v1）
    TaskType,
    get_peft_model,
    PeftModel
)
```

**② 第 230-238 行 peft_config：整块替换**

```python
    peft_config = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",    # P-Tuning v2 = Deep Prompt Tuning（v2论文：Prefix Tuning机制的NLU适配版）
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=16,        # 每层注入的可学习前缀长度
        prefix_projection=False,      # ★ v2关键设计：去掉重参数化MLP；设True则回到原版Prefix Tuning
        inference_mode=False
    )
```

⚠️ **必须删掉这三个参数**：`encoder_hidden_size=128`、`encoder_num_layers=2`、`encoder_dropout=0.1`——`PrefixTuningConfig` 是 dataclass，不接受未定义字段，带着它们构造会直接 `TypeError`（`encoder_dropout`/`encoder_num_layers` 根本不存在；`encoder_hidden_size` 虽存在但只在 `prefix_projection=True` 时才需要，v2 用不到）。

**③ 第 263 行 gradient_checkpointing：True → False（不改必崩）**

```python
    gradient_checkpointing=False,        # ★ v2必须关！gc把past_key_values置None → backward直接RuntimeError
                                         #   （CPU实测：v2+gc第一步backward即报
                                         #    "element 0 of tensors does not require grad"）
```

这是 v1/v2 的本质差异决定的：v1 的 prompt 拼在输入 embedding 端，梯度经拼接点自然回传（实测 v1+gc 七个参数梯度全非零）；v2 的 prompt 以 past_key_values 形式进每层 Attention，gc 重算时 past_key_values 被丢弃，可训练参数整个脱离计算图。你的 Prefix 实验注释里写的「训练假跑」在当前 transformers 4.55 + peft 0.18.1 组合下实测更严重——是直接报错崩溃，不是静默假跑。

**④ 第 256 行 learning_rate：8e-5 → 1e-4**

```python
    learning_rate=1e-4,           # ② v2与Prefix同机制同数据，沿用Prefix实验已验证的1e-4
```

v2 可训练参数与 Prefix 实验同量级（每层 KV：约 52 万，远大于 v1 的 29.6 万），1e-4 是已验证可用的起点。

### 5.3 与已有 Prefix 实验的关系（重要，避免「白做一遍」的错觉）

peft 配置层面，本脚本改完后与 `test_Prefix_qwen3-0.6b.py` **等价**（同为 `PrefixTuningConfig` + 16 虚拟 token + projection=False），跑出来的曲线也应当接近（仅随机性差异）。三个建议任选：

1. **保留两个脚本但明确定位**（推荐）：Prefix 脚本注释标「机制演示」；本脚本标「P-Tuning v2：Qwen3 生成任务」，文档里互相注明 peft 实现等价、叙事不同；
2. **做出真对照**：把 **Prefix 脚本改设 `prefix_projection=True`**（原版 Prefix Tuning 论文形态：每层前缀 + MLP 重参数化），本脚本保持 False（v2 形态：无重参数化）→ 两个脚本变成「重参数化有无」的严格对照实验，学习价值最高；
3. **换载体**：你表格里也写了「BERT 做 P-tuning v2 非常合适」——v2 的原生场景是 NLU 分类，后续在 BERT（MRPC）上做 v2 才是与 Prefix(Qwen 生成) 有区分度的实验（注意 BERT 侧需用 `PeftModelForSequenceClassification` 路径，与本脚本结构差异较大，建议单独立项）。

### 5.4 修改后自查清单

- [ ] import 换成 `PrefixTuningConfig`
- [ ] `peft_config` 换类、`peft_type="PREFIX_TUNING"`、显式 `prefix_projection=False`、删掉 3 个 `encoder_*` 参数
- [ ] `gradient_checkpointing=False`
- [ ] `learning_rate=1e-4`
- [ ] `use_cache=False`（第 240 行）保留——v2 同样需要
- [ ] 第 286/293/306 行打印文案保持 "P-TuningV2" ✅（这回名副其实了）
- [ ] 预期 `print_trainable_parameters()` ≈ **917,504 trainable params (0.154%)**（peft 的 PrefixEncoder 是一张 `num_layers×num_virtual_tokens` 行、每行 2×hidden 的 embedding 表：28 层 × 16 × 2 × 1024；以实际打印为准）
- [ ] 训练正常启动、loss 逐步下降即验证成功

参考来源：
- [P-Tuning v2 论文（arXiv:2110.07602）](https://arxiv.org/abs/2110.07602)
- [P-Tuning v2 ACL 2022 版](https://aclanthology.org/2022.acl-short.8/)
- [THUDM/P-tuning-v2 官方仓库](https://github.com/THUDM/P-tuning-v2)

---

## 六、实战复盘：v2 跑通后 loss +119%、生成乱码 —— 与 Prefix 实验完全同根（等价性实证）

> 现象：eval_loss 4.8659 → 10.6779（+119.4%），ROUGE-L 0.0174 → 0.0052，生成乱码。
> 先说结论：**是的，与之前 Prefix Tuning 跑通后的现象（同文档第七节：+119.1%、ROUGE 0.0038、乱码）是同一个根因**——这不是巧合，也不需要重新排查。

### 6.1 数字复刻对照（这不是两个 bug，是同一个行为）

| | Prefix 实验（此前） | PTuningV2 实验（本次） |
|---|---|---|
| eval_loss | 4.8659 → 10.66（+119.1%） | 4.8659 → **10.6779**（+119.4%） |
| ROUGE-L | 0.0174 → 0.0038 | 0.0174 → 0.0052 |
| 生成 | 乱码（碎片循环） | 乱码 |
| 超参 | lr=1e-4, 10 epochs, gc=False | lr=1e-4, 10 epochs, gc=False |
| peft 配置 | `PrefixTuningConfig(16, projection 默认False)` | `PrefixTuningConfig(16, projection=False)` |

改对之后（5.2 的 4 处修改你全部落实了），v2 脚本与 Prefix 脚本在 peft 层面**就是同一段代码**——5.3 节「跑出来的曲线也应当接近」的预言应验，连失败的形态都复刻到小数点后两位。**这反过来是「v2 = peft 的 PrefixTuning」最硬的实证**：两条独立的实验链，同配置必然同行为。

### 6.2 根因（详见 `PrefixTuning实现检查与报错分析.md` 第七节，此处只收结论）

三组 GPU 对照实验已把责任定位清楚，不再是猜想：

1. **KV 注入门槛（主因，+4 以上）**：transformers 4.55 × Qwen3 组合下，只要 forward 传入 `past_key_values`，无论内容是什么（**全零同样触发**），attention 行为就系统性劣化——这是 KV 缓存路径（position 平移 / cache mask 对齐）的固有代价；
2. **peft 默认 N(0,1) 初始化（额外 +4.3）**：`PrefixEncoder` 对 embedding 无显式初始化，尺度比真实 K/V（~0.05）大 20 倍；
3. **起点坑填不平**：起点 +6.8 的坑，200 步 × lr=1e-4 只爬回 ~1.0（epoch1 即 11.65 → 终点 10.66/10.68），回不到基线 4.87；乱码是模型在坏 hidden 状态下的产物。

**可以排除的嫌疑**：你的脚本实现没有问题（4 处修改全部正确、训练真跑、梯度正常）；peft 也无实现 bug（旧文档实验 2 已用「绕过 peft、原生注入零 KV 同样翻倍」排除）。

### 6.3 定位与修复选项（与 Prefix 线 7.5 节对齐，按目标选）

- **目标 A（推荐）：封存为「机制理解」结论**。v2 线与 Prefix 线共用同一个结论：*「Prefix 系方法（含 P-Tuning v2）在 peft 0.18.1 × transformers 4.55 × Qwen3 × 0.6B 模型 × 100 条数据的组合下存在机制性起点罚金（+4~7），小数据小模型填不平；效果线用 LoRA/AdaLora（起点中性，B=0）」*。这正是你表格里「工业很少落地、被 LoRA 替代」判断的第一手实证；
- **目标 B（可选对照实验，一步之遥）**：把 `prefix_projection` 翻转为 `True` 再跑一次——
  ```python
  peft_config = PrefixTuningConfig(
      peft_type="PREFIX_TUNING",
      task_type=TaskType.CAUSAL_LM,
      num_virtual_tokens=16,
      prefix_projection=True,     # 重参数化MLP：prefix实际值由 Linear 初始化控制（小尺度），
      encoder_hidden_size=512,    #   天然避开 N(0,1) 初始化伤害；代价：可训练参数 0.9M → ~30M
      inference_mode=False
  )
  ```
  这一步同时完成两件事：验证「初始化是主要伤害源」（预期起点坑从 +6.8 缩到 +2 以内）；顺便得到 5.3 节想要的「v2(无重参数化) vs 原版 Prefix(有重参数化)」严格对照。注意 `encoder_hidden_size` 此时才生效（5.2 ②删掉它是因为 v2 形态用不到）；
- **目标 C（换回 v2 的主场）**：v2 论文的主战场是 **NLU 分类 + 较大数据量**（你表格也写了「BERT 跑 P-tuning v2 非常合适」）。Qwen3 生成 + 100 条数据本来就不在 v2 的适用面上——想要 v2 的「正面结果」，载体应是 BERT（MRPC 分类，`PeftModelForSequenceClassification` 无 KV 注入路径，上述门槛问题不适用），建议作为独立实验立项。

### 6.4 本轮学习产出

1. 用两条独立实验链实证了「论文名 ≠ 实现名」：**P-Tuning v2 在 peft 里就是 PrefixTuningConfig**，等价到连失败曲线都复刻；
2. 「起点中性」再次验证为 PEFT 方法选型的第一判据：LoRA 的 B=0（起点=base）vs Prefix 系的 +4~7 起点坑——跨方法对比必须先看起点；
3. 复盘方法论复用：本轮排查零成本，因为旧文档第七节的三组对照实验 + 定位直接迁移——好的归因文档（现象→实验→责任定位）是可以跨实验复用的资产。
