# QLoRA 微调改造方案（test_QLora_qwen3-0.6b.py）

> 前提：`test_QLora_qwen3-0.6b.py` 是 `test_Lora_qwen3-0.6b.py` 的复制（含 has_adapter 两模式 + 包装前基线结构，均沿用）。
> 本文只给方案不改源码。环境：bitsandbytes 0.50.2 / peft 0.18.1 / transformers 4.55.0。
> 关键数字均为本机 GPU 实测（RTX 4050 6GB，实验代码见第四节末尾）。
>
> ⚠️ **2026-09-17 复核：本文的 6 项改动中，改动 1 的关键一行（`quantization_config=bnb_config`）实际未落地**——脚本至今跑的不是 QLoRA。发现经过、影响分析与手动修复指引见 **第六节**，改前必读。
> ⚠️ **改动 1 修复后会立刻遇到第二层报错**：`ValueError: You cannot perform fine-tuning on purely quantized models`（在基线 Trainer 构造处）——这不是改错了，恰恰是量化生效的信号；根因分析与修复方案见 **第八节**。
> ✅ **两层问题均已解决，量化版已跑通（2026-09-17）**：基线 4.9813（纯 4bit base 口径）→ 4.6469（**-6.7%**），结果解读见 **第九节**。

## 一、先回答你的问题："添加一个 bnb 配置就可以了吗？"

**bnb 配置本身写对了**（nf4 / bf16 计算精度 / double quant 全部是 QLoRA 论文推荐组合），**但"只加这一个配置"不成立**。完整清单：

| # | 改动 | 性质 |
|---|------|------|
| 1 | `BitsAndBytesConfig` + 量化加载（你已写出，补 import） | 🔴 必改 |
| 2 | 训练分支加 `prepare_model_for_kbit_training(model, ...)` | 🔴 QLoRA 标准协议，缺了数值稳定性差 |
| 3 | `gradient_checkpointing=True`（当前是 False） | 🔴 QLoRA 标配 |
| 4 | `optim="paged_adamw_32bit"` | 🟠 QLoRA 论文三要素之三（防显存尖峰） |
| 5 | **4 处目录 `qwen3-0.6b_Lora` → `qwen3-0.6b_QLoRA`** | 🔴 隐藏炸弹：不改会污染 LoRA 线（见 3.5） |
| 6 | `LoraConfig` / 数据 / collator / 超参 **全部不动** | ✅ 单变量原则（3.6） |

---

## 二、你的 bnb 配置逐项检查

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # ✅
    bnb_4bit_quant_type="nf4",             # ✅ NF4 是 QLoRA 论文核心（信息论最优 4bit 数据类型）
    bnb_4bit_compute_dtype=torch.bfloat16, # ✅ 反量化后的计算精度，与训练 bf16=True 匹配
    bnb_4bit_use_double_quant=True,        # ✅ 量化常数再量化一次，再省 ~0.4bit/参数
)
```

需要补/说明的三点：

1. **缺 import**：`from transformers import BitsAndBytesConfig`（加进现有的 transformers import 列表）；
2. **`torch_dtype="auto"` 保留** ✓——它的角色是让**不被量化的层**（embedding、norm、lm_head）按 config.json 的 bfloat16 加载（省显存）；4bit 层的 dtype 由 `bnb_4bit_compute_dtype` 管，两者不冲突；
3. **量化范围**（实测确认）：196 个投影层（q/k/v/o/gate/up/down × 28 层）全部变成 `Linear4bit`；`lm_head` 与 `embed_tokens` **不量化**（bnb 默认排除 lm_head，且 embedding 不是 Linear；Qwen3-0.6B 二者权重 tied）——这意味着模型约 1/4 的参数（1.5 亿 embedding）不享受 4bit，是后面显存分析的关键。

---

## 三、其余必改点

### 3.1 改动 1：量化加载（替换脚本开头 L24-32 的 `from_pretrained`）

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,     # ← 新增
    TrainingArguments,
    Trainer,
    TrainerCallback
)
...
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",          # ★ QLoRA 必须保留：量化模型必须放 GPU
    local_files_only=True,
    torch_dtype="auto"
)
```

**这一处改动同时覆盖两种模式**（has_adapter 加载分支用的 base 就是开头这个 `model`）——复用分支无需任何额外改动，这是当前脚本结构的红利。

### 3.2 改动 2：训练分支加 `prepare_model_for_kbit_training`（QLoRA 的标准协议）

替换训练分支（else 分支）里的两行：

```python
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,   # ← 新增
)
...
else:
    # ---------- 模式二：首次运行，完整训练流程 ----------
    peft_config = LoraConfig( ...原样不动... )
    # ★ QLoRA 标准协议（peft 0.18.1 源码核实的三件事）：
    #   ① 冻结全部 base 参数
    #   ② 把所有非量化的 bf16/fp16 参数 cast 成 fp32（norm 层数值稳定性）
    #   ③ enable_input_require_grads + gradient_checkpointing_enable
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},   # 沿用原脚本的显式设置
    )
    model_lora = get_peft_model(model, peft_config)
    # model_lora.enable_input_require_grads()   # ← 删除：prepare 内部已做，重复无益
    model_lora.print_trainable_parameters()
```

为什么必须有它（读 `peft/utils/other.py` 源码确认）：不 cast fp32 的话，4bit 反量化计算叠 bf16 norm 在小模型上数值稳定性差；③ 是梯度检查点的前置条件（你原来的 `enable_input_require_grads()` 被它涵盖）。加载分支（`has_adapter=True`）**不需要** prepare——不训练，无梯度可言。

### 3.3 改动 3：`gradient_checkpointing=True`（L294，当前是 False）

```python
    gradient_checkpointing=True,        # QLoRA 标配：4bit 权重省下的显存要靠它压激活，6G 卡建议开
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },
```

### 3.4 改动 4：paged 优化器（`training_args` 的 `optim`）

```python
    optim="paged_adamw_32bit",   # QLoRA 论文三要素之三：显存不够时优化器状态分页到 CPU，防 OOM 尖峰
```

QLoRA 三要素对照：**NF4**（bnb config ✓）、**Double Quant**（✓）、**Paged Optimizers**（本条）。实测 0.6B 上优化器状态不大（adapter 5M 参数的 Adam ≈ 60MB），paged 的收益小，但它是"完整 QLoRA"的一部分且无副作用。

### 3.5 改动 5（隐藏炸弹）：4 处目录改名

当前脚本 4 处仍指向 `qwen3-0.6b_Lora`：

| 位置 | 内容 | 不改的后果 |
|---|---|---|
| L177 | baseline 的 `output_dir` | 与 LoRA 线混写（轻） |
| 主 `training_args` | `output_dir` / `logging_dir` | checkpoint 混写（轻） |
| **L227** | **`lora_save_path`** | **重**：QLoRA 适配器存进 LoRA 目录；且 `has_adapter` 检测**会误加载 LoRA 线的适配器**——适配器参数与 base 是否量化无关（形状只看 r/target_modules），LoRA 的 adapter 能直接套在 4bit base 上，你的"QLoRA 复用"实验从此用的全是 LoRA 权重，两条线互相污染 |

全部改为 `./training/qwen3-0.6b_QLoRA/...`。

### 3.6 单变量原则：为了与 LoRA 线公平对比，其它一切不动

`LoraConfig`（r=8 / alpha=16 / dropout=0.1 / 7 模块）、数据管线、`num_train_epochs`、不设 `learning_rate`（默认 5e-5）——**全部保持与 LoRA 脚本一致**。这样跑出的 `eval_loss 对比` 差值才能归因于"量化 + 同配置 adapter"。想要更好的绝对效果（调 lr 等）是下一轮实验，别和"验证 QLoRA 跑通"混在一起。

---

## 四、本机实测：显存账与量化损失（改前必读的预期管理）

GPU 实测（bf16 加载 vs 4bit NF4 加载，同一 5-token 前向）：

| 状态 | 显存（已分配） | 单条 loss |
|---|---|---|
| bf16 加载 | 1.12 GiB | 6.3041 |
| 4bit NF4 加载 | **0.51 GiB** | 7.1235（+0.82） |
| 4bit + `prepare_model_for_kbit_training` | 0.80 GiB | —（embedding 被 cast fp32，+0.29 GiB） |
| 4bit + prepare + adapter 包装 | 0.82 GiB | —（adapter 仅 ~10MB） |

三条诚实结论：

1. **显存收益比想象的小**：权重 1.12 → 0.82 GiB，只省 ~27%。原因：embedding（模型 1/4 参数）**不参与量化**，还被 prepare cast 成 fp32。QLoRA 的甜点区是 7B+ 模型（embedding 占比小、bf16 装不下），0.6B 上它是"技术练习 + 激活显存余量"，不是"救命手段"——但 6G 卡上训练峰值余量变大 + gc=True，仍是可感知的收益；
2. **量化损失真实存在**：4bit 前向 loss 比 bf16 高。上面 +0.82 来自无意义 token 的单条前向（仅示意方向，真实数字会小一些）——**以你跑出的基线 eval_loss 对比 4.8659 的差为准**；
3. 由此推出本实验最重要的方法论约束 ↓

> ⚠️ **QLoRA 线的 eval_loss 与 LoRA 线不可直接横比**（底座精度不同：4bit 量化 vs bf16）。
> 有效的对比是两个：① QLoRA 线**内部**前后对比（4bit base 基线 vs 4bit base + adapter）——衡量 adapter 增益；② QLoRA vs LoRA 的**微调后减基线**的**相对变化幅度**对比。把 QLoRA 基线 eval_loss 记下来，作为这条线的锚点数。

实测代码（可复跑）：

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
m = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B", quantization_config=bnb,
                                         device_map="auto", local_files_only=True, torch_dtype="auto")
print(m.is_loaded_in_4bit, torch.cuda.memory_allocated()/2**30)
m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
               target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
```

---

## 五、修改后自检清单

1. `python -m py_compile test_QLora_qwen3-0.6b.py` 通过；
2. 启动打印模型结构里能看到 `Linear4bit`（或运行时打印 `model.is_loaded_in_4bit` 为 True）；
3. **基线 eval_loss 是新数字**（比 4.8659 高，量化噪声所致）——记录它，这是 QLoRA 线的锚点；如果它等于 4.8659 反而说明量化没生效；
4. 训练分支 `print_trainable_parameters` 应为 **5,046,272**（r=8 全 196 模块，与 LoRA 线相同——adapter 不受量化影响）；
5. 训练显存峰值用 `nvidia-smi` 瞄一眼：应显著低于 LoRA 线；跑完 60 步不 OOM；
6. 微调后 eval_loss 相对**自己的基线**应下降（幅度可与 LoRA 线的 -7.1% 相对幅度对照）；
7. `./training/qwen3-0.6b_QLoRA/lora_adapter/` 产出适配器（不含量化信息、不含量化权重——`trainer.save_model` 仍只存 adapter，与 LoRA 相同）；
8. 二次运行复用分支：正常加载（base 由开头 bnb 加载）、基线同样回到纯 4bit base 口径、delta 复现首训结论；
9. `./training/qwen3-0.6b_Lora/` 目录在 QLoRA 运行后**没有任何新文件**（验证 3.5 改干净了）。

---

## 六、2026-09-17 复核：改动 1 实际未落地（发现经过 + 手动修复指引）

> 本节是复盘，不是推翻前五节——**前五节的方案本身全对**，问题出在执行时漏了改动 1 里最关键的一行。留档供学习：这是"改了一半"的典型样本，也是"自检清单写了不等于查了"的活教材。

### 6.1 发现经过与证据链

复盘全仓库参数配置时发现：`bnb_config` 在 L26-31 定义、import 也补了，但 **L33-38 的 `from_pretrained` 没有传 `quantization_config=bnb_config`**——配置对象成了死代码，量化从未发生。

**铁证是两条线的 eval_loss 几乎逐位一致**（来自各自 `checkpoint-*/trainer_state.json`）：

| epoch | LoRA 线 | "QLoRA"线 | 差值 |
|---|---|---|---|
| 1 | 4.5737 | 4.5748 | 0.0011 |
| 2 | 4.5384 | 4.5384 | **0.0000** |
| 3 | 4.5222 | 4.5212 | 0.0010 |

两条线如果真是"bf16 底座 vs 4bit 底座"，不可能逐位一致到千分位——**它们本来就是同一个实验**（bf16/同一 adapter 配置/同一数据）。此前"0.6B 上量化损失观测不到"的解读是错误归因：不是损失观测不到，是量化根本没跑。

### 6.2 改动 1~6 的实际落地情况

| # | 改动 | 方案 | 实际落地 |
|---|---|---|---|
| 1a | `BitsAndBytesConfig` + import | 🔴 必改 | ✅ 已做（L15 import、L26-31 定义） |
| **1b** | **`quantization_config=bnb_config` 传入 `from_pretrained`** | 🔴 必改 | ❌ **漏了——本 bug 的全部所在** |
| 2 | `prepare_model_for_kbit_training` | 🔴 | ✅ 已做（L271-276） |
| 3 | `gradient_checkpointing=True` | 🔴 | ✅ 已做（L299） |
| 4 | `optim="paged_adamw_32bit"` | 🟠 | ✅ 已做（L308） |
| 5 | 4 处目录改名 | 🔴 | ✅ 已做（`qwen3-0.6b_QLora`） |
| 6 | 单变量原则 | ✅ | ✅ 保持 |

### 6.3 为什么漏一行却毫无报错（连锁影响分析）

`transformers` 对"定义了 `BitsAndBytesConfig` 却没传"**不会有任何警告**——它不知道你有这个意图。于是链条逐级静默偏移：

1. `torch_dtype="auto"` 让模型以 **bf16** 加载（权重显存 1.12 GiB，而非 4bit 的 0.51 GiB）；
2. `prepare_model_for_kbit_training` 被照常调用——它**假设模型已量化**，其"把非量化 bf16 参数 cast 成 fp32"的逻辑在未量化模型上作用于**全部参数**（设计预期只 cast embedding/norm/lm_head 这些漏网层）→ 底座变成 **fp32（≈2.4 GiB，反而更大）**；
3. paged 优化器、gc、Trainer 全部照常工作——它们不关心底座精度；
4. 全程零报错、零警告，eval_loss 还"看起来正常"（因为它确实和 LoRA 线一样正常）。

**结果：这条线实际跑的是「fp32 底座 + LoRA + paged 优化器」**——比方案预期的 4bit 底座显存更大、且没有任何 QLoRA 成分。

### 6.4 手动修复（一行，留给你亲手改）

在 `test_QLora_qwen3-0.6b.py` 的 `from_pretrained`（L33-38）里补一行：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,   # ★ 补上这一行——bnb_config 从"死代码"变成生效配置
    torch_dtype="auto",
    device_map="auto",      # ★ QLoRA 必须保留：量化模型必须放 GPU
    local_files_only=True
)
```

其余**一律不动**：`prepare_model_for_kbit_training`（修复后它的 cast fp32 才符合设计本意——只作用于 embedding/norm 等未量化层）、paged 优化器、目录、数据管线全部保持。这一处改动同时覆盖两种模式（复用分支的 base 就是开头这个 `model`，见 3.1 的说明）。

### 6.5 修复后如何确认量化真的生效（按顺序做，任何一条不过就停下）

| # | 验证 | 预期 | 若不符合的含义 |
|---|---|---|---|
| 1 | `print(model.is_loaded_in_4bit)` | `True` | quantization_config 没接上 |
| 2 | `print(model)` 前几层 | `Linear4bit`（`Params: 4bit...`） | 同上 |
| 3 | `torch.cuda.memory_allocated()/2**30` | ≈ **0.5 GiB**（4bit）而非 1.12（bf16）；prepare 后 ≈0.8 | 同上；对照第四节实测表 |
| 4 | 训练分支 `print_trainable_parameters` | 仍为 **5,046,272**（adapter 不受量化影响） | 若变了，说明改错了别处 |
| 5 | **基线 eval_loss** | **一个新数字，比 4.8659 高**（量化噪声）——这才是 QLoRA 线的真锚点 | 若仍 ≈4.8659，量化没生效（第五节清单第 3 条的本意） |

### 6.6 重跑前的清理（不做会复现另一类静默错误）

1. **改名/删除旧产物** `training/qwen3-0.6b_QLora/lora_adapter/`（它是 fp32 底座训出来的，不属于修复后的实验）。不清理的后果：`has_adapter=True` 直接复用旧权重跳过训练，而基线却是新的量化 base——"微调后减基线"的 delta 从此不可信（这正是 3.5 目录串线坑的变体，只是这次发生在同一条线的时间维度上）；
2. 分词缓存**不用动**（`tokenize_func`/max_length 没变，缓存仍有效）；
3. TensorBoard 旧曲线建议一并清掉（`logs/`），避免新旧口径的线画在一张图里。

### 6.7 这次的元教训

1. **"6 项改动"型任务最怕改 5 项漏 1 项**——漏的那项往往是最核心的"接线"（定义了配置≠配置生效），且**静默**。修复时对照方案清单逐项打勾，改完再逐项验证；
2. **自检清单的价值在于逐条执行**：第五节第 2 条"看到 Linear4bit"本可当场拦住这个 bug。清单写了没查，等于没写；
3. **跨线数字"过于接近"和"过于不同"一样值得怀疑**：两条应不同的线逐位一致，比两条线差异大要多想一步——差异可能有十个解释，逐位一致通常只有一个。

---

## 七、修复后本文其他部分的有效性

- **第四节**（显存账与量化损失）：全部成立——那组 1.12/0.51/0.80/0.82 GiB 实测就是用正确写法（`quantization_config=bnb_config`，见 4 节末实测代码）单独跑出来的。**注意：那是"方案验证实验"的数字，训练脚本从未兑现它**，修复重跑后请以训练时的 `nvidia-smi` 为准再记一版；
- **第四节 ⚠️ 不可横比**：修复后才真正适用（之前两条线同底座，其实可以横比——只是没意义）；
- **第五节自检清单**：修复重跑时逐条执行，特别是第 2、3 条。

---

## 八、修复改动 1 后的连锁报错：纯量化模型 × 基线 Trainer

> 现象：按 6.4 补上 `quantization_config=bnb_config` 后重跑，在 **基线 Trainer 构造处**（当前脚本 L195 `trainer_baseline = Trainer(`）立刻崩：
>
> ```
> ValueError: You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on
> top of the quantized model to correctly perform fine-tuning. Please see:https://huggingface.co/docs/transformers/peft
> ```

### 8.1 先明确：这是修复成功的信号，不是倒退

这个检查的触发前提是 `model.is_quantized == True`——**只有量化真的生效它才会响**。bf16 模型 `is_quantized=False`，检查直接跳过；第六节 bug 期间（量化从未发生）这条线跑了若干轮也从未报过，正是因为模型压根不是量化模型。所以看到这个报错应当先确认：**改动 1 修对了**，现在踩到的是被旧 bug 掩盖的下一层问题。

### 8.2 根因（源码级，transformers 4.55 `trainer.py` `__init__`）

```python
_is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
    model, "_hf_peft_config_loaded", False
)
...
# At this stage the model is already loaded
if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
    raise ValueError(
        "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
        " the quantized model to correctly perform fine-tuning. ...")
```

三个条件**同时满足**即 raise，而基线 Trainer 构造（`model=model`，裸量化模型）全中：

| # | 条件 | 基线时刻的状态 |
|---|---|---|
| ① | `is_quantized == True` | ✅ 量化加载成功（你要的状态） |
| ② | 模型还没挂过 peft 配置 | ✅ 此刻 `get_peft_model` 还没执行（它在后面 else 分支里） |
| ③ | 不是 `PeftModel` 实例 | ✅ 传的是裸 `model` |

三个要点：

1. **检查不看 `do_train=False`**。检查的意图是防"训练"纯量化模型——4bit 权重是打包的量化存储，不可微、不可训练，没有 adapter 时"全量微调"等于静默无效。但它无法区分你构造 Trainer 是为了训练还是只为了评估，于是在 `__init__` 一刀切，你的 `do_train=False, do_eval=True` 完全不被理会；
2. **全仓库只有这条线会触发**：其余 10 条线的 base 都是 bf16，①不成立。这是 QLoRA 专属坑；
3. **深层是结构前提被打破**："包装前基线"结构（`trainer_baseline` 用未包装的裸 model）有一个从未写出来的假设——**base 可以直接构造 Trainer**。bf16 线永远成立所以从未显形；量化加载把这个假设打破了。

### 8.3 修复方案对比

| 方案 | 做法 | 判定 |
|---|---|---|
| **A：包装上移 + `disable_adapter()` 基线** | 把 peft 包装移到基线评估**之前**，基线在 `with model_lora.disable_adapter():` 内评 | ✅ **推荐**，见 8.4 |
| B：双次加载 | 量化加载评完基线 → `del model` + `empty_cache()` → 再量化加载供训练 | ⚠️ 可行但笨重：两段加载慢、代码翻倍。若为省事改用 bf16 加载评基线，则**口径错误**——基线是 bf16 base，微调后是 4bit 底座，delta 混入量化损失，违反第四节 ⚠️ 的锚点纪律 |
| C：手写 eval 循环绕开 Trainer | 10 条验证集手写前向算 loss | ❌ 不推荐做主基线：与 Trainer 的 token 加权 loss 口径（`num_items_in_batch` 机制）不严格一致，跨线对比变脏。可作为**附加**验证手段（AdaLora 线的逐条配对分析就是这种用法） |

**方案 A 的正确性论据**：

1. **基线口径正确**：`disable_adapter()` 运行时把 adapter 输出置零，而 LoRA 的 `B=0` 初始化本就使 ΔW=B·A=0——双保险下，with 块内的前向**数学等价纯 4bit base**。这正是第四节要求的锚点口径（"4bit base 基线 vs 4bit base + adapter"）；
2. **对 LoRA 完全干净**：`LoraModel.forward` 没有正则段——AdaLora 的"`disable_adapter` 关不掉 forward 正则段"教训（复盘文档坑 17）**在此不适用**；
3. **两种模式统一**：复用模式（`PeftModel.from_pretrained`）的基线同样用 with 块，首训/复用口径一致；
4. **显存友好**：单模型对象，无第二份权重（6GB 卡）；
5. **主 Trainer 零改动**：`model_lora` 是 `PeftModel`，上表条件③不成立，检查通过。

### 8.4 方案 A 手动修改指引（3 处，留给你亲手改）

> 行号基于当前文件（量化修复后），改动后行号会漂移，**以代码锚点为准**。

**改动 ①：把"加载 + peft 包装"整体上移到基线块之前**（原 L237-281 区域重组）

把原 L237-240 的加载节、原 else 分支里的 `peft_config` / `prepare_model_for_kbit_training` / `get_peft_model` 全部搬到 `training_args_baseline`（L186）**之前**，合并为一个顶格块：

```python
# ===================== 加载（上移：基线评估之前） =====================
lora_save_path = "./training/qwen3-0.6b_QLora/lora_adapter"
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))

# ===== peft 包装（顶格：两种模式各自构造 model_lora）=====
# ★ 量化模型不能直接构造 Trainer（transformers 检查 is_quantized + 未挂 adapter，
#   见 QLoRA 文档 8.2），基线改用 disable_adapter()，故包装必须先于基线评估
if has_adapter:
    print(f"检测到已保存的LoRA适配器：{lora_save_path}，跳过训练")
    model_lora = PeftModel.from_pretrained(model, lora_save_path)
    model_lora.print_trainable_parameters()
else:
    peft_config = LoraConfig(
        ...原样从原 else 分支搬上来，一个字不改...
    )
    # ★ QLoRA 标准协议（三件事注释原样保留）
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
```

两条铁律：**`prepare_model_for_kbit_training` 必须在 `get_peft_model` 之前**（顺序不变）；`peft_config` 的内容原样搬运，不要顺手"优化"。

**改动 ②：baseline Trainer 换 model，评估与基线生成包进 `disable_adapter()`**（原 L195-233）

```python
trainer_baseline = Trainer(
    model=model_lora,              # ← 改：裸量化 model → 挂了 adapter 的 PeftModel（满足 8.2 检查）
    args=training_args_baseline,   # 不动
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

with model_lora.disable_adapter():     # ← 基线 = 纯 4bit base（adapter 输出置零，ΔW=B·A=0）
    print("==== 预训练模型原始基线（纯 4bit base 口径）====")
    baseline_eval = trainer_baseline.evaluate()
    print(baseline_eval)
    baseline_generations = [generate_answer(model_lora, ins) for ins in test_instructions]
```

细节：`generate_answer` 的入参从 `model` 换成 `model_lora`（首训/复用两模式统一走 PeftModel 路径）；其内部的 `m.eval()` / `torch.no_grad()` 在 with 块内照常工作；训练阶段 `trainer.train()` 会自行把模型切回 train 模式，`lora_dropout` 恢复生效。

**改动 ③：删除原位置的旧块**

原 else 分支（含 `peft_config`、prepare、get_peft_model，原 L249-281）整段删除——已由改动 ① 顶格块取代。**训练守卫 `if not has_adapter: trainer.train()`、主 `training_args`、主 `Trainer(model=model_lora, ...)`、评估与保存全部不动。**

顺带收益：`model_lora` 变为顶格定义后，脚本后段所有引用（主 Trainer、evaluate、generate、保存）都在同一作用域——提前规避了 CLAUDE.md 坑 6 那类"定义在分支内、引用在分支外"的 NameError。

### 8.5 修复后的口径自检

| # | 检查 | 预期 |
|---|---|---|
| 1 | **基线 eval_loss** | 一个**新数字，比 4.8659 高**（4bit 量化噪声）——它就是 QLoRA 线的真锚点。若仍 ≈4.8659 → with 块没包住 evaluate，或量化没生效 |
| 2 | 微调后 eval_loss | 相对**自己的基线**应下降；相对幅度可与 LoRA 线的 -7.1% 对照（第四节 ⚠️ 的两个有效对比之一） |
| 3 | 训练首步 `nvidia-smi` | 权重部分 ≈0.8 GiB（第四节实测），**不是**第六节 bug 状态 fp32 底座的 ≈2.4 GiB |
| 4 | 复用模式二次运行 | 基线同样回到纯 4bit base 口径，delta 复现首训结论 |
| 5 | 沿用第五节清单 | 第 9 条：`./training/qwen3-0.6b_Lora/` 仍无任何新文件 |

### 8.6 本节的元教训

1. **报错的字面意思 ≠ 你的场景**：ValueError 说"不能微调"，你在"只想评估"——根因是**检查粒度比需求粗**（`Trainer.__init__` 无法区分构造用途）。解法不是对抗检查（改源码、找 hack 参数），而是**让模型满足检查的假设**：挂上 adapter、用 `disable_adapter()` 拿基线。顺着框架的设计意图走，比对抗它便宜；
2. **结构性方案有隐藏前提**："包装前基线"的前提是"base 可直接构造 Trainer"。量化加载打破了这个从未被写下的前提。复制结构时除了搬参数（6.7 元教训 1），还要搬"前提清单"——这是同一课的更高一层；
3. **修复链思维**：改动 1 修好 → `is_quantized=True` → 被旧 bug 掩盖的结构假设失效 → 新报错。**修好一个 bug 后冒出新报错，多数时候是前进到了下一层，不是改错了**。判断标准：新报错是否由"刚修好的行为"直接触发——是，就是前进。

---

## 九、量化版实测结果（2026-09-17，方案闭环）

按 6.4（改动 1）+ 8.4（方案 A）改造并重跑，全部通过。

### 9.1 实测数字

| 阶段 | eval_loss | 说明 |
|---|---|---|
| 基线（`disable_adapter()`，纯 4bit base 口径） | **4.9813** | 比 bf16 基线 4.8659 高 |
| 微调后（4bit base + adapter，epoch 3） | **4.6469** | 60 步训练 |
| **相对自己的基线** | **-6.7%** | 第四节 ⚠️ 认可的两个有效对比之一 |
| ROUGE-L（字符级，辅助） | 0.1068 → 0.1025 | 落在字符级基线区（0.10~0.14），无字面参考价值 |

### 9.2 对照 8.5 口径自检表

| # | 自检项 | 结果 |
|---|---|---|
| 1 | 基线是"比 4.8659 高的新数字" | ✅ 4.9813（量化噪声真实可见，量化确认生效） |
| 2 | 微调后相对自己的基线下降 | ✅ -6.7% |
| 3 | 训练显存 `nvidia-smi` | —（本次未记录，下次补） |
| 4 | 复用模式二次运行 | ⏳ 待跑（注意 8.4 后口径观察：加载模式基线未过 prepare，锚点可能与首训差在第 3~4 位小数） |
| 5 | LoRA 目录无新文件 | ✅（目录分立已确认） |

### 9.3 三个结论

1. **量化噪声首次有了真实口径的数字：+0.115 NLL（+2.4%）**（4.9813 vs bf16 基线 4.8659，同 collator/同批量/同 bf16 Trainer）。此前第四节只有"无意义 token 单条前向 +0.82"的方向示意——真实损失比那个示意小一个数量级。**"4bit 量化在 0.6B 上损失多大"这个此前未测的问题，答案是 +2.4%**；
2. **4bit 底座不伤 adapter 的训练增益（相对意义下）**：QLoRA -6.7% vs LoRA -7.1%，基本持平。即：量化损失是"底座起点"的一次性偏移（+0.115），adapter 在量化底座上的**相对**学习能力没有被削弱——这与 QLoRA 论文的核心主张（量化底座 + adapter ≈ 全精度底座 + adapter）在 0.6B 小模型上的定性吻合。**注意纪律不变**：4.6469 与 LoRA 的 4.5222 仍不可横比（底座不同），能比的只有相对幅度；
3. **方案六项改动的最终价值排序**（复盘视角）：真正决定成败的是改动 1b（`quantization_config` 接线）和 8.4 的结构改造（包装先于基线）——其余（paged 优化器、double quant、目录改名）是正确性/卫生项，但 0.6B 上对结果数字的影响都远小于这两项。

### 9.4 与第四节预期管理的对账

| 第四节当时的说法 | 实测后的修正/确认 |
|---|---|
| "基线 eval_loss 是新数字（比 4.8659 高），如果它等于 4.8659 反而说明量化没生效" | ✅ 精确命中：4.9813，且它从此成为 QLoRA 线的锚点数 |
| "量化损失真实存在……真实数字会小一些——以基线 eval_loss 对比 4.8659 的差为准" | ✅ 实测 +0.115，确实比示意值 +0.82 小一个量级 |
| "权重显存 1.12 → 0.82 GiB（方案阶段单独实测）" | 本次训练未记录 nvidia-smi，显存口径下次补（训练线兑现量化的间接证据：基线 loss 从 4.8659 变为 4.9813，量化路径已参与前向） |
| "QLoRA 线 eval_loss 与 LoRA 线不可横比" | ✅ 纪律维持；现在两条线都有了各自的锚点（4.8659 / 4.9813），横向只比相对幅度 |
