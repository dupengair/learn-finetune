# QLoRA 微调改造方案（test_QLora_qwen3-0.6b.py）

> 前提：`test_QLora_qwen3-0.6b.py` 是 `test_Lora_qwen3-0.6b.py` 的复制（含 has_adapter 两模式 + 包装前基线结构，均沿用）。
> 本文只给方案不改源码。环境：bitsandbytes 0.50.2 / peft 0.18.1 / transformers 4.55.0。
> 关键数字均为本机 GPU 实测（RTX 4050 6GB，实验代码见第四节末尾）。

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
