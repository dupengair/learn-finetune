# AdaLora 微调代码审查报告（test_AdaLora_qwen3-0.6b.py）

> 审查基准：`peft 0.18.1` + `transformers 4.55.0` 源码逐行核对（本地 conda 环境 `ai-gpu`）。
> 本文只指出问题与修改建议，**未改动任何源码**，请对照文中代码块手动修改，便于学习。

## 结论速览

| # | 问题 | 位置 | 严重级别 |
|---|------|------|---------|
| 1 | `lora_dropout = 0.1` 行末缺逗号，SyntaxError | L181 | 🔴 阻塞：脚本无法运行 |
| 2 | `AdaLoraConfig` 缺必填参数 `total_step`，运行时抛 ValueError | L162-185 | 🔴 阻塞：修完 #1 仍无法运行 |
| 3 | 全脚本从未调用 `update_and_allocate()`，秩预算调度完全失效，AdaLora 退化 | 全局 | 🔴 核心：跑通也不是 AdaLora |
| 4 | `r=8` 参数无效，控制最终秩的是 `target_r` | L175 | 🟠 语义错误（当前恰好巧合） |
| 5 | `tinit/tfinal/deltaT` 与实际训练步数（≈60 步）严重不匹配 | L177-179 | 🟠 实验失效 |
| 6 | `modules_to_save=["classifier"]` 是 BERT 脚本残留 | L182-184 | 🟡 无害但误导 |
| 7 | `output_dir/logging_dir` 仍是 `qwen3-0.6b_Lora`，与 LoRA 实验产物混写 | L221-222 | 🟡 工程混乱 |
| 8 | `enable_input_require_grads()` 与 `gradient_checkpointing=False` 矛盾 | L187 / L237 | 🟡 冗余 |
| 9 | 若干注释对 AdaLora 语义理解有偏差 | L177-180 等 | ⚪ 认知纠偏 |
| 10 | 加载模式（复用成果分支）`trainer.evaluate()` 崩溃：`trainable_adapter_name` 不存在，需 `is_trainable=True` | 加载分支 | 🔴 阻塞（详见第八节，实测复现并验证修复） |

---

## 一、阻塞性错误（脚本目前根本跑不起来）

### 错误 1：缺逗号 → SyntaxError（L181）

```python
        lora_dropout = 0.1               # 防止过拟合      ← 行末缺逗号
        modules_to_save = [
```

实际验证（`python -m py_compile test_AdaLora_qwen3-0.6b.py`）：

```
  File "test_AdaLora_qwen3-0.6b.py", line 181
    lora_dropout = 0.1               # 防止过拟合
                   ^
SyntaxError: invalid syntax. Perhaps you forgot a comma?
```

**修复**：`lora_dropout = 0.1,`（同时见下文建议直接删掉 `modules_to_save`）。

### 错误 2：缺必填参数 `total_step`（peft 0.18.1 强制校验）

peft 0.18.1 的 `peft/tuners/adalora/config.py` `__post_init__` 中：

```python
if self.total_step is None or self.total_step <= 0:
    raise ValueError("AdaLoRA does not work when `total_step` is None, supply a value > 0.")

if self.tinit >= (self.total_step - self.tfinal):
    raise ValueError(
        "The supplied schedule values don't allow for a budgeting phase. Decrease `tfinal`/`tinit` or "
        "increase `total_step`."
    )
```

即 AdaLora 的三阶段调度（热身 → 衰减 → 定秩微调）**必须预先知道总训练步数**，不传直接抛异常。
这和 LoRA 的"填 r 就能用"完全不同——AdaLora 的参数是**一套互相约束的调度计划**，不是独立旋钮。

**修复**：见第五节"推荐配置"，`total_step` 必须等于真实训练总步数（本脚本 ≈ 60，见错误 5 的计算）。

---

## 二、核心功能缺失：从未调用 `update_and_allocate()`（最重要的问题）

这是 AdaLora 与普通 LoRA **最本质的区别**：peft **不会**自动做秩预算调度。`peft/tuners/adalora/model.py` 的 docstring 明确写着：

```
This should be called in every training step after `loss.backward()` and before `zero_grad()`.
```

该方法内部完成两件事（`peft/tuners/adalora/layer.py` `RankAllocator`）：
1. **重要性打分**：`ipt = |p × p.grad|`，用 EMA 平滑（`beta1`）并叠加不确定性量化（`beta2`）——只有调用它，梯度信息才会被收集；
2. **预算分配**：按三阶段调度算出当前总预算 budget，用 `torch.kthvalue` 选出最不重要的 `init_bgt - budget` 个"三元组"，把它们的 `lora_E` 置 0（mask）。

你当前脚本**一次都没调用它**，后果是：
- 预算永远停在 `init_bgt`（全部层按 `init_r=12` 铺满），`init_r → target_r` 的衰减**从未发生**；
- `tinit / tfinal / deltaT / beta1 / beta2` 全部形同虚设；
- 最终实际训练的只是一个"秩 12 的三因子（B·diag(E)·A）参数化 LoRA + 正交正则"——**跑通了也不是 AdaLora**，实验结论会完全失真。

### 正确的触发时机（transformers 4.55 有一个坑）

网上常见写法是用 `TrainerCallback.on_step_end` 传 `state.global_step`。**在 transformers 4.55 下这个写法会崩溃**，因为训练循环里的顺序是（`transformers/trainer.py` L2648-2663）：

```python
self.optimizer.step()                                    # L2648
...
model.zero_grad()                 # L2660：梯度被置 None！
self.state.global_step += 1                              # L2661
self.control = self.callback_handler.on_step_end(...)    # L2663：此时 p.grad 已经是 None
```

而重要性打分需要读 `p.grad`（`(p * p.grad).abs()`），`on_step_end` 时梯度已被清空，会抛
`TypeError: unsupported operand type(s) for *`。

**正确时机是 `on_optimizer_step`**（L2650）。`trainer_callback.py` L393 的 docstring 原文："Event called **after the optimizer step but before gradients are zeroed out**"——恰好逐字对上 peft 官方示例中 `update_and_allocate` 的位置（`optimizer.step()` 之后、`zero_grad()` 之前），且该事件位于梯度累积边界内，**每个 optimizer step 只触发一次**，正合 AdaLora 以"步"为单位的调度语义。

为什么不用更早的 `on_pre_optimizer_step`（L2639，optimizer.step 之前）？梯度在两个位置都完整、重要性打分都能算，但 **mask 的生效语义不同**：若在 optimizer.step 之前 mask，紧接着的 Adam 更新会立刻把刚置 0 的 `lora_E` 又推开（Adam 的更新量只取决于梯度和动量，与参数当前值无关），mask 被部分撤销；尤其在定秩微调段，我们希望 forward 看到的就是 mask 后的"降秩"模型。在 optimizer.step **之后** mask，置 0 的值能完整保持到下一轮前向。

**修改建议**（加在 Trainer 构建之前，并传入 `callbacks`）：

```python
from transformers import TrainerCallback

class AdaLoraBudgetCallback(TrainerCallback):
    """AdaLora 秩预算调度必须每步手动触发。
    时机选 on_optimizer_step（optimizer.step 之后、zero_grad 之前），
    与 peft 官方示例的位置一一对应：
      backward → optimizer.step() → update_and_allocate → zero_grad
    而常见的 on_step_end 写法在 transformers 4.55 中位于 zero_grad 之后，
    此时 p.grad=None，update_ipt 会直接报错。"""
    def on_optimizer_step(self, args, state, control, **kwargs):
        base = kwargs["model"].base_model      # PeftModel -> AdaLoraModel
        base.update_and_allocate(state.global_step)

trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[AdaLoraBudgetCallback()],   # ← 新增
)
```

三点说明：
- `kwargs["model"]` 可用：`CallbackHandler.call_event`（trainer_callback.py L554-562）会自动向每个回调注入 `model=...`、`optimizer=...` 等关键字参数，无需自己捕获模型引用；
- 回调触发时 `state.global_step` 尚未自增（依次为 0, 1, 2, ...），与手写循环的 1, 2, 3, ... 只差一步平移，调度按步单调，无实质影响；
- 对照 peft docstring 给的手写循环写法，可以看清 `update_and_allocate` 在流程中的位置：
  ```python
  loss = model(**inputs).loss
  loss.backward()
  optimizer.step()
  model.base_model.update_and_allocate(i)   # ← 就是这个位置
  optimizer.zero_grad()
  ```

---

## 三、配置语义错误

### 错误 3：`r=8` 在 AdaLora 中完全无效（L175）

`peft/tuners/adalora/config.py` 源码原文：

```python
if self.r != 8:  # 8 is the default value for 'r' in LoraConfig
    warnings.warn(
        "Note that `r` is not used in AdaLora and will be ignored."
        "If you intended to set the initial rank, use `init_r` instead."
    )
```

AdaLora 中真正起作用的是：
- `init_r`：初始秩（你已设 12 ✓，各层构建时用的就是它，见 `adalora/model.py` `_create_and_replace` 里 `"r": lora_config.init_r`）；
- `target_r`：最终平均秩（**默认 8**，你没传）。

你现在传的 `r=8` 恰好等于 LoraConfig 的默认值，**连警告都不会触发**，属于纯静默无效——最危险的一种写法，因为将来有人改 `r` 时会误以为它生效。

**修改建议**：删掉 `r = 8`，显式写 `target_r = 8`，并纠正注释（"初始化秩"的说法属于 `init_r`）。

### 错误 4：`modules_to_save=["classifier"]` 是 BERT 残留（L182-184）

Qwen3-0.6B 是 `AutoModelForCausalLM`，没有 `classifier` 模块。peft 对 `modules_to_save` 中匹配不到的模块名**静默忽略**（不报错），所以这行无害但纯属误导。`modules_to_save=["classifier"]` 是 BERT SEQ_CLS 实验的必备项，复制脚本时忘了删。

**修改建议**：整个删除。

### 错误 5：`tinit=200 / tfinal=1000 / deltaT=10` 与训练规模严重不匹配（L177-179）

先算真实步数：
- 训练集 80 条（`select(range(100))` 后 80%）；
- `per_device_train_batch_size=1` × `gradient_accumulation_steps=4` → 每 epoch 80/4 = **20 步**；
- `num_train_epochs` 未设置（TrainingArguments 默认 **3**）→ 总步数 ≈ **60 步**。

而调度参数要求：前 200 步是热身（预算不动），要到第 1000 步附近才完成衰减。**就算修好错误 1/2/3，60 步全程都在热身段，秩永远不会衰减**——AdaLora 依然名存实亡。

`total_step` 的校验关系（config.py 源码）：`tinit < total_step - tfinal`，即 **`total_step > tinit + tfinal`**。

**修改建议**（配套参数，可直接抄）：

```python
target_r = 8,
init_r   = 12,
tinit    = 5,      # 热身段：回调步号 0~5 预算不动
tfinal   = 15,     # 定秩段：从第 total_step−tfinal = 45 步起（45 步为切换点：
                   #   强制 mask + 清空重要性统计；46 步起每步按 rank_pattern 重新 mask）
deltaT   = 2,      # 衰减段内每 2 步 mask 一次
total_step = 60,   # ★ 必填：= (80/4) × num_train_epochs；回调传入的步号是 0~59
```

并在 `TrainingArguments` 里**显式**写出 `num_train_epochs=3`（目前隐式依赖默认值），保证 `total_step` 与真实步数一致。若改了数据量或 epoch 数，记得同步重算 `total_step`。

### 错误 6：输出目录没改（L221-222）

```python
output_dir="./training/qwen3-0.6b_Lora/output",
logging_dir="./training/qwen3-0.6b_AdaLora/logs",   # ← 实际两行都是 _Lora
```

`lora_save_path`（L149）已经改成 `qwen3-0.6b_AdaLora`，但 checkpoint / TensorBoard 目录还是 `_Lora`——两个实验的产物会混进同一目录。

**修改建议**：两处都改为 `./training/qwen3-0.6b_AdaLora/...`。

---

## 四、行为提醒（不是 bug，但会误解实验结果）

### 4.1 `eval_loss` 语义变了，不能与 LoRA 实验直接横比

`AdaLoraModel.forward`（`peft/tuners/adalora/model.py`）会**自动**把正交正则加进 loss：

```python
outputs.loss += orth_reg_weight * regu_loss   # orth_reg_weight 默认 0.5
```

评估时也传了 labels → **`eval_loss` 里同样含这项正则**。注意正则的作用对象：regu 循环只遍历名字含 `lora_A` / `lora_B` 的参数（计算 `‖AAᵀ−I‖_F + ‖BᵀB−I‖_F` 后取平均），**`lora_E`（包括对它的 mask）根本不参与正则项**。正则项的数值随 A/B 的正交化程度变化（训练中通常下降），这部分下降不是模型学得更好。与 `test_Lora_qwen3-0.6b.py` 比较时，`eval_loss` 数值没有可比性，看**相对变化趋势**和**生成对比**即可。

### 4.2 peft 的"降秩"是 mask，不是真缩秩

peft 源码里 `resize_modules_by_rank_pattern`（真正缩模块形状）被官方注释禁用（"for some reason, this freezes the trainable parameters"），实际做法是 `masked_fill_(lora_E, 0)`。因此：
- 训练中**显存不会随秩衰减而下降**（模块形状不变）；
- `print_trainable_parameters` 的参数量训练前后不变；
- 可训练参数量与同秩 LoRA 几乎相同（每个目标层只多 `r` 个奇异值参数 `lora_E`）；
- **保存/加载时形状会真正缩小**：保存端 `save_and_load.py` L119 按 `rank_pattern` 截断 state_dict（只保留未被 mask 的三元组），加载端 L502 调 `resize_modules_by_rank_pattern` 按各模块实际秩**缩形重建**——加载后的适配器比训练结束时更小（本项目实测：训练时 6,539,568 → 加载后 5,039,648）。但这个重建也是第八节崩溃问题的触发环节之一，加载分支需要加 `is_trainable=True`。

### 4.3 `enable_input_require_grads()` 与 `gradient_checkpointing=False` 矛盾（L187 / L237）

`enable_input_require_grads()` 只在开梯度检查点时才需要；且 L238 已配 `use_reentrant: False`（非 reentrant 检查点本就不依赖输入梯度）。当前配置下这行是冗余的。二选一：
- 保持 `gradient_checkpointing=False` → 删掉 L187；
- 或干脆开启 `gradient_checkpointing=True` → L187 就有意义了。考虑到 AdaLora 要额外做正则计算与重要性打分，6GB 卡上想省显存可以开（你的 `use_reentrant=False` 已配好，无需再改）。

### 4.4 训练前基线等价性：结论碰巧成立，但推理依据要修正

L294 注释"B=0 数学上等价 base"——对 LoRA 是 B=0；对 AdaLora 是 **`lora_E` 零初始化**（`reset_lora_parameters` 中 `nn.init.zeros_(lora_E)`），使得 ΔW = B·diag(E)·A = 0。结论不变（训练前评估仍是纯 base），但学习时注意两者的零初始化对象不同。

---

## 五、注释纠偏（不影响运行，建议一并修正认知）

| 位置 | 现注释 | 问题与正确语义 |
|------|--------|---------------|
| L175 | `r = 8` | 直接删除（无效参数），改为 `target_r = 8`（最终平均秩） |
| L176 | `init_r = 12  # 初始化秩…` | ✓ 正确 |
| L177 | `tinit=200 # 迭代前200步保持init_r不变，之后再逐渐衰减到r` | 前半句对；末尾"r"应为 `target_r`；且数值需按第五节配套改小 |
| L178 | `tfinal=1000 # 迭代到1000步时` | ✗ 不是"第 1000 步"。语义：**最后 tfinal 步**为"定秩微调段"（从 `total_step - tfinal` 步起，预算固定为 `target_r` 并强制 mask） |
| L179 | `deltaT=10 # 每隔deltaT步衰减秩一次` | ✓ 基本正确；精确说法是衰减段内每 deltaT 步执行一次 mask（kthvalue→E 置 0），其余步只更新重要性分数 |
| L180 | `lora_alpha=16 # 一般为2r` | ✗ AdaLora 的缩放是 `delta × scaling / ranknum`，其中 `scaling=lora_alpha` 固定、`ranknum=init_r`（不随 mask 变化），与 LoRA 的 `alpha/r` 语义不同；16 也不是 2×12 |

---

## 六、AdaLora 原理速览（对照 peft 0.18.1 实现）

帮助理解上面的修改为什么是这些：

1. **三因子参数化**：ΔW = B·diag(E)·A（论文记号 PΛQᵀ）。三组可训练参数：`lora_E`（r×1，奇异值）、`lora_A`（r×in）、`lora_B`（out×r）。
2. **三阶段预算调度**（`RankAllocator.budget_schedule`，单位是优化器步）：
   - `step ≤ tinit`：热身，总预算 = `init_bgt`（不 mask）；
   - `tinit < step ≤ total_step - tfinal`：**立方调度**衰减 `budget = (init_bgt - target_bgt)·(1-(step-tinit)/(total_step-tfinal-tinit))³ + target_bgt`，且仅当 `step % deltaT == 0` 才真正 mask；
   - `step == total_step - tfinal`：**切换点**，`force_mask=True` 强制 mask 到当前预算，并 `reset_ipt()` 清空重要性统计；
   - `step > total_step - tfinal`：定秩微调段，预算固定 = `target_bgt`，每步按已存的 `rank_pattern` 重新 mask（弥补每步 optimizer 更新对 mask 的扰动）。
3. **budget 是"全模型奇异值分量总数"**，不是每层的秩。本脚本：28 层 × 7 个目标模块 = 196 个 SVDLinear → `init_bgt = 196×12 = 2352`，`target_bgt = 196×8 = 1568`。调度就是从 2352 逐渐压到 1568（平均每层砍 4 个奇异值分量）。
4. **重要性打分**：`ipt = |参数 × 梯度|`，EMA 平滑（`beta1`）+ 不确定性量化（`beta2`）；三元组（E 的一个分量 + A 的一行 + B 的一列）联合打分；`torch.kthvalue` 选出最不重要的 `init_bgt - budget` 个三元组置零。
5. **正交正则**：‖AAᵀ−I‖_F + ‖BᵀB−I‖_F，由 `AdaLoraModel.forward` **自动**加进 loss（见 4.1），`orth_reg_weight` 保持默认 0.5 即可，无需手动添加。

---

## 七、修改后的自检清单

1. `python -m py_compile test_AdaLora_qwen3-0.6b.py` 通过（确认逗号）；
2. 首次运行不再抛 `ValueError: AdaLoRA does not work when total_step is None`；
3. 在回调里临时加一行 `print(state.global_step, base.rankallocator.budget_schedule(state.global_step))`，应观察到 budget 从 **2352 逐步降到 1568**（回调步号 0~5 恒为 2352，6 起开始下降，45 为切换点，46 起恒为 1568）；
4. 训练结束后检查 mask 是否生效（lora_E 的零值比例应从 0 升至约 1/3）：
   ```python
   for n, p in model_lora.named_parameters():
       if "lora_E" in n:
           print(n, f"{(p == 0).float().mean().item():.2%}")
   ```
5. 确认 `./training/qwen3-0.6b_AdaLora/` 下产出 checkpoint / logs，不再写入 `_Lora` 目录；
6. 删除旧实验的 `training/qwen3-0.6b_Lora/` 中混入的 AdaLora 产物（如有）；
7. 跑通一次后删除 `datasets/zhihu-kol/cache/` 下的 `.arrow` 缓存再重跑一次，确认没有陈旧缓存干扰（仓库已知坑 #1）。

---

## 八、实战复盘：复用训练成果分支（加载模式）崩溃分析与修复

> 前七节是首训前的审查；本节是**首训成功后**第二次运行（`has_adapter=True` 走加载分支）的真实崩溃。
> 全部结论已在本地用 CPU + 真实适配器复现并验证修复（复现脚本见 8.8）。

### 8.1 现象

```
检测到已保存的LoRA适配器：./training/qwen3-0.6b_AdaLora/lora_adapter，跳过训练
trainable params: 5,039,844 || all params: 601,089,764 || trainable%: 0.8385
==== 基线评估（disable_adapter 临时关闭适配器 ≙ base 模型）====
AttributeError: 'Qwen3ForCausalLM' object has no attribute 'trainable_adapter_name'
```

两个看似矛盾的线索：
- ① `print_trainable_parameters` 显示 **0.8385%** 可训练（适配器参数没被冻结——但脚本注释写的是"默认 is_trainable=False……显示 0% 是预期的"）；
- ② `trainable_adapter_name` 属性不存在（说明走的是 `inference_mode=True` 分支）。

先解决直接死因 ②，再解释谜团 ①。

### 8.2 崩溃点定位

报错最后一行的类名是 `Qwen3ForCausalLM` 而不是 `AdaLoraModel`，这是 `BaseTuner.__getattr__`（tuners_utils.py L1239-1243）的兜底链造成的：AdaLoraModel 上找不到属性 → 转发给内层 `self.model`（Qwen3）→ 还是没有 → 抛错。**真正访问缺失属性的位置**是 `peft/tuners/adalora/model.py` L230（`AdaLoraModel.forward` 的正交正则段）：

```python
def forward(self, *args, **kwargs):
    outputs = self.model.forward(*args, **kwargs)
    if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
        # ↓ 只要 forward 返回 loss（= 传了 labels）就无条件执行
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight   # ← L230 崩溃
```

而 `trainable_adapter_name` 只在 `AdaLoraModel.__init__` 的 **else 分支**里设置：

```python
if self.peft_config[adapter_name].inference_mode:      # PeftModel.from_pretrained 默认走这里
    _freeze_adapter(self.model, adapter_name)          #   → 只冻结，不设置该属性
else:                                                   # 训练模式（get_peft_model）走这里
    self.trainable_adapter_name = adapter_name          #   → 属性在这里才存在
    self.rankallocator = RankAllocator(...)
```

### 8.3 根因链（peft 0.18.1 的缺陷叠加）

1. `PeftModel.from_pretrained(model, path)` 默认 `is_trainable=False` → `config.inference_mode=True`（peft_model.py L525：`config.inference_mode = not is_trainable`；且磁盘上的 `adapter_config.json` 本来就存着 `inference_mode: true`，因为 `save_pretrained` 写盘瞬间会强制置 True）；
2. `AdaLoraModel.__init__` 走 if 分支 → `trainable_adapter_name` **从未被设置**；
3. `trainer.evaluate()` 的前向传了 labels → `outputs.loss` 非 None → 正则段无条件访问 `self.trainable_adapter_name` → AttributeError。

崩溃三要素：**纯加载模式（is_trainable=False）+ AdaLora + 带 labels 的 forward**。这也解释了：
- **首训为什么不炸**：`get_peft_model` 走 else 分支，属性存在；
- **`test_Lora-load_bert.py` 为什么不炸**：`LoraModel.forward` 没有正交正则段，这是 **AdaLora 特有**的问题——从 LoRA 加载脚本复制来的加载分支意识不到这个坑。

### 8.4 谜团 ①：为什么显示 0.8385% 而不是注释承诺的 0%

这是 AdaLora 与 LoRA 加载行为的又一个差异，触发器是**训练保留下来的 `rank_pattern`**。加载路径 `peft/utils/save_and_load.py` L497-501：

```python
if config.peft_type == PeftType.ADALORA:
    rank_pattern = config.rank_pattern
    if rank_pattern is not None:
        model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
```

`resize_modules_by_rank_pattern` 内部对每个模块调用 `update_layer(adapter_name, rank, ...)` **重建 `lora_A/lora_B/lora_E/ranknum` 为全新 `nn.Parameter`**（默认 `requires_grad=True`）。时序上：

```
__init__: inject 创建参数 → _freeze_adapter 冻结全部     ← 冻结发生在这
from_pretrained 继续: load_adapter → resize 重建参数      ← 重建发生在这之后，逃过了冻结
```

实测数字完全对上：用本仓库真实适配器 CPU 加载，可训练参数 = **5,039,648**（与报错输出的 5,039,844 仅差 196 = `ranknum` 总数 28 层 × 7 模块，两次保存的 rank_pattern 细节差异所致）。而用 rank_pattern=null 的空白适配器加载，可训练参数 = **0**（冻结正常生效）——差异唯一来源就是 rank_pattern 触发的重建。

顺带修正 4.2 节的原有表述：**保存/加载时形状是真正缩小的**（保存端 L119 截断 state_dict、加载端 L502 缩形重建），不是"按 init_r 原形重建"。

### 8.5 为什么两条"绕开"思路都行不通

- **`with model_lora.disable_adapter():` 救不了**：disable 只关掉 SVDLinear 内部的 adapter 旁路（走 base_layer），前向链仍然经过 `AdaLoraModel.forward` 的正则段——实测修复前后在 disable 上下文里 forward 带 labels 的行为完全一样（修复前崩、修复后通）。
- **`orth_reg_weight=0` 也关不掉**：正则段第一行就是
  ```python
  if orth_reg_weight <= 0:
      raise ValueError("orth_reg_weight should be greater than 0. ")
  ```
  peft 设计上正则不可关闭。

### 8.6 修复方案（已实验验证）

加载分支改一个参数：

```python
if has_adapter:
    print(f"检测到已保存的LoRA适配器：{lora_save_path}，跳过训练")
    # ★ is_trainable=True：让 AdaLoraModel 走"训练模式"初始化分支，
    #   设置 trainable_adapter_name（并创建 RankAllocator）——否则 forward 带 labels
    #   时正交正则段访问该属性直接 AttributeError（peft 0.18.1 的 AdaLora 特有缺陷）
    model_lora = PeftModel.from_pretrained(model, lora_save_path, is_trainable=True)
    # 注意：此时 print_trainable_parameters 显示 ~0.8% 而非 0%（rank_pattern 缩形重建的
    # 新参数逃过了 _freeze_adapter，见 8.4）——无害，本分支不会调用 trainer.train()，
    # evaluate/generate 都在 eval()/no_grad 下执行，不会发生任何参数更新
    model_lora.print_trainable_parameters()
```

实验证据（CPU + 本仓库真实适配器）：

| 场景 | trainable_adapter_name | forward 带 labels |
|---|---|---|
| 默认 `is_trainable=False`（空白适配器） | ✗ | 崩溃（loss 段） |
| 默认 `is_trainable=False`（真实适配器） | ✗ | 崩溃（= 本次报错） |
| **`is_trainable=True`（真实适配器）** | ✓ | **成功**（普通与 disable_adapter 上下文均成功） |

两点副作用说明：
- `is_trainable=True` 会让 `RankAllocator` 一并被创建——无害：它只在构造时数参数、设预算，不碰梯度；`adapter_config.json` 里已保存 `total_step=60`，config 校验也能通过。本分支不会再调用 `update_and_allocate`（只有 `trainer.train()` 触发回调）；
- 原注释"默认 is_trainable=False……显示 0% 是预期的"两处均已失效，需按上面新注释一并修改（0% 不成立的原因见 8.4）。

### 8.7 修复后的验证清单

1. 重新运行脚本，加载分支不再抛 `AttributeError`，能打印基线 `eval_loss`；
2. 加载分支的 `print_trainable_parameters` 显示 ~0.8%（5,039,xxx）属预期（8.4 解释了为什么不是 0）；
3. 基线/微调后两次 `trainer.evaluate()` 均正常完成；
4. 生成对比与 ROUGE 输出正常。

### 8.8 本次分析用的复现/验证实验

```python
# CPU 快速复现（不占 GPU）：构造最小 AdaLora → 保存 → 分别以两种模式加载
import torch
from transformers import AutoModelForCausalLM
from peft import AdaLoraConfig, TaskType, get_peft_model, PeftModel

m = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B", local_files_only=True)
cfg = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj"],
                    init_r=4, target_r=2, total_step=10, tinit=2, tfinal=2, deltaT=1)
get_peft_model(m, cfg).save_pretrained("/tmp/adalora_repro_adapter")

ids = torch.tensor([[1, 2, 3, 4, 5]])
for tag, kw in [("默认", {}), ("is_trainable=True", {"is_trainable": True})]:
    m2 = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B", local_files_only=True)
    loaded = PeftModel.from_pretrained(m2, "/tmp/adalora_repro_adapter", **kw)
    print(tag, "有属性:", hasattr(loaded.base_model, "trainable_adapter_name"))
    try:
        print(tag, "loss =", loaded(input_ids=ids, labels=ids).loss.item())
    except AttributeError as e:
        print(tag, "崩溃:", e)
```

再换成真实路径 `./training/qwen3-0.6b_AdaLora/lora_adapter` 重复加载，即可复现 5,039,648 trainable / 崩溃 / 修复三种现象。

---



- 语法验证：`python -m py_compile`（错误 1 实证）；
- peft 行为核对：`ai-gpu` 环境 `site-packages/peft/tuners/adalora/{config,model,layer}.py`（0.18.1）；
- Trainer 回调时序核对：`site-packages/transformers/trainer.py` L2639-2663、`trainer_callback.py` L387-395（事件 docstring）与 L554-562（`call_event` 注入 `model=` 等关键字参数）（4.55.0）；
- 模型结构核对：`model/Qwen3-0.6B/config.json`（`num_hidden_layers=28` → 28×7=196 个 SVDLinear）；
- 第八节崩溃分析：`peft/peft_model.py` L525（inference_mode 赋值）、`peft/utils/save_and_load.py` L119/L497-501（rank_pattern 保存截断与加载重建）源码核对 + CPU 实测复现（空白适配器/真实适配器 × 默认/is_trainable=True 共四组，结论见 8.6 表格）；
- 数据管线、custom_collate_fn、generate 对比、ROUGE 部分与 `test_Lora_qwen3-0.6b.py` diff 比对，未发现新引入的问题，本文不再重复。
