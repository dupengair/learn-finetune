# AdaLora 微调 BERT 改造方案（test_AdaLora_bert.py）

> 前提：`test_AdaLora_bert.py` 目前是 `test_Lora_bert.py` 的原样复制（仍是 `LoraConfig`），尚未做任何 AdaLora 改造。
> 本文给出完整修改方案，**未改动任何源码**，请对照代码块手动修改。
> AdaLora 的通用原理、回调时序、加载分支坑等已在《AdaLora微调代码审查报告.md》详述，本文不重复，只讲 BERT 侧的差异与具体改法。
> 环境：peft 0.18.1 / transformers 4.55.0（`ai-gpu`），关键结论均经源码核对或 CPU 实验验证（见附录）。

## 一、修改清单总览

| # | 修改 | 位置 | 级别 |
|---|------|------|------|
| 1 | `LoraConfig` → `AdaLoraConfig`，重写配置参数（删 `r`、加 `target_r/init_r/total_step/tinit/tfinal/deltaT`） | L1-6、L133-145 | 🔴 阻塞 |
| 2 | 新增 `AdaLoraBudgetCallback` 并传给训练 Trainer——**没有它就不是 AdaLora** | Trainer 构造处 | 🔴 核心 |
| 3 | `modules_to_save=["classifier"]` **保留**（BERT 侧必需，与 Qwen 侧结论相反） | L141 | ✅ 不动 |
| 4 | 4 处目录名 `bert-base-uncased_Lora` → `bert-base-uncased_AdaLora` | L111、L150、L151、L204 | 🟡 防混写 |
| 5 | `load_best_model_at_end=True` **必须改为 False**：训练结束后 `_load_best_model` 加载 best checkpoint 会大面积 size mismatch（首次实测崩溃，详见第六节） | L167 | 🔴 阻塞 |
| 6 | 认知项：`eval_loss` 含正交正则，别用于 `metric_for_best_model` | L169 | ⚪ 说明 |
| 7 | 添加"复用已训练成果"分支（`has_adapter` 判断，仿 Qwen 侧），含 BERT 特有差异 | 多处 | 🟢 功能（第五节） |

---

## 二、逐项修改

### 2.1 import（L1-6）

```python
from peft import (
    AdaLoraConfig,     # ← 替换 LoraConfig
    TaskType,
    get_peft_model,
)
```

### 2.2 配置改造（L133-145）

**先算总步数**（`total_step` 必填，且必须等于真实训练步数）：
- MRPC train = 3668 条（本地实测），`per_device_train_batch_size=4`、无梯度累积（默认 1）；
- 每 epoch = ⌈3668/4⌉ = **917 步**，`num_train_epochs=3` → 总步数 = **2751**。

```python
peft_config = AdaLoraConfig(
    task_type = TaskType.SEQ_CLS,      # 序列分类，不是seq2seq
    inference_mode = False,
    target_modules=["query","key","value"],
    bias = "all",                      # 与 LoRA 版一致：打开所有bias参与训练
    modules_to_save=["classifier"],    # ★ BERT侧必须保留！分类头不参与秩调度（不是SVDLinear），
                                       #   但不存它的话保存的适配器里就没有分类头，加载后无法分类
    # ================= AdaLora 专属参数 =================
    target_r = 8,        # 最终平均秩（替代 LoRA 的 r）
    init_r   = 12,       # 初始秩：BERT 12层×q/k/v = 36个SVDLinear，初始总预算 36×12=432
    total_step = 2751,   # ★ 必填 = 917步/epoch × 3 epochs；不填直接 ValueError
    tinit    = 250,      # 热身段：前250步（约0.27个epoch）预算不动，攒重要性统计
    tfinal   = 500,      # 定秩段：第 total_step−tfinal=2251 步起切换（约最后0.55个epoch）
    deltaT   = 10,       # 衰减段（251~2250步，约2000步）内每10步mask一次，共约200次，
                         #   平均每次砍 (432−288)/200 ≈ 0.7 个三元组，节奏平缓
    lora_alpha = 16,
    lora_dropout = 0.1,
)
```

校验关系（peft 源码强制）：`tinit < total_step − tfinal`，即 250 < 2251 ✓。最终总预算 = 36×8 = **288**（从 432 衰减到 288，平均每层砍 4 个奇异值分量）。

原配置里的 `r = 8` **删除**——AdaLora 中 `r` 完全无效（控制最终秩的是 `target_r`），且 `r=8` 恰好等于默认值连警告都不触发。

### 2.3 秩预算调度回调（最核心的新增）

与 Qwen 侧完全一致的机制：peft **不会自动**做秩调度，必须每个 optimizer step 手动调用 `update_and_allocate`。加在第二个 Trainer 构造之前：

```python
from transformers import TrainerCallback

class AdaLoraBudgetCallback(TrainerCallback):
    """AdaLora 秩预算调度必须每步手动触发。
    时机选 on_optimizer_step（optimizer.step 之后、zero_grad 之前），
    与 peft 官方示例位置一一对应：
      backward → optimizer.step() → update_and_allocate → zero_grad
    若用常见的 on_step_end：transformers 4.55 中 zero_grad 在其之前执行，
    p.grad=None，update_ipt 读梯度直接 TypeError。"""
    def on_optimizer_step(self, args, state, control, **kwargs):
        kwargs["model"].base_model.update_and_allocate(state.global_step)

trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[AdaLoraBudgetCallback()],   # ← 新增
)
```

注意：`trainer_baseline`（L117-125，基线评估用）**不要**传这个回调——它持有的是未包装的原始 model，而且 `do_train=False` 根本不会触发 `on_optimizer_step`，传了无害但没必要。

### 2.4 目录改名（4 处）

| 行 | 原值 | 改为 |
|---|------|------|
| L111 | `./training/bert-base-uncased_Lora/output` | `./training/bert-base-uncased_AdaLora/output` |
| L150 | `./training/bert-base-uncased_Lora/output` | `./training/bert-base-uncased_AdaLora/output` |
| L151 | `./training/bert-base-uncased_Lora/logs` | `./training/bert-base-uncased_AdaLora/logs` |
| L204 | `./training/bert-base-uncased_Lora/lora_adapter` | `./training/bert-base-uncased_AdaLora/lora_adapter` |

不改的话 AdaLora 的 checkpoint / TensorBoard / 适配器会全部混进 LoRA 实验的目录。

---

## 三、BERT 侧与 Qwen 侧的差异点（避坑对照）

### 3.1 `modules_to_save=["classifier"]`：这次必须保留

Qwen 审查报告里我们**删掉**了它（CausalLM 没有 classifier，是 BERT 脚本复制残留）；BERT SEQ_CLS 正相反，**必须保留**——分类头是最终输出层，不进 `modules_to_save` 就会被冻结且不随适配器保存，加载适配器后等于用随机分类头预测。它也不参与秩调度（只有 `target_modules` 命中的 Linear 才会被替换成 SVDLinear），两者互不干扰。

### 3.2 基线评估结构：BERT 脚本天然无坑，不用动

`trainer_baseline` 在 `get_peft_model` **之前**用原始 model 评估（L117-130）——此时模型还没被包装成 PeftModel，前向不经过 `AdaLoraModel.forward` 的正则段，不存在 Qwen 侧 `disable_adapter` 那些问题。这个结构保持原样即可。

### 3.3 `eval_loss` 含正交正则：`metric_for_best_model` 别碰 loss

`AdaLoraModel.forward` 会自动把 `0.5 × mean(‖AAᵀ−I‖_F + ‖BᵀB−I‖_F)` 加进 loss（带 labels 的前向都会加，评估也一样）。对 BERT 侧的影响：
- `compute_metrics` 的 accuracy / F1 / macro_f1 用 **logits（argmax）** 计算，**不受影响** ✓；
- `eval_loss` 数值与 LoRA 实验不可比（Qwen 报告 4.1 节同样问题）；
- 当前脚本 `metric_for_best_model="macro_f1"`（L169）**正好正确**，千万别图省事改成 `eval_loss`——那会用含正则的失真 loss 挑最佳 checkpoint。

### 3.4 `load_best_model_at_end=True` 在 AdaLora 下**不兼容，必须关闭**（首次实测崩溃，详见第六节）

这是 BERT 脚本特有的配置（Qwen 侧没有）。机制（peft 0.18.1 + transformers 4.55 源码）：

- 每个 epoch 保存 checkpoint 时，adapter 权重按**当时**的 `peft_config.rank_pattern` 截断，`adapter_config.json` 也存**当时**的 pattern（两者自洽）；
- 但 `train()` 结束时 `_load_best_model` 调 `model.load_adapter(ckpt, "default")`，而 peft 的 `load_adapter` 对**已存在**的 adapter 名**不读 checkpoint 的 config**（`peft_model.py` L1360：`if adapter_name not in self.peft_config:` 才读），缩形重建用的是**内存中**的 rank_pattern——即**训练最后一步**的；
- best checkpoint 几乎必然不是最后一个 epoch（`macro_f1` 最高的是较早轮次），其权重形状（保存时的 pattern）与内存 pattern（终态 pattern）不匹配 → **大量 size mismatch，训练白跑**。

修改（L167）：

```python
    # load_best_model_at_end=True,          # ← 删除/注释：AdaLora 下必崩（见第六节）
    load_best_model_at_end=False,           # ★ 最终模型 = 最后一步（定秩段微调完的终态结构）
```

`metric_for_best_model="macro_f1"` 可保留：它仍驱动 `state.best_model_checkpoint` 的记录（`save_total_limit` 用它保护最佳 checkpoint 不被轮转删除，也供第六节方案 B 手动加载用），只是不再触发训练结束后的自动回滚。若确实想要 best epoch 的模型，用第六节方案 B 手动 `PeftModel.from_pretrained` 加载（fresh 模型会读 checkpoint 的 config，pattern 匹配）。

### 3.5 `bias="all"` 与 `rank_pattern` 键格式：两个小认知点

- `bias="all"` 打开的所有 bias 参数名不含 `lora_` 前缀 → **不参与**重要性打分（`update_ipt` 只看 `lora_` 参数）也不参与正交正则（只看 `lora_A/lora_B`），与 AdaLora 调度无冲突，保留即可；
- `rank_pattern`（存进 `adapter_config.json`）的键是**内层模型**的参数名（如 `model.encoder.layer.0.attention.self.query.lora_E`），**不含** `base_model.model.` 前缀。手工构造或调试检查时写错前缀，加载时会报 `KeyError`（附录 B 实验第一个版本就踩了这个）。

### 3.6 预期参数量变化（看到数字变大不要慌）

`print_trainable_parameters` 对比 LoRA 版：

| | LoRA (r=8) | AdaLora (init_r=12) |
|---|---|---|
| adapter 参数（36 模块） | 36×(8×768+768×8+8) ≈ 44.6 万 | 36×(12×768+768×12+12) ≈ 66.4 万 |
| bias="all" + classifier | ≈ 12.5 万 | ≈ 12.5 万（不变） |
| 合计 | ≈ 57 万 | ≈ 79 万 |

AdaLora 启动时按 `init_r=12` 铺满，**训练中参数量不变**（mask 不缩形），保存/加载后才真正缩到平均秩 8。数字比 LoRA 大是 `init_r > r` 的预期结果，不是配置错误。

---

## 四、修改后自检清单

1. `python -m py_compile test_AdaLora_bert.py` 通过（注意配置各项之间逗号，Qwen 侧就漏过一个）；
2. 首次运行不抛 `ValueError: AdaLoRA does not work when total_step is None`；
3. 回调里临时加 `print(state.global_step, kwargs["model"].base_model.rankallocator.budget_schedule(state.global_step))`，应看到 budget 从 **432 逐步降到 288**（步号 0~250 恒为 432，251 起开始下降，2251 起恒为 288）；
4. 训练结束后检查 mask 生效（lora_E 零值比例终值约 1/3）：
   ```python
   for n, p in model_lora.named_parameters():
       if "lora_E" in n:
           print(n, f"{(p == 0).float().mean().item():.2%}")
   ```
5. **`load_best_model_at_end=False` 已设置**（第六节教训：True 时训练虽能跑完，但 `train()` 末尾加载 best checkpoint 必崩 size mismatch，适配器保存代码根本执行不到）；
6. `./training/bert-base-uncased_AdaLora/` 下产出 output/logs/lora_adapter，`_Lora` 目录不再新增文件；
7. 最终 `macro_f1` 与 LoRA 实验（同数据同轮数）对比：AdaLora 论文的卖点是"同参数预算下更优"，理想观察是 init_r=12 衰减到 target_r=8 后效果 ≥ LoRA r=8（生成式任务上差异明显，MRPC 这种小数据集差异可能不显著，属正常）；
8. **无需删除** `datasets/glue/cache-mrpc/` 的 arrow 缓存——本次改造不动 `tokenize_func` 与 max_length（与 Qwen 侧改造不同，那边也没动，只是提醒这个已知坑的适用边界）。

## 五、复用已训练成果分支（has_adapter）实现方案

仿照 `test_AdaLora_qwen3-0.6b.py` 的两模式结构：检测到已保存适配器则加载并跳过训练，否则完整训练。共 5 处改动，**其中 2 处不能照抄 Qwen**（5.3 节）。

### 5.0 前置条件（当前不满足，先做这一步）

`./training/bert-base-uncased_AdaLora/lora_adapter/` **目前不存在**——上次训练崩在 `_load_best_model`（第六节），保存代码没执行到。二选一：
- **重训**（推荐）：按第六节修好 `load_best_model_at_end=False` 后完整跑一次（约 4 分半），训练正常结束会自动保存 `lora_adapter/`；
- **挽救**：用 6.6 节的两行代码从已落盘的 `checkpoint-2751` 直接生成 `lora_adapter/`。

`adapter_config.json` 存在性是分支的判据，这个文件就位前 `has_adapter` 恒为 False。

### 5.1 改动一：import（两处）

```python
from peft import (
    AdaLoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,        # 新增：加载已保存的适配器
)
# ...
import random, torch, os       # ← 追加 os（has_adapter 检测要用）
```

### 5.2 改动二~五：分支结构

**(2) 在 `peft_config = AdaLoraConfig(...)` 之前**（即原 `trainer_baseline` 评估之后）定义路径与判据：

```python
# ===================== 加载 =====================
lora_save_path = "./training/bert-base-uncased_AdaLora/lora_adapter"
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))
```

**(3) 把原来的 `peft_config = ...` + `get_peft_model` + `print_trainable_parameters` 整体包进 `else:`，并新增加载分支**（原有配置内容原样缩进，不用改任何参数）：

```python
if has_adapter:
    # ---------- 模式一：加载已训练权重，跳过训练 ----------
    print(f"检测到已保存的AdaLora适配器：{lora_save_path}，跳过训练")
    # ★ is_trainable=True 必须：evaluate/predict 的前向都带 labels，
    #   AdaLoraModel.forward 正则段访问 trainable_adapter_name（默认 False 时该属性
    #   不存在 → AttributeError，Qwen 报告第八节的坑在 BERT 侧同样成立）
    model_lora = PeftModel.from_pretrained(model, lora_save_path, is_trainable=True)
    # 此时 print 显示 ~0.5%（55万上下）而非 0%——rank_pattern 缩形重建的新参数
    # 逃过了 _freeze_adapter（Qwen 报告 8.4）。无害：本分支不会 trainer.train()，
    # evaluate/predict 都在 no_grad 下执行，不会发生任何参数更新
    model_lora.print_trainable_parameters()
else:
    # ---------- 模式二：首次运行，完整训练流程 ----------
    peft_config = AdaLoraConfig(          # ← 原有配置原样搬进来
        task_type=TaskType.SEQ_CLS,
        # ...（2.2 节的完整配置，参数不动）
    )
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
```

**(4) 训练判断**（原 `trainer.train()` 一行）：

```python
if not has_adapter:
    trainer.train()
```

**(5) 保存判断**（文末 `trainer.save_model(lora_save_path)` 一行）：

```python
if not has_adapter:
    # 保存AdaLora权重、adapter配置（含 rank_pattern），不保存BERT主干
    trainer.save_model(lora_save_path)
    print(f"AdaLora适配器已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")
```

Trainer（含 `AdaLoraBudgetCallback`）、`training_args`、最终 `evaluate()`、`predict()` 与混淆矩阵**都不用动**——两种模式下 `model` 参数都是 `model_lora`；加载模式不调 `train()`，回调永不触发。

### 5.3 与 Qwen 实现的三点差异（不能照抄的地方）

1. **基线评估不用动、也不需要 `disable_adapter`**。Qwen 侧把基线评估放在 peft 包装**之后**，所以要靠 `disable_adapter()` 上下文临时关适配器；BERT 脚本的 `trainer_baseline.evaluate()` 本来就在包装**之前**用原始 model 跑（结构更干净），两种模式下都是纯 base 前向，原样保留即可——这是 BERT 侧唯一"不需要仿照"的部分。代价只是加载模式多跑 1 秒多的基线评估，换来的是两种模式基线口径完全一致。
2. **`is_trainable=True` 同样必需，但触发面更广**：Qwen 侧只有 `evaluate()` 带 labels；BERT 侧 `evaluate()` 和 `predict()` 都带 labels（`compute_metrics`/混淆矩阵需要 `label_ids`），忘了这个参数两处都会崩。
3. **`num_labels=2` 是加载端的生命线**：脚本加载 base model 处的 `num_labels=2`（L36）保证分类头形状与适配器里 `modules_to_save` 保存的 classifier 权重匹配，改了就 size mismatch（CLAUDE.md 已知坑 5 的 AdaLora 版）。

另有两个加载模式下的预期现象（不是 bug）：
- `print_trainable_parameters` 显示约 **0.5%（55 万上下）**：构成 ≈ adapter 44.6 万（36 模块 × 平均秩 8）+ `bias="all"` 约 10.3 万 + classifier 1,538——比训练启动时（init_r=12，约 79 万）**小**，因为加载的是缩形后的终态结构；
- 最终 `eval_loss` 与基线 `eval_loss` 不直接可比（基线是纯 CE，微调后含正交正则），看 `macro_f1`/`accuracy`/混淆矩阵的对比。

### 5.4 验证清单

1. 按 5.0 就位 `lora_adapter/` 后重跑：日志出现"检测到已保存的AdaLora适配器…跳过训练"，无训练进度条；
2. 加载分支 `print_trainable_parameters` 约 0.5%（若显示 0% 或报 `AttributeError: trainable_adapter_name` → `is_trainable=True` 没加）；
3. `evaluate()` 正常输出，`macro_f1` 与训练结束时最后一轮的值一致（加载的就是保存的终态权重）；
4. `predict()` 与混淆矩阵正常输出；
5. 末尾打印"已训练权重来自…（无需重复保存）"，`lora_adapter/` 文件时间戳未变。

---

## 六、实战复盘：训练结束后大量 size mismatch 崩溃（已实测）

> 按 2.1~2.4 完成改造并保留 `load_best_model_at_end=True` 首次训练的真实结果：2751 步全部跑完、
> 每 epoch 指标正常（末轮 macro_f1≈0.799），但 `trainer.train()` 末尾崩溃，适配器未保存。

### 6.1 现象

```
100%|...| 2751/2751 [04:33<00:00]   ← 训练本身成功完成
  File "trainer.py", line 2728, in _inner_training_loop
    self._load_best_model()
  File "trainer.py", line 3015, in _load_best_model
    model.load_adapter(self.state.best_model_checkpoint, active_adapter)
  File "save_and_load.py", line 565, in set_peft_model_state_dict
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
RuntimeError: Error(s) in loading state_dict for PeftModelForSequenceClassification:
    size mismatch for ...query.lora_A.default: copying a param with shape [5, 768],
        the shape in current model is [4, 768].
    size mismatch for ...key.lora_A.default: copying a param with shape [4, 768],
        the shape in current model is [5, 768].
    ...（36 个模块全部 mismatch）
```

### 6.2 报错里藏着的两条关键证据

1. **current 形状是 4/5/10，不是 `init_r=12`**——说明 `resize_modules_by_rank_pattern` **确实执行了**（模块被重建过），不是"没缩形导致 12 对不上"；
2. **checkpoint 形状（5/4/8）与 current 形状（4/5/10）逐模块不同，但都在 4~10 区间（平均≈8=target_r）**——说明两边用的是**两个不同训练时刻的 rank_pattern**：一个在衰减中段，一个已是定秩终态。

### 6.3 根因链（五个环节叠加）

1. 回调每步调 `update_and_allocate` → **持续改写内存中的** `peft_config["default"].rank_pattern`（衰减段每 `deltaT` 步一变；第 2251 步起固定为定秩 pattern）；
2. 每个 epoch 结束（第 917/1834/2751 步）保存 checkpoint：state_dict 按**当时** pattern 截断，`adapter_config.json` 也存**当时** pattern——单看这个 checkpoint，权重与 config 自洽；
3. `train()` 末尾 `_load_best_model` 调 `model.load_adapter(best_ckpt, "default")`；
4. peft 的 `load_adapter`（`peft_model.py` L1360-1361）：

   ```python
   if adapter_name not in self.peft_config:   # "default" 已存在 → 整个跳过
       peft_config = ...(model_id)            #   ← 只有为"新" adapter 才读 checkpoint 的 config
       self.add_adapter(adapter_name, peft_config, ...)
   ```

   已存在的 adapter **不读 checkpoint 的 `adapter_config.json`**，后续 `set_peft_model_state_dict` 里的 `resize_modules_by_rank_pattern` 用的 rank_pattern 来自**内存**（= 训练最后一步的定秩 pattern）→ 模块被重建成 4/5/10；
5. best checkpoint 是 `macro_f1` 最高的**较早 epoch**（本例不是最后一轮），其权重按当时的 pattern（5/4/8）截断 → 逐一 mismatch → `load_state_dict` 抛 RuntimeError（size mismatch 与 `strict=False` 无关，形状不匹配必报）。

一句话：**保存时用"当时的 pattern"，加载时用"现在的 pattern"，而 AdaLora 的 pattern 随训练演化，两次不一样。**

### 6.4 为什么 Qwen 侧与本文附录 B 实验都没暴露

- Qwen 脚本的 `TrainingArguments` 没设 `load_best_model_at_end`（默认 False），不触发这条路径；
- 附录 B 的验证实验里，`rank_pattern` 设置一次后**从未变化**（保存时 = 加载时），漏掉了"pattern 随调度演化 + best 是中途 epoch"这两个真实训练才有的维度。该实验证明的"`load_adapter`→resize 链路本身能走通"仍然成立（本次报错恰恰说明 resize 执行了），但"训练场景下兼容"是**错误推论**，3.4 节已修正。

### 6.5 修复方案

**方案 A（推荐，改一行）**——见 3.4 节：`load_best_model_at_end=False`。

- 训练后的 `trainer.evaluate()` / `trainer.predict()` / `trainer.save_model()` 全部自洽：内存 pattern = 定秩 pattern，保存时 state_dict 与 config 用同一个 pattern；
- 语义上反而更正确：AdaLora 的 `tfinal` 定秩段本来就是"在最终秩结构上精调"，最后一步的模型才是论文意义上的成品；衰减中途 epoch 的"最佳"是在**临时秩结构**上取得的，回滚到它反而可疑；
- 每轮指标照常打印（`eval_strategy="epoch"`），哪轮最好看日志即可。

**方案 B（确实需要 best epoch 模型时）**——`load_best_model_at_end=False` 前提下，训练后手动加载：

```python
from peft import PeftModel

best_ckpt = trainer.state.best_model_checkpoint   # metric_for_best_model 仍驱动此记录（trainer.py L3248）
print("最佳 checkpoint：", best_ckpt)
# 重新加载一份干净的 base model（不能用训练中的 model 对象，它内部结构已被改）
base2 = AutoModelForSequenceClassification.from_pretrained(
    "./model/bert-base-uncased", local_files_only=True, num_labels=2,
    torch_dtype="auto", device_map="auto")
# ★ 关键区别：fresh PeftModel 的 "default" 不存在 → from_pretrained 会读 checkpoint 的
#   adapter_config.json → resize 用 checkpoint 自己的 rank_pattern → 形状必然匹配
model_best = PeftModel.from_pretrained(base2, best_ckpt, is_trainable=True)
```

为什么 `from_pretrained` 行、`load_adapter` 不行：差别只在第 6.3 节第 4 步那个 `if`——fresh 模型的 adapter 名不存在，走"读 config"分支，用 **checkpoint 的** pattern 缩形（Qwen 报告第八节实验验证的正是这条路径，含截断权重加载成功）。`is_trainable=True` 同时规避第八节的 `trainable_adapter_name` 崩溃。注意 `model_best` 是新对象，后续评估要围绕它新构造 Trainer（或手写循环），原来的 `trainer` 仍持有旧模型。

### 6.6 本次训练成果不必作废（已核实）

崩溃发生在 `train()` 末尾的 `_load_best_model`，但 `save_strategy="epoch"` 的三个 checkpoint（checkpoint-917 / 1834 / 2751）**已全部落盘**。已核实 `checkpoint-2751`（最后一个 epoch = 定秩终态）：`adapter_config.json` 里 rank_pattern 覆盖全部 36 个模块、平均秩恰为 8.00（= target_r），权重与 config 自洽。修复脚本后**无需重训**，直接把最终成果取回来：

```python
# 等价于"训练正常结束 + save_model"的产物：
model_lora = PeftModel.from_pretrained(
    model, "./training/bert-base-uncased_AdaLora/output/checkpoint-2751", is_trainable=True)
model_lora.save_model("./training/bert-base-uncased_AdaLora/lora_adapter")
```

（当然整个训练才 4 分半，重跑一遍验证修复也无妨。）

### 6.7 修复后自检

1. 重跑训练，`train()` 结束后**不再抛 size mismatch**，直接进入最终 `evaluate()`；
2. `save_model` 正常产出 `lora_adapter/`（含截断后的适配器权重与配套 config）；
3. （可选，验证闭环）另开加载脚本 `PeftModel.from_pretrained(干净base, lora_save_path, is_trainable=True)` + 一次带 labels 的前向，能正常出 loss 即闭环（参考 Qwen 报告第八节 8.8 的实验代码）。

## 附录 A：本文结论的验证方式

- MRPC 数据量 / 步数：本地 `load_dataset('./datasets/glue','mrpc')` 实测（3668/408/1725 → 917 步/epoch → 2751 步）；
- peft 行为（`total_step` 必填、`r` 无效、回调语义、正则只作用 A/B、保存/加载缩形）：`ai-gpu` 环境 `peft/tuners/adalora/{config,model,layer}.py`、`peft/utils/save_and_load.py` L119/L497-501（0.18.1）源码核对，部分已在 Qwen 报告中实验验证；
- `load_adapter` 对已存在 adapter 跳过读 checkpoint config：`peft/peft_model.py` L1360-1361（`if adapter_name not in self.peft_config:`）源码核对——第六节根因第 4 步；
- `_load_best_model` 路径与 `best_model_checkpoint` 记录条件：`transformers/trainer.py` L3015、L3245-3248 源码核对。

## 附录 B：`load_adapter`→resize 链路验证实验（CPU）及其局限

```python
# 模拟：训练结束(peft_config.rank_pattern已设置) -> save_pretrained(checkpoint被截断)
#       -> train()末尾 _load_best_model 实际调用的 model.load_adapter(ckpt, "default")
import torch, warnings
from transformers import AutoModelForCausalLM
from peft import AdaLoraConfig, TaskType, get_peft_model

m = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B", local_files_only=True)
cfg = AdaLoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj"],
                    init_r=12, target_r=6, total_step=10, tinit=2, tfinal=2, deltaT=1)
pm = get_peft_model(m, cfg)

# 模拟训练后 rank_pattern（★键必须用内层参数名，无 base_model.model. 前缀）
rp = {}
for n, p in pm.base_model.model.named_parameters():
    if "lora_E.default" in n:
        rp[n.replace(".lora_E.default", ".lora_E")] = [i % 2 == 0 for i in range(p.shape[0])]
pm.peft_config["default"].rank_pattern = rp

pm.save_pretrained("/tmp/adalora_loadbest_ckpt")          # state_dict 按 rank_pattern 截断保存
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pm.load_adapter("/tmp/adalora_loadbest_ckpt", "default")  # = _load_best_model 的调用方式
print([(n, tuple(p.shape)) for n, p in pm.named_parameters() if "lora_E" in n])
print(pm(input_ids=torch.tensor([[1,2,3,4,5]]), labels=torch.tensor([[1,2,3,4,5]])).loss.item())
```

结果：加载成功，`lora_E` 形状 12→6（真实缩形），forward 带 labels 正常输出 loss。
（用 Qwen CAUSAL_LM 做载体是因为它小、CPU 跑得动；`load_adapter` 的这条链路与任务类型无关，BERT SEQ_CLS 走同一代码路径。第一版实验误用 PeftModel 级参数名构造 rank_pattern，报 `KeyError` 双重前缀键——即 3.5 节第二个认知点的来源。）

**本实验的局限（写本文档时的教训，也是第六节崩溃没有提前暴露的原因）**：实验里 `rank_pattern` 设置一次后**从未变化**——保存与加载时内存/磁盘的 pattern 恒相同，resize 出的形状自然匹配。真实训练中 pattern 随调度持续演化（衰减段每 `deltaT` 步一变），且 best checkpoint 几乎必然是中途 epoch，"保存时的 pattern"与"加载时内存的 pattern"不同——这正是第六节 size mismatch 的两个必要条件。设计复现实验时，**状态在保存与加载之间发生演化**这一维度必须纳入，否则"链路验证通过"推不出"训练场景兼容"。
