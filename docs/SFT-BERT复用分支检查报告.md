# SFT-BERT 复用分支检查报告（test_SFT_bert.py）

> 检查对象：`test_SFT_bert.py`（参考 `test_SFT_qwen3-0.6b.py` 的实现方式，新增复用上次训练成果分支）
> 检查方式：通读全文 + `py_compile` 验证 + 与兄弟脚本（`test_Lora_bert.py`/`test_AdaLora_bert.py`）和 Qwen SFT 复盘（`docs/SFT复用分支与数据避坑改造方案.md`）逐项对照。
> 结论先行：**没有致命问题，语法通过，双模式逻辑正确，可以直接跑**。有 2 处与兄弟脚本/仓库教训的口径不一致建议统一（baseline 指标缺失、`metric_for_best_model` 口径），以及一批命名/注释的正名项。

---

## 一、复用分支逐项核对（核心部分全部正确 ✅）

| 检查项 | 状态 | 说明 |
|---|---|---|
| 判断文件改为 `model.safetensors`（L131） | ✅ | SFT 语义正确（全量权重，非 adapter_config.json） |
| 加载类用 `AutoModelForSequenceClassification`（L140） | ✅ | 分类类，与训练一致；**且无需显式传 `num_labels`**——`save_model` 已把 config（含 num_labels=2）存进目录，from_pretrained 自动读回 |
| `del trainer_baseline, model` 释放旧引用（L138） | ✅ | 机制正确（重载是新增一份权重）；BERT 体量小（0.11B ≈ 0.22GB），双份也无压力，属好习惯 |
| 重载后无需 device_map/.to()（L144 注释） | ✅ | 主 Trainer 构造时自动迁移，与 Qwen SFT 同机制 |
| else 分支 `pass`（L145-147） | ✅ | SFT 无包装步骤 |
| 两处守卫 `if not has_adapter`（L182/L203） | ✅ | 训练守卫 + 保存守卫 |
| 保存分支全量语义（L202-209） | ✅ | `save_model` 存完整权重+config+tokenizer |
| `trainer`/`training_args` 顶格定义（L151/L172） | ✅ | CLAUDE.md 坑 6 已遵守（Lora_bert 的同款坑本脚本没有） |
| `gradient_checkpointing=True`（L166） | ✅ | SFT 非 KV 注入方法，gc 安全 |
| `bf16=True`（L165） | ✅ | 无 Qwen SFT 的 fp16×bf16 冲突（权重 bf16 配 bf16 AMP，混精匹配律） |
| `load_best_model_at_end` + `save_strategy="steps"`（L156/L167） | ✅ | 2751 步的全量训练用 steps/100 合理（与 Qwen SFT 60 步用 epoch 策略的场景不同，两者都对） |
| `learning_rate=2e-5`（L164） | ✅ | 全量微调标准量级（与 adapter 系的 5e-4 区分正确） |

---

## 二、建议修改项（无阻塞，按价值排序）

### 2.1 🟡 baseline Trainer 缺 `compute_metrics`（L114-120）——判别任务的基线指标缺失

当前 baseline Trainer 没传 `compute_metrics` → `baseline_result` 只有 eval_loss，**没有 acc/f1/macro_f1 基线**。兄弟脚本 `test_Lora_bert.py` 的 baseline 是带 `compute_metrics` 的（L125）。对判别任务，"微调前 vs 后的 macro_f1 对比"比 eval_loss 直觉得多（BERT MRPC 的随机分类头基线就是全 1：acc 0.684 / macro_f1 0.406——PTuningV2 报告 7.1 的教训）。

修复：L114-120 的 `trainer_baseline = Trainer(...)` 补一行：

```python
trainer_baseline = Trainer(
    model=model,
    args=training_args_baseline,
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics      # ★ 补上：基线也要有 acc/f1/macro_f1
)
```

### 2.2 🟡 `metric_for_best_model="f1"`（L168）——与兄弟脚本口径不一致，且 f1 在不平衡数据上虚高

`test_Lora_bert.py` / `test_AdaLora_bert.py` 都已改用 **`macro_f1`**（并注释掉了 f1），本脚本还是 `"f1"`（GLUE mrpc 的 f1 = 正类 f1）。MRPC 正类占 ~67%，**正类 f1 对"全预测正类"的退化解很宽容**（全 1 时 f1=0.812 vs macro_f1=0.406，实测教训见 PTuningV2-BERT 报告 7.1）。`load_best_model_at_end` 按它选最优 checkpoint，口径宽了可能选到偏科模型。

修复：L168 改 `metric_for_best_model="macro_f1"`（与两兄弟对齐；`compute_metrics` 已输出该键，L98，直接可用）。

### 2.3 🟡 `device_map="auto"`（L27）与 CLAUDE.md 坑 7 的统一性

Qwen SFT 的教训（坑 7）：`device_map="auto"` 与 Trainer 冲突，全量 SFT 脚本应注释掉（Trainer 自行迁移模型）。本脚本训练模式却保留着它（历史上跑通过——BERT 小、单卡放得下，未触发冲突）。但新增复用分支后出现**双模式行为不一致**：训练模式走 device_map（模型直接上 GPU），加载模式走重载+Trainer 迁移（CPU→GPU）。两条路径都工作，但风格分裂。

建议：与 Qwen SFT 对齐，注释掉 L27（`# device_map="auto"`），让两种模式统一走"加载在 CPU → Trainer 迁移"。不是必须（历史已验证能跑），属一致性收编。

### 2.4 ⚪ 命名与文案正名（与 Qwen SFT 方案同步）

| 位置 | 现状 | 建议 |
|---|---|---|
| L129 | `lora_save_path = ".../lora_adapter"` + 旧注释"← 从文件末尾上移到此处定义" | `sft_save_path = "./training/bert-base-uncased_SFT/full_model"`；**改目录名时 L131/L141/L205/L207/L209 同步** |
| L131 | 变量名 `has_adapter`（查的已是 model.safetensors） | 改 `has_model`；同步 L133/L182/L203 |
| L130 | 注释"adapter_config.json 存在"与代码不符 | 改"model.safetensors 存在（全量权重完成标志）" |
| L138-139 | 注释里的 "~1.2G"/"0.6B模型" 是 Qwen 数字 | BERT-base bf16 ≈ **0.22GB** / 0.11B 参数 |
| L207 | "全量微调**适配器**已保存至" | SFT 没有"适配器"，改"全量微调权重已保存至" |

### 2.5 ⚪ 可选清理

- L46 `# tokenizer = AutoTokenizer.from_pretrained(model_path)` 旧注释；L59 `padding=False` 是默认值（无害）；L33 `print(model)` 冗长可精简。
- 磁盘账：`full_model/` ≈ 0.44GB + output 3 个 checkpoint ≈ 共 ~1.8GB，无压力。

---

## 三、运行自查清单

1. ✅ `py_compile` 已通过；
2. 首次运行：分词秒过（复用 `cache-mrpc/*.arrow`）；基线输出**应含 acc/f1/macro_f1**（补 2.1 后）——预期基线为全 1 水平（acc ≈0.68 / macro_f1 ≈0.41，PTuningV2 报告 7.1）；
3. 训练 2751 步：每 100 步 1 次 eval + 1 checkpoint，TB 曲线 ~27 个 eval 点；
4. 结束 `full_model/` 出现 `model.safetensors`（~0.44GB）+ config + tokenizer；
5. **二次运行**：打印"检测到已保存的全量微调权重…跳过训练"，evaluate/predict/混淆矩阵全流程走通（trainer 顶格，无 NameError 风险）；最终指标与 best checkpoint 一致（`load_best_model_at_end` + macro_f1）；
6. 强制重训：删除 `full_model/` 即可。

---

## 四、方法论一句话

**"复用全量权重"的模式迁移到新架构时，真正要检查的只有三件事**：判断文件（model.safetensors）、加载类（SequenceClassification/CausalLM 对应）、config 是否随权重自包含（num_labels 读回）——本脚本三件都对。剩下的差异都在**任务语义层**（分类任务的基线指标、不平衡数据的选优口径），这类坑在参数高效系脚本的迁移里不会出现，是判别任务线独有的检查项。
