# PTuningV2-BERT 实现检查报告（test_PTuningV2_bert.py）

> 检查对象：`test_PTuningV2_bert.py`（由 `test_Lora_bert.py` 复制改造，P-Tuning v2 × BERT × MRPC 分类）
> 检查方式：① 通读 peft 0.18.1 的 `PeftModelForSequenceClassification`（`peft_model.py` L1597-1841）与 transformers 4.55 的 `modeling_bert.py`；② CPU 实测五组对照实验（三种 prompt 方法的兼容性、gc/use_cache 对 prefix 梯度的影响）。
> 结论先行：**这个组合在当前环境（peft 0.18.1 × transformers 4.55）下机制上是可行的**——此前「BERT 不支持 prompt learning」的认知需要修正；但脚本存在**一个致命问题（`gradient_checkpointing=True` 会让 prefix 梯度彻底消失、训练假跑）**和几处从 LoRA 脚本带来的遗留。

---

## 一、核对结论总览

| 检查项 | 结论 |
|---|---|
| `PrefixTuningConfig` + `TaskType.SEQ_CLS` + BERT 组合 | ✅ peft 官方支持（docstring 的 Example 就是这个组合）；**且 4.55 重构版 BERT 的 BertSelfAttention 实现了 KV 注入路径**，prefix 真实生效（实测见第二节） |
| classifier 分类头 | ✅ peft 自动把 `["classifier","score"]` 加入 `modules_to_save`，无需手动传（实测可训练参数 296,450 = prefix 294,912 + classifier 1,538，正好对上） |
| [CLS]/pooler 位置 | ✅ 无坑：KV 注入只加 key/value，**query 序列不变**，`[CLS]` 仍在位置 0（这是 KV 注入与拼输入的本质区别） |
| `gradient_checkpointing=True`（L183） | 🔴 **致命：prefix 梯度消失 → 假跑**（实测 3 个可训练参数只剩 2 个有梯度），必须改 False |
| `config.use_cache = False`（L162） | ✅ 无害（实测单独设置时 prefix 仍生效），但 BERT 无缓存概念，建议删掉避免误导 |
| baseline 的 `output_dir`（L112） | ⚠️ 还是 `bert-base-uncased_Lora`（复制遗留），应改为 `PTuning-V2` |
| 文案/注释残留 | ⚠️ L224/226「LoRA适配器」、加载分支的 `is_trainable=True` 注释讲的是 AdaLora 的坑，与本实验无关 |

---

## 二、核心发现 1（认知修正）：PREFIX_TUNING + BERT 在 peft 0.18.1 下真实可用

**为什么这次能跑**：transformers 4.55 的 BERT 是重构版，`BertSelfAttention.forward` 带 `past_key_value: Optional[Cache]` 参数且实现了完整的 cache update 逻辑（`modeling_bert.py` L226-278，与 decoder 同构）。peft 的注入链路：

1. `PeftModelForSequenceClassification.forward` 对 PREFIX_TUNING 走 `_prefix_tuning_forward`；
2. 它用 `inspect.signature` 探测：`BertForSequenceClassification.forward` 无 `past_key_values` → fallback 到 backbone `BertModel`（**有**该参数）→ 直接调 `BertModel(**kwargs)`；
3. prefix KV 进入每层 BertSelfAttention 的 attention 计算（K/V 拼接），**真实参与前向**（实测 prompt embedding 梯度非零）。

**两个此前担心的问题实测都不存在**：

- ~~「BERT 是 encoder 没有 KV cache，PREFIX_TUNING 会崩/假跑」~~ → 4.55 重构版已带 KV 路径（CPU 实测：`loss=0.53`，3/3 参数有非零梯度）；
- ~~「prompt 拼在输入前会把 [CLS] 挤出位置 0，pooler 吃到虚拟 token」~~ → **KV 注入不改 query 序列**，`[CLS]` 仍在位置 0，`outputs[1]`（pooler_output）→ dropout → classifier 的链路完全正常（这段逻辑由 peft 的 `_prefix_tuning_forward` 手工接管，不走 BertForSequenceClassification 的原始 forward）。

> 注意这条链路的一个实现细节：`_prefix_tuning_forward` 绕过了 `BertForSequenceClassification.forward`，自行完成 pooler→dropout→classifier→loss，其中 loss 按 `problem_type` 现算——`num_labels=2` + 整型 labels → `single_label_classification`（CrossEntropyLoss），与原生路径等价 ✓。

**同时也修正本仓库旧结论**：`PrefixTuning实现检查与报错分析.md` 2.5 节和 PTuningV2 报告 6.3 目标 C 中「BERT（SEQ_CLS）不支持 prompt learning 系列 / KV 注入问题不适用」的说法**对当前版本组合不成立**——支持，且门槛问题的表现需要在 BERT 上重新观察（分类任务的损失形态与 Qwen 生成不同，见第五节观察建议）。

---

## 三、核心发现 2（唯一必改项）：`gradient_checkpointing=True` → prefix 梯度消失、训练假跑

**位置**：L183 `gradient_checkpointing=True`（从 LoRA 脚本带来的）

**机制**：prefix KV 走的正是 `GradientCheckpointingLayer` 会丢弃的 `past_key_value` 参数——与 Qwen 线第一轮的假跑（`PrefixTuning实现检查与报错分析.md` 第二节）**完全同构**，只是这次连 "Caching is incompatible" 警告都不出现（BERT 的 layer 调用方式不同，假跑更加静默）。

**实测对照**（CPU，同一组输入）：

| 配置 | 有非零梯度的可训练参数 | 判定 |
|---|---|---|
| gc 不开 | **3/3**（prompt embedding + classifier.weight + classifier.bias） | 真学习 |
| gc=True + use_cache=False（**你的现状**） | **2/3**——prompt embedding 梯度为 None | **假跑**：2000+ 步只训练 classifier，prefix 纹丝不动 |

**修改方案**（L183）：

```python
    gradient_checkpointing=False,    # ★ v2的prefix走past_key_value路径，gc会把它丢弃→假跑
                                     #   （CPU实测：gc开时prompt embedding梯度为None）。BERT-base
                                     #   batch4×128激活很小，gc的省显存收益本来就低，关掉无压力
```

显存核算：BERT-base 冻结主干 + prefix 0.29M + classifier 0.0015M，batch=4 × max_length=128 的激活显存极小，LoRA 线开着 gc 是为了批量 8 的历史原因，这里关闭毫无压力。

---

## 四、其余修改清单（按优先级）

### 4.1 `config.use_cache = False`（L162）：无害但无意义，建议删

BERT 没有 KV 缓存机制，这行是从 Qwen 脚本带过来的惯性写法。实测单独设置时 prefix 梯度正常（3/3），**保留不报错**；但它暗示「这里存在需要关缓存的问题」，误导后续阅读。建议删除。

### 4.2 baseline 的 `output_dir`（L112）：路径残留

```python
    output_dir="./training/bert-base-uncased_Lora/output",   # ← 改为 ./training/bert-base-uncased_PTuning-V2/output
```

上一轮刚修正过 LoRA/AdaLora 的目录混写，这个同理：不同实验的 checkpoint 输出不要落进同一个目录。

### 4.3 文案与注释残留（可选清理）

- L224 注释 / L226 `"LoRA适配器已保存至"` → PTuningV2；
- L143-149 加载分支的整段 `is_trainable=True` 注释讲的是 **AdaLoraModel.forward 正则段的坑**（Qwen AdaLora 报告第八节），对 prompt tuning **完全不适配，且 `is_trainable=True` 会直接 ValueError 崩溃**（peft 对 prompt learning 系显式禁止，实跑已触发，修复方案见第八节——**此处更正本报告初版「is_trainable=True 本身可以保留」的错误判断**）；
- L136 `lora_save_path` 变量名、目录名 `lora_adapter`：同前三份报告的遗留命名问题，可顺手改为 `adapter_save_path` / `prefix_adapter`（**改目录名时 L138 的 `has_adapter` 判断要同步**）。

### 4.4 已验证无需修改的部分

1. **数据管线**：与 LoRA 线逐字相同，合法复用 `datasets/glue/cache-mrpc/*.arrow` ✓；
2. **`lr=1e-4`**：可训练参数 296,450（0.26%），MRPC 3668 条 × 3 epochs ≈ 2751 步，量级合理 ✓；
3. **`metric_for_best_model="macro_f1"` + `load_best_model_at_end=True`** ✓（继承自 LoRA 线，正好用于跨方法对比）；
4. **`peft_type="PREFIX_TUNING"` 显式传参** ✓（会被 `__post_init__` 强制覆盖为同名枚举，无害）；`prefix_projection=False` 是 v2 论文形态，**保持不动**——它是本实验的「被测对象」；
5. **双 Trainer 结构、双模式（has_adapter）** ✓。

---

## 五、跑起来之后的观察建议

1. **预期参数打印**：`trainable params: 296,450 (0.26%)`（若不是这个数，说明 config 有出入）；
2. **真跑判据**（修完 gc 后必查）：TensorBoard 里 train/loss 应随步数下降；若 loss 长期贴着基线不动，回来查 gc 是否关干净；
3. **起点坑监控（可选，对齐 Qwen 线方法）**：基线 eval（L130）用的是未包装 model；想量化 prefix 注入的「起点罚金」，可在 `get_peft_model` 之后、`trainer.train()` 之前补一次 `trainer.evaluate()`——包装后未训练的 macro_f1 若明显低于基线，即存在起点坑。Qwen 线的教训是「跨方法对比先对齐起点」；
4. **与 LoRA 线的对比标准**：同数据同超参结构下，看最终 macro_f1 与 LoRA 线（已有结果）的相对高低。若 v2 形态（projection=False）明显落后，那正是论文谱系的预期（小数据下重参数化必要，见 Qwen 线第八节），届时可开 `prefix_projection=True`（+ `encoder_hidden_size=512`）做与 Qwen 线对称的对照；
5. **随机性提醒**：`seed=42` 只固定了 random/np/torch，但 `BertForSequenceClassification` 的 classifier 是 `from_pretrained` 时随机初始化的（每次运行结果会有小幅波动，多次对比时注意）。

---

## 六、一句话行动建议

改一行就能跑：**L183 `gradient_checkpointing=True` → `False`**（其余 L112 路径、L162、文案顺手清理）。修完后这将是仓库第一条在 BERT 上跑通的 prompt 系实验线——且恰好补全了三方法对照矩阵：P-Tuning v1/v3 尚未在 BERT 上做（`PromptEncoderConfig`/`PromptTuningConfig` + SEQ_CLS 同样可用，CPU 实测均能前向反传），需要时按本报告第二节的路数即可。

---

## 七、实战复盘：全 1 预测的定位与拯救（实跑后补充，七组对照实验）

> 实跑现象：trainable params 296,450（0.27%，**与预期完全一致，不是异常**）；eval_loss 0.6341 → 0.6149（仅 -0.02）；accuracy/f1/macro_f1 与基线**一字不差**；预测全 1，混淆矩阵 [[0,129],[0,279]]。

### 7.1 合理性判定：结果「合理」但属于退化解，不是随机波动

1. **基线本身就是全 1**：acc 0.684 = MRPC 验证集多数类占比（279/408），macro_f1 0.406 = 全 1 时的 macro F1——随机初始化的 classifier 天然输出单侧；
2. **训练后纹丝不动**：三个指标与基线完全相同 → 模型停留在「随机分类头」的退化解上，只把输出概率往类 1 轻推了一点（loss -0.02 的全部含义）；
3. **退化解为什么是全 1**：MRPC 类不平衡（正类 ~67%），当模型学不到判别信号时，**全预测多数类就是 loss 的局部最优**——这是类不平衡数据上「没学动」的标准形态，也解释了为什么 loss 不升反微降。

### 7.2 七组对照实验定位（CPU 小样本口径，train 200~1500 条 / val 200 条 / batch16）

| # | 配置 | loss 变化 | acc | macro_f1 | 预测类别数 |
|---|---|---|---|---|---|
| E1 | linear probe：frozen BERT **只训 classifier**，lr=1e-3 | 0.628→0.640 | 0.685 | 0.407 | **1** |
| E2 | **LoRA 对照**（r8/qk/v，lr=5e-4） | 0.716→**0.402** | **0.820** | **0.763** | **2** ✅ |
| E3 | v2 现状（proj=False，lr=1e-4） | 0.563→0.644 | 0.685 | 0.407 | 1 |
| E4 | proj=True（lr=5e-4） | 0.685→0.614 | 0.685 | 0.407 | 1 |
| E5 | v2 + **lr=1e-3**，3ep | 0.563→0.683 | 0.640 | **0.627** | **2** ✅ |
| E6 | v2 + **lr=3e-3**，3ep | 0.563→0.638 | 0.670 | **0.646** | **2** ✅ |
| E7 | **proj=True + 小尺度0.02初始化 + lr=1e-3**，3ep | 0.588→**0.435** | **0.770** | **0.665** | **2** ✅ |

三个决定性结论：

1. **E1（决定性）**：连「不包装、只训 classifier、lr 1e-3」都全 1 → 「frozen 主线 + 轻量分类头」这一类方法（v2 属于此类，主干冻结）在 MRPC 上天然起步极慢。**不是你的实现有问题，是方法类属决定的天花板起点**；
2. **E2 对照**：LoRA 同条件下轻松到 0.763——它能学不是因为分类头，而是 LoRA 直接改 attention 权重（表示本身在适配）。这从机制上解释了「LoRA 工业主流 vs prompt 系学术为主」；
3. **E5-E7**：**lr 是决定性变量**——1e-4 → 1e-3 一步就脱离全 1；E7 再叠加重参数化 + 小尺度初始化后 loss 才出现实质下降（0.435），学习曲线才健康。

### 7.3 机制解释（三层叠加）

- **方法层**：v2 冻结主干 → 分类的表示质量由 frozen BERT 决定，0.27% 的 prefix 参数要从被 N(0,1) 污染的 attention 里硬推出判别边界，步长（lr）不够时永远停在退化解；
- **数据层**：类不平衡提供了「全 1」这个容易掉进去的 loss 局部最优；
- **与论文的差距**：v2 论文 MRPC 达到 SOTA 的条件是 **RoBERTa-large/BERT-large + prefix 长度 100+ + lr 0.5~1 量级（Adafactor）+ 精调**——bert-base + lr 1e-4 与论文条件差了几个数量级，全 1 是这些差距的集中体现。

### 7.4 优化方案（E7 实测最优，直接可抄）

**方案一（实测最优）**——重参数化 + 小尺度初始化 + 大 lr（⚠️ 开启 projection 后脚本实际跑的是**原版 Prefix Tuning 形态**，命名定位见下方「方案怎么选」）：

```python
# L154-160 的 peft_config 改为：
peft_config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=16,
    prefix_projection=True,       # 重参数化（原版Prefix形态）：初始化尺度受Linear控制
    encoder_hidden_size=512,      # True时生效（peft默认512，显式写出便于理解）
    inference_mode=False
)
model_trained = get_peft_model(model, peft_config)
# ★ 新增小尺度初始化（紧接 get_peft_model 之后）：
with torch.no_grad():
    model_trained.prompt_encoder["default"].embedding.weight.normal_(0, 0.02)
```

```python
# TrainingArguments 两处：
learning_rate=1e-3,            # ★ 1e-4→1e-3（实测E7）；可在5e-4~3e-3扫
num_train_epochs=5,            # 3→5，配合已有的 load_best_model_at_end 按 macro_f1 保最优
```

**方案二（v2 论文纯度，且超参更贴论文）**：`prefix_projection=False` 不动，把 `learning_rate` 提到 1e-3~3e-3、epochs 提到 5。实测 E5/E6（lr=1e-3/3e-3）**均脱离全 1**（macro_f1 0.627/0.646），离 E7 的 0.665 只差约 2 个点——这不是妥协，反而更贴论文：**prompt 系文献给虚拟 token 参数的 lr 本来就在 1e-3~1e-2 量级**（v2 官方搜的就是这个区间，Lester 的 Prompt Tuning 甚至用到 0.1~1），此前的 1e-4 是 adapter 系（LoRA）直觉的错位。E5 的 loss 仍在 0.68 高位、曲线不如 E7 健康，全量训练（步数 ~15 倍）下应明显改善。

#### 方案怎么选：命名与定位（重要，避免「标签与内容不符」）

严格按论文形态定义，`prefix_projection` 开关决定你跑的是哪个方法：

| 开关 | 论文形态 | 核心特征 |
|---|---|---|
| `prefix_projection=False` | **P-Tuning v2**（Liu 2022） | 每层前缀 + **去重参数化**（论文 Fig.1 标注 "No re-param"）、去 verbalizer |
| `prefix_projection=True` | **原版 Prefix Tuning**（Li & Liang 2021） | 每层前缀 + **MLP 重参数化**（为稳定性提出的设计） |

三点判断：

1. **实现层面两者是同一段代码**：peft 只有一个 `PrefixTuningConfig`，区别仅是这个超参开关——「是哪个方法」在你仓库里只体现在叙事定位（脚本名/目录/文档）上；
2. **重参数化只是 v2 的定义性设计之一，不是全部**：v2 的核心是「深度提示（每层注入）+ NLU 适配」；BERT + MRPC 分类本来就是 v2 主场，方案二改的只是 lr，方法纯度没有损失；
3. **想保 v2 纯度不需要开 projection**：方案二按论文量级给足 lr 即可出非退化解（E5/E6 实证）。

落地建议（与 Qwen 侧现有结构对称：Prefix=效果线 / PTuningV2=机制线）：

- **`test_PTuningV2_bert.py` 保持 v2 纯度**：`projection=False` + `lr=1e-3~3e-3`——标签与内容一致；
- **想要方案一的最强效果**：另建 `test_Prefix_bert.py`（BERT 侧正好缺 Prefix 线），复制本脚本改 `projection=True` + 小尺度初始化——两个名字、两份脚本、各做各的；
- **不推荐**在 PTuningV2 脚本里直接开 projection：标签与内容不符，几个月后回看会误导自己。

**注意事项**：
- 方案一的可训练参数变为 **≈9.86M（9.0%）**（embedding 12k + Linear1 ≈0.4M + Linear2 ≈9.4M），适配器体积 ~40MB（fp32），显存增加约 120MB——6G 卡无压力；
- 与 Qwen 线的对照意义：Qwen 线上 proj=True 也起效（-9.1%），但 BERT 分类上单靠 proj=True 不够（E4 仍全 1），**必须配合大 lr**——「重参数化 + 足够步长」是 Prefix 系稳定训练的两个必要条件，缺一不可；
- 实验口径说明：E1-E7 是 CPU/fp32/batch16 的小样本快筛，你的全量训练（3668 条 × 2751 步，是实验步数的 ~15 倍）跑方案一应显著好于 E7 的 0.665，天花板参考 E2 的 LoRA（0.763 同口径）。

### 7.5 方法论沉淀

1. **退化解检测器：看「预测类别数」**——分类任务上 `len(set(preds))==1` 就是「没学动」的硬信号，比 loss 曲线更直观（loss 微降可能只是偏置滑动）；
2. **frozen 主线方法的 lr 直觉要重校**：adapter 系（LoRA 权重旁路）1e-4 够用，embedding/prompt 系参数需要 1e-3 起步（Qwen 线 PromptTuning 用 5e-3 同理）——「冻结得越彻底，需要的步长越大」；
3. **E1 的价值**：给任何「frozen 主线 + 轻量头」实验先跑一个 linear probe 下限，就知道终点差距里多少是方法不行、多少是任务本身就难。

---

## 八、实战复盘：加载分支 `is_trainable=True` 报错 ValueError（二次运行触发）

> 触发时机：首次训练+保存成功后，**二次运行进入 `has_adapter` 加载分支**时报错：
> `ValueError: Cannot set a prompt learning adapter to trainable when loading pretrained adapter.`（test_PTuningV2_bert.py L146）
> ⚠️ 本节同时**勘误本报告 4.3 节**：初版判断「is_trainable=True 本身可以保留」是错误的，实跑证明 peft 对 prompt learning 系直接 raise——已在 4.3 就地更正。

### 8.1 根因（peft 源码，无条件禁止）

`peft/peft_model.py` `PeftModel.from_pretrained` L523-525：

```python
if config.is_prompt_learning and is_trainable:
    raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
else:
    config.inference_mode = not is_trainable
```

保存的 `adapter_config.json` 里 `peft_type=PREFIX_TUNING`，其配置类继承 `PromptLearningConfig` → `is_prompt_learning=True` → 传 `is_trainable=True` 必然命中 raise。**设计原因**：prompt learning 的 checkpoint 保存的是**编码后的最终 prompt embedding**（不含重参数化网络，Qwen 侧 PTuningV2 报告「已验证部分」第 5 条），peft 不支持从这种格式续训，所以加载路径干脆禁止 trainable。

这行 `is_trainable=True` 是从 `test_Lora_bert.py` 复制来的——**LoRA 不是 prompt learning**，那里它合法且有存在必要（AdaLora 的 `trainable_adapter_name` 坑）；跨方法复制脚本的又一例「参数带着语义走」。

### 8.2 修复方案（一行：删掉 `is_trainable=True`）

L146 改为（L143-149 那段 AdaLora 注释一并删除或改写）：

```python
    model_trained = PeftModel.from_pretrained(model, lora_save_path)
    # prompt learning 系（Prefix/PTuningV2/PromptTuning/P-Tuning）加载时禁止 is_trainable=True：
    # peft 源码对 is_prompt_learning 配置直接 raise（保存的是最终prompt embedding，不支持续训）。
    # 默认 is_trainable=False 即推理/评估模式，本分支只 evaluate/predict，完全够用
```

### 8.3 修复后验证（CPU 实测全通过）

保存→`PeftModel.from_pretrained`（默认参数）→forward 链路：

```
[保存内容] 含classifier键: True（modules_to_save 的分类头权重一并保存，总键数3）
[加载+forward] OK, loss=0.3043
[权重恢复] prompt embedding 逐元素一致: True | classifier 恢复(≠随机初值): True
trainable params: 294,912 || trainable%: 0.2686
```

两个预期内的"反直觉"细节，遇到不要慌：

1. **加载后 `print_trainable_parameters()` 显示 294,912（0.27%）而非 0%**——prompt embedding 的 `requires_grad` 默认仍为 True。无害：加载分支不会调 `trainer.train()`，evaluate/predict 全在 no_grad 下，不发生任何参数更新（脚本 L226-227 注释里"显示 0%"的预测不准确，可顺手改成"显示 ~29.5万"）；
2. **加载后 evaluate 的指标应与训练结束时的最优 checkpoint 一致**——若差异大，先查 `load_best_model_at_end` 是否生效（最终保存的是 best 还是 last）。

### 8.4 推广结论：全部脚本 has_adapter 分支的 is_trainable 用法（源码+实测）

> 问题：既然 prompt learning 崩了，是否所有脚本的加载分支都该把 `is_trainable` 去掉？
> **答案：不能一刀切——按方法类属三分，AdaLora 反而必须保留 `True`。**

**规则（由 peft 源码机制决定）**：

| 方法类属 | `is_trainable=True` 的后果 | 正确写法 | 机制依据 |
|---|---|---|---|
| **prompt learning**（Prefix/PTuningV2/PromptTuning/P-Tuning） | ❌ 直接 ValueError | **不传（默认 False）** | `from_pretrained` 对 `is_prompt_learning` 配置无条件 raise（8.1 源码） |
| **LoRA 系**（LoRA/QLoRA） | ✅ 能跑但无意义 | **不传（默认 False）更干净** | `LoraModel.forward` 只是委托 base model，不访问任何 trainable 专用属性 |
| **AdaLora** | ✅ 且**必要** | **必须传 True** | `trainable_adapter_name` 只在 `inference_mode=False` 时赋值（`adalora/model.py` L85-87）；而 `AdaLoraModel.forward` 只要 **loss 非 None 就无条件访问它**（L230，无 eval-mode 守卫）——evaluate/predict 的前向都带 labels → 默认 False 加载后**评估必崩 AttributeError** |

**CPU 实测四组对照**（BERT/SEQ_CLS，保存→加载→带 labels 前向）：

| 试验 | 结果 |
|---|---|
| LoRA + `is_trainable=True`（test_Lora_bert.py 现状） | OK（能跑，但无必要） |
| LoRA + 默认 False（删掉后） | **OK** → 可安全删 |
| AdaLora + `is_trainable=True`（AdaLora 脚本现状） | OK |
| AdaLora + 默认 False（删掉后） | **崩：AttributeError: ... has no attribute 'trainable_adapter_name'** → 不能删 |

**全部脚本逐一核对**：

| 脚本 | 方法 | 现状 | 处置 |
|---|---|---|---|
| test_Lora_qwen3 / test_QLora_qwen3 | LoRA/QLoRA | 不传 | ✓ 无需动 |
| test_Prefix / test_Prompt / test_PTuningV2（Qwen 侧） | prompt learning | 不传 | ✓ 无需动 |
| test_Lora-load_bert.py（纯推理脚本） | LoRA | 不传 | ✓ 无需动（且纯 generate/predict 不带 loss，连 AdaLora 式风险都没有） |
| **test_AdaLora_qwen3-0.6b.py（L240）** | AdaLora | `True` | **保留！**（删掉→二次运行 evaluate 崩，注释里的理由真实有效） |
| **test_AdaLora_bert.py（L147）** | AdaLora | `True` | **保留！**（同上） |
| **test_Lora_bert.py（L145）** | LoRA | `True` | **可删**：注释声称的「AdaLoraModel.forward 访问 trainable_adapter_name」是 AdaLora 专属机制，LoRA 前向不涉及（上表实测）；这是从 AdaLora 脚本复制时连注释带参数一起迁移的残留。删掉后另需同步改写 L142-144 注释 |
| **test_PTuningV2_bert.py（L146）** | prompt learning | `True` | **必删**（8.2，即本次报错） |

一句话规则：**`is_trainable=True` 只在「加载后还要继续训练」或「AdaLora 评估」这两种场景有存在意义；纯评估/推理的加载分支，除 AdaLora 外一律不传。**

### 8.5 方法论沉淀

**跨方法复制脚本时，「加载分支」比「训练分支」更容易踩雷**：训练分支的参数（lr/gc/projection）失败会立刻显形，而加载分支的参数带着**上一方法的语义**（这里的 `is_trainable` 是 AdaLora→LoRA 一脉的必需品）悄悄迁移，直到二次运行才爆。清单式自查：复制脚本后，对 `PeftModel.from_pretrained` / `save_model` / 配置类的**每个参数**问一句"这个参数在新方法下还有意义吗"。

---

## 九、实战复盘（续）：加载分支 `NameError: trainer is not defined`（结构性缩进 bug）

> 触发时机：修掉 8.2 的 `is_trainable` 后，二次运行推进到 L206 报：
> `NameError: name 'trainer' is not defined. Did you mean: 'Trainer'?`
> 这是**首次训练能跑通、二次加载必崩**的结构性 bug——与第八节同属"加载分支从未被真正执行过"一族。

### 9.1 根因：`trainer` 定义在 `else`（训练分支）内部，分支外引用

当前结构（L151-206 骨架）：

```python
if has_adapter:
    model_trained = PeftModel.from_pretrained(...)      # 加载分支：只定义了 model_trained
    model_trained.print_trainable_parameters()
else:
    peft_config = PrefixTuningConfig(...)
    model_trained = get_peft_model(model, peft_config)
    ...
    training_args = TrainingArguments(...)              # ← 缩进在 else 内！
    trainer = Trainer(...)                              # ← 缩进在 else 内！

if not has_adapter:
    trainer.train()                                     # 加载模式跳过，没事

eval_res = trainer.evaluate()                           # ← L206：两种模式都要执行；
                                                        #   加载模式下 trainer 从未创建 → NameError
```

Python 的变量作用域按执行流走：加载模式（`has_adapter=True`）只走 `if` 分支，`else` 块整体不执行，`training_args`/`trainer` 从未被绑定。首次训练走 `else` 所以没事——**bug 被默认路径掩盖**。

### 9.2 波及面：三个 BERT 脚本结构不一，LoRA_bert 同病

| 脚本 | `training_args`/`trainer` 位置 | 加载模式 L2xx `trainer.evaluate()` | 处置 |
|---|---|---|---|
| **test_PTuningV2_bert.py**（本脚本） | else 内（L165/L190） | ❌ NameError（本次报错） | 按 9.3 修 |
| **test_Lora_bert.py** | else 内（L169/L194） | ❌ **同样的潜伏 NameError**（L210；只是从未跑过二次运行没暴露） | **同修** |
| test_AdaLora_bert.py | 顶格（L175/L210） | ✅ 两模式都有 trainer | 无需动 |
| Qwen 侧全部（Lora/QLora/AdaLora/Prefix/Prompt/PTuningV2） | 顶格 | ✅ | 无需动 |

修复本脚本时**顺手把 test_Lora_bert.py 一起修**（同样把 L169-202 的两块定义去掉一层缩进）——它已经踩在同一个坑上。

### 9.3 修复方案：把两块定义整体移出 `else`（与 Qwen 侧/AdaLora_bert 结构对齐）

手改方式：将 L165-188 的 `training_args = TrainingArguments(...)` 和 L190-198 的 `trainer = Trainer(...)` **整体去掉一层缩进**（从 `else:` 内移到模块顶层），`else:` 分支里只留 peft 配置与包装部分。修复后骨架：

```python
if has_adapter:
    model_trained = PeftModel.from_pretrained(model, lora_save_path)
    model_trained.print_trainable_parameters()
else:
    peft_config = PrefixTuningConfig(...)
    model_trained = get_peft_model(model, peft_config)
    model_trained.config.use_cache = False
    model_trained.print_trainable_parameters()

# ↓↓↓ 两块定义移到这里（顶格，两种模式都会执行） ↓↓↓
training_args = TrainingArguments(...)     # 原 L165-188，去掉一层缩进
trainer = Trainer(                         # 原 L190-198，去掉一层缩进
    model=model_trained,                   # 两种模式下 model_trained 都已定义（if/else 各自赋值）
    ...
)

if not has_adapter:
    trainer.train()

eval_res = trainer.evaluate()              # 两种模式都正常
```

**为什么这样改是对的**（不是绕过而是对齐正确结构）：

1. `Trainer` 构造本身**无副作用**（不训练、不写 checkpoint），加载模式下构造它只为后续 `evaluate()`/`predict()` 服务，与 Qwen 侧脚本、`test_AdaLora_bert.py` 的既有结构完全一致；
2. `model_trained` 在 if/else 两个分支里**都会被赋值**（加载=PeftModel.from_pretrained，训练=get_peft_model），移出后引用安全；
3. `if not has_adapter: trainer.train()` 的守卫保持不变——加载模式跳过训练，只评估；
4. `load_best_model_at_end=True` 只在 `train()` 结束时触发加载最优 checkpoint，对纯 evaluate 无影响。

### 9.4 修复后预期与两个小提醒

- **加载模式全流程应通**：evaluate → predict → 混淆矩阵 → "已训练权重来自…（无需重复保存）"；eval 指标应与训练结束时的 best checkpoint 一致（`load_best_model_at_end` + `metric_for_best_model="macro_f1"` 保存的是最优）；
- **提醒①（TB 日志追加）**：修复后加载模式的 `trainer.evaluate()` 因 `report_to="tensorboard"` 也会往 logs 写一个 eval 点（step 按当前 global_step，通常为 0）。无害，但若想让训练曲线干净，可在加载分支给 evaluate 前临时设 `training_args.report_to = "none"`（可选项，不修也行）；
- **提醒②（旧注释清理）**：L143-145 那段 AdaLora 的 `is_trainable` 注释（8.2 已要求删除）若还在，这次一起清掉——它描述的是已经不存在的代码。
