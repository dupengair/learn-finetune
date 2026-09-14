# LoRA 微调 `element 0 of tensors does not require grad` 错误分析与修复

> 适用脚本：`test_Lora_qwen3-0.6b.py`（Qwen3-0.6B + peft LoRA + Trainer）
> 环境：transformers 4.55.0 / peft 0.18.1 / torch 2.8.0 / RTX 4050 6GB

---

## 一、错误现象

运行 `python test_Lora_qwen3-0.6b.py`，进入训练第 1 步就在 `loss.backward()` 处崩溃：

```
开始 SFT 微调：
  0%|                                                                | 0/603732 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
torch/utils/checkpoint.py:85: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
Traceback (most recent call last):
  File "test_Lora_qwen3-0.6b.py", line 233, in <module>
    trainer.train()
  ...
  File ".../accelerate/accelerator.py", line 2852, in backward
    loss.backward(**kwargs)
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

其中两条日志是定位关键：

| 日志 | 含义 |
|---|---|
| ``UserWarning: None of the inputs have requires_grad=True. Gradients will be None`` | **直接证据**：被 checkpoint 的计算段输入全都没有梯度，反向重算无法建图 |
| ``RuntimeError: element 0 of tensors does not require grad`` | loss 张量没有 `grad_fn`，`backward()` 无图可回传 |

（`use_cache=True is incompatible with gradient checkpointing` 只是框架自动把 KV cache 关掉了，无害，可忽略。）

---

## 二、结论速览

**根因一句话**：LoRA 把包括 embedding 在内的主干权重全部冻结后，`gradient_checkpointing=True` 默认启用的 **reentrant 梯度检查点** 找不到任何 requires_grad 的输入张量，前向计算图断裂，loss 无法反向传播。

**修复 = 两处一行级修改（推荐都加，双保险）**：

```python
# ① get_peft_model 之后加一行（脚本约 195 行）
model_lora.enable_input_require_grads()

# ② TrainingArguments 里加一个参数（脚本约 210 行附近）
gradient_checkpointing_kwargs={"use_reentrant": False},
```

`gradient_checkpointing=True` 本身是 6G 显存卡的刻意 OOM 规避参数，**必须保留**，不能删。

---

## 三、原理分析

### 3.1 LoRA 冻结了哪些参数

`get_peft_model(model, peft_config)`（脚本 194 行）之后：

- 主干全部权重 `requires_grad=False`，**包括 `embed_tokens` 嵌入层**；
- 只有 LoRA 低秩矩阵 A/B（挂在 `q/k/v/o/gate/up/down_proj` 上）可训练。

`model_lora.print_trainable_parameters()` 会显示可训练参数占比约 1% 不到，这是 LoRA 的正常语义。

### 3.2 梯度检查点为什么断图

`TrainingArguments(gradient_checkpointing=True)`（脚本 210 行）会让 Trainer 调用
`model.gradient_checkpointing_enable()`。在 transformers 4.55 中，**默认使用 reentrant 版本的
`torch.utils.checkpoint.checkpoint`**。

reentrant 检查点的工作方式：

1. 前向时对每个被 checkpoint 的段（Qwen3 里是每个 decoder 层）在 `no_grad` 下跑一遍，只记录段边界、不建图；
2. 反向时对每段**重新前向计算**来重建局部计算图，再回传梯度；
3. 这个机制有一个硬性前提：**每个段的输入张量中至少有一个 `requires_grad=True`**，否则反向重算时 PyTorch 不知道要从哪里开始建图。

失败链条：

```
embedding 权重被 LoRA 冻结 (requires_grad=False)
        ↓
embedding 输出（即第一个 decoder 层的输入）requires_grad=False
        ↓
reentrant 检查点发现段输入全都不需要梯度
        → UserWarning: None of the inputs have requires_grad=True. Gradients will be None
        ↓
整条前向图没有建立，loss 没有 grad_fn
        ↓
loss.backward() → RuntimeError: element 0 of tensors does not require grad
```

换句话说：**这不是显存问题、不是数据问题，而是"冻结主干 + reentrant 检查点"的组合在数学上就没有可回传的路径入口**。全量微调不会遇到（所有权重都需要梯度），纯推理更不会遇到（不 backward）——只有 LoRA/Prefix 等部分参数微调 + 梯度检查点的组合会踩中。

---

## 四、修改方法（手动改两处）

### 修改 ①：`get_peft_model` 之后加 `enable_input_require_grads()`

位置：脚本 194–195 行附近，`get_peft_model` 与 `print_trainable_parameters()` 之间。

```python
model_lora = get_peft_model(model, peft_config)
model_lora.enable_input_require_grads()   # ✅ 新增：让冻结embedding的输出可求导，配合梯度检查点
model_lora.print_trainable_parameters()  # 打印可训练参数
```

**原理**：这是 peft 在 `PeftModel` 上提供的官方接口。它在 embedding 层注册一个 forward hook，
把 embedding 的**输出**张量 `requires_grad_(True)`（不改变任何权重的可训练性，LoRA 语义完全不变）。
这样第一个 decoder 层的段输入就有了梯度入口，reentrant 检查点可以正常建图；梯度沿 LoRA 层回传，
最终仍然只更新 A/B 矩阵。

### 修改 ②：`TrainingArguments` 里改用非 reentrant 检查点

位置：脚本 197–221 行的 `TrainingArguments` 内，加在 `gradient_checkpointing=True` 旁边。

```python
    gradient_checkpointing=True,         # ✅ 梯度检查点，大幅降低激活显存，速度会慢一点
    gradient_checkpointing_kwargs={"use_reentrant": False},  # ✅ 新增：非reentrant检查点，不依赖输入梯度
```

**原理**：PyTorch 2.x 提供的非 reentrant 实现（`use_reentrant=False`）使用不同的 hook 机制，
**反向重算不要求段输入有梯度**，从机制上绕开了这个坑。这也是 PyTorch 官方目前推荐的方式
（reentrant 版本已处于维护/弃用状态），transformers 通过 `gradient_checkpointing_kwargs`
直接透传该参数。

### 推荐与说明

- **两处都加**最稳妥：①是 PEFT 文档的标准做法，②是更现代的检查点实现，两者叠加互不冲突。
- 只加其中任何一处**都能**解决本错误；若只加一处，优先 ②（一行参数、语义更干净）。
- 其余参数一律不动：`per_device_train_batch_size=1`、`gradient_accumulation_steps=4`、
  `gradient_checkpointing=True`、`bf16=True`、`optim="adamw_torch_fused"` 都是 6G 卡的
  刻意 OOM 规避配置。

---

## 五、不受影响的注意事项

| 项 | 说明 |
|---|---|
| 分词 arrow 缓存 | 本次修复**不涉及** `tokenize_func` / `max_length`，`datasets/zhihu-kol/cache*/` 下的 `.arrow` 文件**无需删除**（仅当以后改了分词逻辑才需要清缓存） |
| `device_map="auto"` | 脚本 24 行保留是仓库既定做法（仅推理/LoRA 脚本保留），与本错误无关，不动 |
| `use_cache` 警告 | 框架自动处理，无需理会 |
| BERT 线 | `test_Lora_bert.py` 同样开了 `gradient_checkpointing=True`，目前未报错是任务形状不同；若日后在 BERT LoRA 上遇到同样报错，按同样两步修复即可 |

---

## 六、验证步骤

从仓库根目录运行（脚本内全是相对路径，必须在根目录执行）：

```bash
python test_Lora_qwen3-0.6b.py
```

修复成功的判据：

1. **不再出现** `None of the inputs have requires_grad=True` 警告；
2. 进度条正常推进（603732 step 是 80 万样本 ÷ 有效批量 4 得到的总步数，**不需要跑完**）；
3. 每 10 step 打出一次 loss（`logging_steps=10`），loss 数值合理（初始通常在 2~4 之间浮动）；
4. step 200 会触发第一次保存（`save_steps=200`）、step 500 第一次评估（`eval_steps=500`）。

确认前几十步稳定、无 OOM 后即可 `Ctrl+C` 停止，产物会存到 `./training/qwen3-0.6b_Lora/output/`。

---

## 七、延伸知识

### 7.1 reentrant vs 非 reentrant 检查点

| | reentrant（旧默认） | non-reentrant（`use_reentrant=False`） |
|---|---|---|
| 建图方式 | 反向时重算整段，要求段输入有 requires_grad | 前向时用 autograd hook 记录，无输入梯度要求 |
| 与 PEFT 冻结主干组合 | **必须**额外 `enable_input_require_grads()` | 开箱即用 |
| RNG/随机性处理 | 有额外约束 | 更宽松 |
| 状态 | PyTorch 维护模式，趋于弃用 | 官方推荐 |

### 7.2 与 QLoRA 实验的关系

后续做 4bit 量化（QLoRA）实验时会用到 peft 的 `prepare_model_for_kbit_training(model)`——
它内部**已经自动调用了等价的 `enable_input_require_grads()`**，所以 QLoRA 脚本往往"顺带"没踩这个坑。
本脚本是纯 bf16 LoRA（不走 4bit），所以需要手动补上这一步。

### 7.3 一句话记忆

> **PEFT 部分参数微调 + `gradient_checkpointing=True` ⇒ 必须保证梯度能"流进"冻结的网络**：
> 要么 `enable_input_require_grads()` 打通入口，要么换 `use_reentrant=False` 检查点。

---

## 八、性能优化：系统内存吃满（14.9/15.8GB）与训练速度慢

> 场景：RTX 4050 6GB 显存 + WSL2，宿主机内存 32GB（WSL2 默认分配上限 15.8GB）。
> 现象：LoRA 训练时系统内存占比过高（14.9/15.8GB），训练速度很慢。

### 8.1 实测诊断数据（2026-09-10）

| 检查项 | 实测值 | 结论 |
|---|---|---|
| 仓库文件系统 | `/dev/sdd` ext4（WSL2 本地虚拟盘） | ✅ 排除 Windows 9p/drvfs 挂载盘 I/O 慢的问题 |
| 分词 arrow 缓存 | **train.arrow 42GB + val.arrow 5.3GB + test.arrow 5.3GB ≈ 53GB** | ⚠️ 原始 parquet 才 1.4GB，膨胀约 38 倍 |
| 训练总步数 | 进度条 603732 step × 有效批量 4 ≈ **训练集约 240 万条** | ⚠️ 学习实验规模严重超标 |
| 验证集规模 | 按 42GB:5.3GB 比例推算 **约 30 万条**，`eval_steps=500` + `per_device_eval_batch_size=1` | 🔴 **每次评估要做约 30 万次前向传播**，是速度慢的最大单一来源 |
| `free -h`（空闲时） | used 3.5Gi，buff/cache 2.5Gi，available 4.3Gi | 说明高内存不是进程泄漏（见 8.2） |

### 8.2 为什么系统内存会吃到 14.9/15.8GB

**根源：53GB arrow 缓存被 memory-map 后，页缓存把 WSL2 虚拟机内存撑满。**

链条：

1. `datasets` 库对 `.arrow` 文件采用 **memory-map（mmap）** 方式读取——不是把 53GB 全部读进进程内存，而是按需映射；
2. 训练过程中不断随机读取 `train.arrow` 的各个分片，**读过的页会留在 Linux page cache（页缓存）里**；
3. WSL2 是一台轻量虚拟机，其内存上限默认约为宿主机内存的一半（32GB → 15.8GB 上限）。页缓存不断增长直到顶到 VM 上限；
4. Windows 任务管理器只看到"WSL2 这个 VM 占了 15.8GB"，于是呈现为 14.9/15.8GB。

**判断是否真的有问题**：在 WSL 内跑 `free -h`——

- `used` 高（进程真实占用）→ 需要优化；
- `buff/cache` 高、`available` 充足 → 是页缓存，**可随时回收，实际无害**，只是观感吓人。

本例中 `used` 只有 3.5Gi，大头是页缓存 → 属于"无害但可观感差"，且根源是数据集太大。

**缓解手段**（按效果排序）：

1. **根治 = 缩小数据集**（见 8.3 第 1 条）：缩到 1~2 万条后 arrow 缓存只有几百 MB，问题自然消失；
2. 删除不再使用的旧缓存（确认旧配置不再复跑后）：
   ```bash
   rm ./datasets/zhihu-kol/cache/train.arrow ./datasets/zhihu-kol/cache/val.arrow ./datasets/zhihu-kol/cache/test.arrow
   ```
   可回收约 53GB 磁盘，也消除 mmap 页缓存来源；
3. WSL2 全局兜底：在 Windows 侧 `%UserProfile%\.wslconfig` 里设置 `memory=12GB`（给 VM 封顶）和 `[experimental] autoMemoryReclaim=gradual`（自动归还页缓存），改完 `wsl --shutdown` 重启生效。**不推荐作为首选**——它治标，数据集缩小后不需要。

### 8.3 训练速度慢的三大根源与优化（按优先级）

#### 优先级 1：缩小训练/验证集（收益最大，学习实验标配）

60 万 step 对学习/验证性质实验毫无必要。知乎 1~2 万条样本足够观察 loss 下降和 LoRA 的生成效果变化：

```python
# 在 load_dataset 之后、train_test_split 之前，或直接对切分后的 split 做 select
raw_datasets["train"] = raw_datasets["train"].select(range(20000))      # 2万条训练
# validation/test 同样 select(range(1000))
```

收益：训练步数 60 万 → 约 5000 step（`bs=1`×`accum=4`），从"跑不完"变成"几十分钟级"。

#### 优先级 2：别让评估拖垮训练（隐形最大开销）

当前配置 `eval_steps=500` + 30 万条验证集 + `per_device_eval_batch_size=1`：
**每 500 个训练 step 触发一次 30 万次前向传播的评估**——评估耗时是这 500 步训练的几十倍，进度条长期卡在 eval 上。

三管齐下：

```python
    # ① 验证集缩到 1000 条（配合优先级 1 的 select）
    # ② 评估频率降低
    eval_steps=1000,                    # 或直接 eval_strategy="epoch"
    # ③ eval 不做反向传播/梯度检查点，批量可以比训练大得多
    per_device_eval_batch_size=8,       # ⚠️ 勘误（见第十三节）：15万词表下此建议错误，会OOM，应保持1
```

收益：单次评估从 ~30 万次前向 → ~125 次前向，快三个数量级。

#### 优先级 3：`max_length` 从 4096 降到 1024

按 8.1 的数据反推，语料平均长度约 730 tokens，4096 只服务极少数超长回答的长尾。
长尾样本既吃显存（激活与长度成正比）又拖慢单步：

```python
    tokenized_full = tokenizer(full_texts, truncation=True, max_length=1024)
    tokenized_user = tokenizer(user_parts, truncation=True, max_length=1024)
```

⚠️ **改了 `max_length` 必须删旧 arrow 缓存**（本仓库关键坑 1）：删除 8.2 列出的三个 `.arrow`
文件，否则改动不生效、还会继续沿用 42GB 旧缓存。

#### 优先级 4（可选，逐项试）：训练侧进一步提速

| 手段 | 做法 | 说明 |
|---|---|---|
| 试关梯度检查点 | `gradient_checkpointing=False` 跑一次 | 0.6B 小模型 + `max_length=1024` + bs=1 时激活很小，6G 卡**很可能放得下**；成功则提速 30~40%。OOM 就开回来 |
| 批量 2×累积 2 | `per_device_train_batch_size=2` + `gradient_accumulation_steps=2` | 有效批量不变（4），padding 利用率和吞吐更高；显存不够就退回 |
| 显存碎片整理 | 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 缓解长序列下碎片化 OOM，代价极小 |
| 数据加载 | `dataloader_num_workers=2` | 默认 0（主进程加载）；WSL2 上收益有限，内存紧张时**不要**调大 |

`group_by_length=True` 等按长度分组策略只在 bs>1 时有意义，bs=1 可忽略。

### 8.4 优化后的推荐配置（示例，供手动修改参考）

```python
# ===== 数据侧 =====
raw_datasets["train"] = raw_datasets["train"].select(range(20000))     # 2万条
# validation / test 各 select(range(1000))
# tokenizer(..., truncation=True, max_length=1024)
# ⚠️ 改完删除 ./datasets/zhihu-kol/cache/ 下旧 .arrow 缓存再运行

# ===== TrainingArguments =====
training_args = TrainingArguments(
    output_dir="./training/qwen3-0.6b_Lora/output",
    logging_dir="./training/qwen3-0.6b_Lora/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # OOM 规避参数（保留；梯度检查点可先关试试）
    per_device_train_batch_size=1,      # 可试 2 + accum 2
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,       # ⚠️ 勘误（见第十三节）：15万词表下会OOM，应保持1
    gradient_checkpointing=False,       # 先试关；OOM 则改回 True
    gradient_checkpointing_kwargs={"use_reentrant": False},  # 保留（若检查点开启）
    bf16=True,

    eval_strategy="steps",
    eval_steps=1000,

    optim="adamw_torch_fused",
    report_to="none",
)
```

配套地：`get_peft_model` 之后仍需加 `model_lora.enable_input_require_grads()`
（见第四节——若保持 `gradient_checkpointing=False` 则可省，但建议保留写法以兼容日后重开检查点）。

### 8.5 预期效果小结

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 训练总步数 | 603732 step（跑不完） | ~5000 step（几十分钟级） |
| 单次评估前向次数 | ~30 万次 | ~125 次 |
| arrow 缓存磁盘占用 | ~53GB | 数百 MB |
| 系统内存（WSL2 VM 视角） | 顶到 15.8GB 上限 | 低位平稳（页缓存来源消失） |
| 单步速度 | 受 4096 长尾 + 检查点拖累 | 更快（且大概率可关检查点再提 30~40%） |

### 8.6 注意事项

- 本文所有优化**均为文档建议，未改动任何原始脚本**；
- `select()` 会改变数据集指纹，`map(cache_file_name=...)` 检测到指纹变化会重新分词写缓存——
  为避免新旧缓存混淆，**建议先删旧 `.arrow` 再跑**；
- 三个 OOM 底线参数（bs=1 / 梯度累积 / bf16）与 `adamw_torch_fused` 的设计意图不变；
  关梯度检查点是"实验项"而非"推荐项"，OOM 立即回退；
- 若日后要恢复全量数据复跑原配置，需要重建大缓存（原脚本逻辑不变即可，代价是时间和 53GB 磁盘）。

---

## 九、进一步提速分析：1.33 s/it 还能快多少？

> 上一轮优化（第八节）生效后的新现象：数据集缩小，进度条 `16/6000 [00:27<2:12:38, 1.33s/it]`，
> 预计仍需 2~3 小时。本轮分析"每步 1.33 秒"是否还有压缩空间。

### 9.1 现状确认（脚本当前生效配置）

| 项 | 当前值 | 状态 |
|---|---|---|
| `max_length` | 1024 | ✅ 已降 |
| `gradient_checkpointing` | **False** | ✅ 已关（上轮建议的实验项，成功） |
| `enable_input_require_grads()` | 已加 | ✅（保留正确，兼容日后重开检查点） |
| 训练批量 | bs=1 × 累积 4（有效批量 4） | — |
| 评估 | bs=8、每 1000 步 | ✅ 已优化 |
| 总步数 6000 反推 | 6000 × 4 = 2.4 万条训练样本（应为原始 select 3 万条后 80/10/10 切分的结果） | — |

### 9.2 瓶颈判断：LoRA 训练的算力账（v2 修正）——仍接近硬件上限

> **修正说明**：本节原稿误用了全量微调的 `6ND` 公式。LoRA 冻结主干后，autograd 会跳过
> 冻结权重的梯度（dW）计算，正确的每 token 算力应为 **4ND** 而非 6ND。结论方向不变，
> 数字已按下表修正。

先澄清最容易算错的一点：**LoRA 冻结主干省的不是前向，而是"权重梯度"这部分反向算力，
以及优化器状态显存**：

| 阶段 | 全量微调 | LoRA 微调 | 原因 |
|---|---|---|---|
| 前向 | 2ND | 2ND | 前向必须过全部 0.6B 参数，LoRA 适配器（约 500 万参数）只多 ~1% |
| 反向·激活梯度 dX | 2ND | 2ND | 梯度必须流经所有层，才能传到每一层里的 LoRA 权重，**不能省** |
| 反向·权重梯度 dW | 2ND | ≈ 0 | 冻结权重 `requires_grad=False`，PyTorch autograd 跳过 dW，只算 LoRA A/B |
| **合计算力** | **6ND ≈ 3.6 GFLOP/token** | **4ND ≈ 2.4 GFLOP/token** | LoRA 比全量微调省约 1/3 反向算力 |
| 优化器状态显存 | Adam 2 份状态 × 0.6B ≈ 2.4GB | 仅 LoRA 参数 ≈ 20MB | **这才是 LoRA 省显存的主要来源** |

这也解释了一个常见疑问：为什么换 LoRA 后训练速度没有比全量快很多——**前向算力和激活
显存一点没少，只是反向少算了 dW 那一份**。

修正后的每步算力账（N≈0.6e9；有效批量 4 × 平均序列 ~700 ≈ 2800 token/步；bf16、无激活重算）：

```
每步算力 ≈ 2800 token × 2.4 GFLOP/token（LoRA 4ND）≈ 6.7 TFLOP
4050 Laptop bf16 有效训练吞吐 ≈ 4~7 TFLOPS（峰值十几，训练利用率典型 30~50%）
→ 理论单步耗时 ≈ 1.0 ~ 1.6 s
实测 1.33 s/it ≈ 5.0 TFLOPS ≈ 35~40% 利用率
```

**结论：修正数字后，1.33 s/it 仍属这块卡跑 0.6B × 2800 token/步的正常水平**（35~40% 的
利用率正是训练的典型值），不是代码写错或配置不当；且实测值已落在估算区间中上沿，说明
9.3 的软件优化收益预期维持 10~20% 不变，想大幅缩短只能减少总工作量（9.4）。

自查方法：训练时另开终端跑 `nvidia-smi -l 1`，若 GPU 利用率持续 >90% 即为算力瓶颈，
此时软件优化空间只剩"减少无效开销"。

附带说明：WSL2 走 CUDA 直通比原生 Linux 慢约 5~10%，这部分无法消除；脚本开头的数据加载/
分词（已命中 arrow 缓存）只在启动时执行一次，不计入 1.33 s/it。

### 9.3 还能榨的空间（按性价比排序）

#### ① `per_device_train_batch_size=2` + `gradient_accumulation_steps=2`（可试，大概率 OOM）

bs=1 时 GPU 利用率不满（小模型 kernel 启动开销占比高），翻倍批量能提升吞吐 10~20%。
**但 6G 卡上希望不大**——显存大头不是模型（bf16 权重仅 1.2GB），而是 **15.2 万词表的 logits**：

```
logits 张量 [bs, 1024, 151936]
  bs=1: bf16 0.6GB + CE损失 fp32 副本 1.25GB ≈ 1.9GB   ← 现在能跑的原因
  bs=2: ≈ 3.7GB + 激活 ~2GB + 权重 1.2GB + CUDA上下文 ≈ 撑爆 6GB
```

可以试，OOM 就退回 bs=1（无副作用）。

#### ② `torch_compile=True`（实验性，收益 10~20%）

`TrainingArguments(torch_compile=True)` 让 torch 把前向图编译融合。代价：首次编译要等几分钟、
PEFT/动态序列长度可能触发图重编译；WSL2 + torch 2.8 + transformers 4.55 组合**不保证顺利**。
其他手段用尽后再试，出现编译报错直接移除即可。

#### ③ 杂项（各百分之几，顺手改）

- `save_steps=200 → 500`：6000 步要存 30 次 checkpoint，虽然 LoRA 文件小，拉长间隔减少 I/O 打断；
- 确认注意力实现是 SDPA（transformers 4.55 + torch 2.8 默认已启用，可 `print(model.config._attn_implementation)` 验证）；
- 验证集若超过 1000 条可再缩——每 1000 步一次 eval，量大了会累积成分钟级开销。

#### ④ 明确不建议的"提速"手段

- **削减 `target_modules`（如只留 q/v）或降低 r**：这是以改变实验设计换速度，破坏学习目标；
- **fp16 换 bf16**：Ada 架构上两者吞吐相同，无收益；
- **`group_by_length=True`**：只在 bs>1 时有意义，bs=1 无效果。

### 9.4 真正有效的一档：减少总工作量

软件手段全用上大约只能省 20~30%，且都带不确定性。按 9.2 的算力账，**时间 ∝ token 总量**，
最直接的杠杆是样本数：

| 方案 | 预计耗时 | 说明 |
|---|---|---|
| 2.4 万条跑完（现状） | ~2.5 小时 | 单 epoch 完整收敛 |
| 训练集减半至 1.2 万条 | ~1.2 小时 | LoRA 学习实验完全够看 loss 收敛与生成效果 |
| 2.4 万条 + 提前停止 | 视监控 | zhihu 这类语料 loss 通常几千步即趋平，TensorBoard 看到平台期即可 Ctrl+C |

学习场景推荐：**先用 1 万条左右小集跑通全流程、观察 LoRA 是否有效**，确认后再决定是否值得
花 2~3 小时跑大集合。

### 9.5 预期效果小结

| 手段 | 预期收益 | 风险 |
|---|---|---|
| bs=2 + 累积 2 | 10~20%，或直接 OOM | 高（15 万词表 logits 是显存大头） |
| `torch_compile=True` | 10~20% | 中（编译耗时、兼容性） |
| save_steps 拉长等杂项 | ~5% | 无 |
| 训练集 2.4 万 → 1.2 万条 | **~50%** | 无（学习目标允许的前提下） |

### 9.6 注意事项

- 本节同样**仅为分析文档，未改动任何脚本**；
- 若测试 bs=2 触发 OOM，恢复 `per_device_train_batch_size=1` + `gradient_accumulation_steps=4` 即可，
  显存不会留下残余（进程退出即释放）；
- `nvidia-smi` 空闲时显示低功耗低占用属正常，判断瓶颈必须在**训练进行中**观察。

---

## 十、如何评估 LoRA 微调效果（对比 BERT 的前后 evaluate 做法）

> 背景：用 100 条数据快速跑通了微调，查看 `generate_sample` 输出后产生两个疑问：
> ① 怎么看出微调前后的差异/提升？② BERT 线是"微调前后各跑一次 evaluate"清晰对比，
> Qwen + 自定义 zhihu 数据集能否照搬？

### 10.1 先纠正一个关键问题：当前的生成评估 prompt 拼错了

先看实际输出里隐藏的异常（sample 0）：

```
<|im_start|>user
刚接触摄影要拍摄运动会，请大佬们给些拍摄想法？<|im_end|>
<|im_start|>assistant
人山人海的运动会非常能提升摄影师的摄影水平 ... 可以利用构图和景深<|im_end|>   ← 第一段
<|im_start|>assistant
好的，用户刚接触摄影，想要拍摄运动会...                                      ← 第二段（风格完全不同）
```

**第一段不是模型生成的，是数据集里的金标答案（RESPONSE）本身。** 原因在 `generate_sample`：

```python
input_ids = torch.tensor(batch["input_ids"][idx:idx+1])...
```

`tokenized_datasets["validation"]` 里每条样本是 `tokenize_func` 拼好的**完整对话**
（`user_str + RESPONSE + <|im_end|>`），把它整体当生成提示，输出就变成
"提示词（含金标答案）+ 模型续写"：

- sample 0 第一段 = 金标答案（来自提示，被原样解码出来）；第二段"好的，用户刚接触摄影……"
  才是模型生成的（Qwen3 基座的说教式续写风格）；
- "生成到 `<|im_end|>` 不停"的假象也随之解释：模型本地 `generation_config.json` 的
  `eos_token_id: [151645, 151643]` 是**正确的**（151645 即 `<|im_end|>`），但那个
  `<|im_end|>` 出现在**提示**里而非生成序列里，生成阶段不会触发停止；模型续写开出的
  新回合 `<|im_start|>assistant` 更不是停止符，一直跑满 `max_new_tokens=256`。

**结论：当前输出无法用于评判微调效果——生成提示必须只含指令部分（user_str），
不含答案。** 这是讨论"怎么看提升"之前必须先修的前提（见 10.4 方案 B）。

> **勘误**：此前此处写过"sample 0 第二段能侧面看到微调学到短答+`<|im_end|>`模式"——不准确。
> 第二段恰恰是**模型生成的**部分，它呈现的是基座式的说教长文续写且没有闭合；而第一段的
> "短答+`<|im_end|>`"是金标数据本来的样子，不构成微调生效的证据。100 条样本的微调对生成
> 风格的实际影响，必须按 10.4 修正评估口径后才能下结论。

#### 追问：为什么 sample 0 出现两对 `<|im_start|>assistant / <|im_end|>`，sample 1 只有一对？

差异不在模型，而在**两条样本的提示文本结构不同**——根源是金标答案长度不同，导致
`max_length=1024` 截断对两条样本的影响完全不同：

| | sample 0（摄影问题） | sample 1（OceanBase 问题） |
|---|---|---|
| 金标答案长度 | 几十字，完整保留在提示里 | 千字级，远超 1024 token 上限 |
| 提示文本形态 | 完整回合：`<|im_start|>assistant\n{金标}<|im_end|>` **正常闭合** | `truncation=True` 从右截断，**第一个被切掉的正是补在答案末尾的 `<|im_end|>`**——回合未闭合，文本停在半句 |
| 模型续写的起点 | 一个"已闭合回合"之后 | 一个"未闭合回合"的半句中间 |
| 续写行为 | 依训练分布另起新回合：生成 `<|im_start|>assistant\n好的，用户...`，256 token 预算耗尽仍未闭合 | 顺理成章把半句写下去；预算耗尽前从未生成终止符，因此全篇看不到第二个回合标签 |
| 解码结果 | 两段 assistant：第一段=金标（**有**闭合符），第二段=生成（**无**闭合符，严格说第二"对"并不完整） | 一整段 assistant：前 ~990 token=金标截断版，后 ~256 token=模型生成，接缝藏在段落中间不可见 |

两个推论：

1. **输出里的标签结构完全由提示文本决定，与模型能力无关**。sample 1 看似"模型独立写出了
   一大篇完整回答"，其实绝大部分是提示里的金标答案截断版，模型只贡献了最后 ~256 token；
2. 顺带暴露一个训练侧隐患：**超长样本被截断后，答案永远没有终止符**。这类样本训练时教给
   模型的是"长答案写到一半也不结束"，会加重"生成停不下来"的倾向。按 10.4 修正评估口径后，
   若仍观察到不停止，可考虑对超长样本做"整条丢弃"或"按回答边界截断"的数据处理。

#### 再追问："前 ~990 token = 金标截断版，后 ~256 token = 模型生成"这个数字怎么看出来的？

不是逐字符数出来的，是由**两个硬性 token 预算**推算的，并且能用代码精确验证：

1. **提示部分恰好 1024 token**：`tokenize_func` 里 `truncation=True, max_length=1024`——
   超长文本会被**截到恰好 1024 token**（不是"大约"，是精确值）。sample 1 金标答案远超 1024，
   所以它的提示必然是整 1024 token：模板 + 问题（约 30 token）+ 金标答案前段（1024 − 30 ≈ 990）；
2. **生成部分恰好 256 token**：`generate(max_new_tokens=256)` 是硬预算。sample 1 的续写停在
   半句（"……每个节点的处理"）说明没触发 eos 停止，即 256 个 token 一个不少地跑满了；
3. **总序列 = 提示 + 生成**：decoder-only 的 `generate()` 返回结果是"输入提示原样保留在前，
   新生成 token 接在后面"，而脚本对整个返回序列做了 decode——所以解码文本天然由这两段拼成；
4. **旁证（独立口径交叉验证）**：粘贴出的这段文字约 1800~2000 字，按 Qwen 中文分词约
   1.4~1.5 字/token 折算 ≈ 1250~1300 token，与 1024 + 256 = 1280 token 吻合。

精确验证——接缝位置在代码里可以直接切出来：

```python
prompt_len = inputs["input_ids"].shape[1]                 # 提示 token 数（sample 1 应 = 1024）
print("生成 token 数 =", output[0].shape[0] - prompt_len)  # sample 1 应 = 256（跑满预算）
print(tokenizer.decode(output[0][:prompt_len]))            # 前段：提示 = 金标截断版
print(tokenizer.decode(output[0][prompt_len:]))            # 后段：模型真正生成的部分
```

（~990 中的"~"来自"1024 − 模板与问题长度"这个减法是估算；256 是精确值。）

### 10.2 问题 1：生成式任务怎么衡量"微调带来的提升"

判别式（BERT/GLUE）有 accuracy/F1 单一指标；生成式（zhihu 问答）没有，需要**构造对比**。
三个前提 + 三条途径：

**前提（不满足则对比无效）：**

1. **提示只含指令**：prompt = `<|im_start|>user\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n`；
2. **可复现的解码**：对比时用 greedy（`do_sample=False`）或固定 `torch.manual_seed`——
   当前 `do_sample=True, temperature=0.7` 下同一模型两次生成都不同，前后差异可能是采样噪声；
3. **同一批测试样本**：固定选 validation 里同样的 N 条（100 条数据规模下取 10~20 条即可）。

**途径：**

| 途径 | 内容 | 对应 BERT 线的什么 |
|---|---|---|
| A. eval_loss 前后对比 | 微调前后各跑 `trainer.evaluate()`，比 eval_loss | **与 BERT 的做法完全同构**，最推荐 |
| B. 生成样例 + ROUGE | 同一提示分别用基线/微调模型生成，人工对比 + ROUGE-L 定量 | BERT 没有的，生成式特有 |
| C. loss 曲线 | TensorBoard / `trainer.state.log_history` 看训练过程 | 训练过程观测 |

### 10.3 问题 2：能否像 BERT 那样"前后各 evaluate 一次"？——能，而且更顺

完全可以照搬，且 Qwen 侧有一个比 BERT 更干净的性质：

**LoRA 的 B 矩阵初始化为 0** ⇒ `get_peft_model` 之后、训练之前的模型在数学上与 base 模型
完全等价（eval 模式下 dropout 也关闭）。因此：

- **训练前** `trainer.evaluate()` → 得到的 eval_loss **就是 base 模型在 zhihu 验证集上的基线**
  （不用额外加载原始 base 模型）；
- **训练后** `trainer.evaluate()` → 微调后的 eval_loss；
- 两者同结构、同数据、同 collator，直接可比——与 `test_Lora_bert.py` 的
  `trainer_baseline.evaluate()` → 训练 → `trainer.evaluate()` 结构一一对应。

eval_loss 的含义：模型对验证集"标准答案"的负对数似然（只在未被 mask 成 -100 的
回答部分计算）。**下降 = 模型更贴合 zhihu 回答的分布**。注意两点：

- 前后必须用**同一个**验证集（当前 select + arrow 缓存固定，天然满足）；
- 100 条的小数据容易过拟合：可能出现 train loss 一路降、eval_loss 先降后升——
  这恰恰是 eval_loss 对比的价值所在（帮助判断"该训练多少步"）。

另外发现一个相关配置问题：当前 `report_to="none"`，`logging_dir` 配了但**不会产生
TensorBoard 日志**。想走途径 C 需把 `report_to` 改为 `"tensorboard"`（或事后直接打印
`trainer.state.log_history`）。

### 10.4 修改方案草案（供讨论，未动原代码）

#### 方案 A：前后 eval_loss 对比（最小改动，推荐必做）

对齐 BERT 脚本结构，在现有代码上插入两处：

```python
# 位置：trainer 构造完成后、trainer.train() 之前
print("==== LoRA 微调前基线（B=0，等价 base 模型）====")
baseline_eval = trainer.evaluate()
print(baseline_eval)

print("开始 SFT 微调：")
trainer.train()

# train() 之后
after_eval = trainer.evaluate()
print("==== LoRA 微调后 ====", after_eval)
print(f"eval_loss 变化：{baseline_eval['eval_loss']:.4f} -> {after_eval['eval_loss']:.4f}")
```

#### 方案 B：生成对比（修 prompt + greedy，可叠加 ROUGE 定量）

改造 `generate_sample`：从 raw 验证集取指令重新拼 prompt，只含 user 部分：

```python
def generate_answer(instruction, max_new_tokens=256):
    user_str = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(user_str, return_tensors="pt").to(model_lora.device)
    with torch.no_grad():
        out = model_lora.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy：前后对比必须可复现
            pad_token_id=tokenizer.pad_token_id,
        )
    # 只解码新生成部分，去掉提示
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

# 前后对比：对同样的 N 条指令各生成一次
test_instructions = [raw_datasets["validation"][i]["INSTRUCTION"] for i in range(10)]
# 微调前调用一次存结果（基线版需在 train 前生成），微调后再生成一次，并排打印人工对比
```

要点：生成到 `<|im_end|>` 会自然停止（eos 配置已正确）；`skip_special_tokens=False`
保留 `<|im_end|>` 便于确认停止行为。若想定量，可用本地 vendored 指标算 ROUGE-L：

```python
import evaluate
rouge = evaluate.load("./datasets/evaluate/metrics/rouge/rouge.py")
# predictions=微调后生成列表, references=金标答案列表 → rougeL 与基线版对比
```

⚠️ 该指标依赖 `rouge_score`/`nltk` 包，离线环境若未安装会加载失败，届时退化为纯人工对比即可
（zhihu 开放式生成下 ROUGE 本身也只是参考指标，看相对变化而非绝对值）。

#### 方案 C：训练过程可观测（可选）

`report_to="none"` → `"tensorboard"`，训练后 `tensorboard --logdir ./training/qwen3-0.6b_Lora/logs`
看 train/eval loss 曲线；或不改配置，训练后直接 `print(trainer.state.log_history)`。

### 10.5 BERT 线与 Qwen 线评估方式对照表

| | BERT（判别式） | Qwen（生成式） |
|---|---|---|
| 基线评估 | `trainer_baseline.evaluate()`（原始权重） | `trainer.evaluate()`（LoRA B=0，等价 base） |
| 指标 | accuracy / F1（`./datasets/evaluate` 的 glue 指标） | eval_loss（主线）+ 生成样例 / ROUGE-L（辅助） |
| 对比口径 | 单次 evaluate 结果直接可比 | 必须 greedy/固定 seed + 相同测试集 |
| 额外注意 | — | **prompt 只含指令不含答案**（当前脚本的主要问题） |

### 10.6 修改方案完整实现（A 必做 + B 叠加，供手动修改）

> 本节是与 `test_Lora_qwen3-0.6b.py` 当前代码（`trainer` 构造在 201–208 行、
> `trainer.train()` 在 211 行、原评估段在 214–238 行）逐一对应的**完整可粘贴代码**。
> 10.4 是设计草案，本节是落地实现；共 3 处改动，原有的"保存 LoRA 适配器"段（241 行起）
> **保持不动**。

#### 改动总览

| 改动 | 位置（以当前脚本为锚） | 内容 |
|---|---|---|
| 改动 1 | `trainer = Trainer(...)` 之后、`print("开始 SFT 微调：")`（210 行）之前**插入** | 生成对比函数 + 测试样本准备 + 基线 eval_loss + 基线生成 |
| 改动 2 | `trainer.train()`（211 行）之后**插入** | 微调后 eval_loss + 微调后生成 |
| 改动 3 | 原"评估"段（214–238 行，从 `# ===== 评估 =====` 到 `print(text_out)` 的 for 循环）**整段替换** | 前后对比打印 + 可选 ROUGE |

#### 改动 1（插入）

```python
# ========== 改动1：微调前基线评估 + 生成对比准备 ==========
# 注意：本段必须在 trainer.train() 之前执行！
# LoRA 的 B 矩阵初始化为 0，此刻 model_lora 与 base 模型数学等价，
# 所以"微调前"的评估/生成直接用 model_lora 即可，无需再加载一份 base 模型。

def generate_answer(instruction, max_new_tokens=256):
    """方案B核心：只用指令构造提示（不含答案！），greedy 解码保证前后对比可复现"""
    user_str = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(user_str, return_tensors="pt").to(model_lora.device)
    model_lora.eval()   # 关掉 dropout(lora_dropout=0.1)，否则 train 模式下 greedy 也不可复现
    with torch.no_grad():
        output = model_lora.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # greedy：同输入必同输出，前后对比才有效
            pad_token_id=tokenizer.pad_token_id
        )
    # generate 返回 = 提示原样在前 + 新生成在后；切掉提示只解码新生成部分
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

# 固定同一批测试样本（微调前后必须完全一致）
N_COMPARE = 5    # 验证集不足5条就调小
test_instructions = [raw_datasets["validation"][i]["INSTRUCTION"] for i in range(N_COMPARE)]
test_references   = [raw_datasets["validation"][i]["RESPONSE"]    for i in range(N_COMPARE)]

# ---- 方案A：微调前基线 eval_loss ----
print("==== LoRA 微调前基线（B=0，等价 base 模型）====")
baseline_eval = trainer.evaluate()
print(baseline_eval)

# ---- 方案B：微调前基线生成（同样是 B=0 的模型在生成） ----
baseline_generations = [generate_answer(ins) for ins in test_instructions]
print("基线生成完成")
```

#### 改动 2（插入）

```python
# ========== 改动2：微调后评估 ==========
# ---- 方案A：微调后 eval_loss ----
print("==== LoRA 微调后 ====")
after_eval = trainer.evaluate()
print(after_eval)
delta = (after_eval["eval_loss"] - baseline_eval["eval_loss"]) / baseline_eval["eval_loss"] * 100
print(f"eval_loss 对比：{baseline_eval['eval_loss']:.4f} -> "
      f"{after_eval['eval_loss']:.4f}（{delta:+.1f}%）")

# ---- 方案B：微调后生成同一批指令 ----
after_generations = [generate_answer(ins) for ins in test_instructions]
```

#### 改动 3（整段替换原评估段）

```python
# ========== 改动3：前后对比打印 + 可选 ROUGE ==========
for i in range(N_COMPARE):
    print(f"\n======== 样本 {i} ========")
    print(f"【指令】{test_instructions[i]}")
    print(f"【微调前 base】{baseline_generations[i]}")
    print(f"【微调后 LoRA】{after_generations[i]}")
    ref = test_references[i]
    print(f"【金标参考】{ref[:200]}{'...' if len(ref) > 200 else ''}")

# （可选）ROUGE-L 定量对比。
# 两个注意点：
# ① vendored rouge 指标内部是英文分词（非 a-z0-9 字符全部过滤），中文直接算全为 0，
#    必须逐字加空格"伪装"成英文词后再算（字符级 n-gram 粗略指标，仅看相对变化）；
# ② 依赖 rouge_score / nltk 包，离线环境未安装会报错，try 包住可优雅跳过。
try:
    import evaluate as ev
    rouge = ev.load("./datasets/evaluate/metrics/rouge/rouge.py")
    zh = lambda texts: [" ".join(t) for t in texts]   # 中文逐字切开
    r_before = rouge.compute(predictions=zh(baseline_generations), references=zh(test_references))
    r_after  = rouge.compute(predictions=zh(after_generations),  references=zh(test_references))
    print(f"ROUGE-L：微调前 {r_before['rougeL']:.4f} -> 微调后 {r_after['rougeL']:.4f}")
except Exception as e:
    print("ROUGE 计算跳过：", e)

# ========== 以下原有的"保存 LoRA 适配器"代码保持不变 ==========
```

#### 预期输出（示意）

```
==== LoRA 微调前基线（B=0，等价 base 模型）====
{'eval_loss': 3.12, 'eval_runtime': ...}

======== 样本 0 ========
【指令】刚接触摄影要拍摄运动会，请大佬们给些拍摄想法？
【微调前 base】好的，用户刚接触摄影，想要拍摄运动会...（基座说教长文）
【微调后 LoRA】人山人海的运动会非常能提升...<|im_end|>（知乎短答风格，正常停止）
【金标参考】人山人海的运动会非常能提升摄影师的摄影水平 ...
ROUGE-L：微调前 0.0812 -> 微调后 0.2347
eval_loss 对比：3.1200 -> 2.4500（-21.5%）
```

（数值为示意；实际取决于训练数据量与超参。）

#### 验证清单

1. 基线生成的风格应明显"不像知乎回答"（基座说教体）——证明评估的确实是 base；
2. 微调后生成应出现 `<|im_end|>` 并停止——提示只含指令后，eos 逻辑才真正生效；
3. `eval_loss 对比` 打出定量变化；若**上升**，说明 100 条数据下过拟合（eval_loss 先降后升），
   属正常学习现象，恰好证明 eval_loss 对比的价值；
4. `ROUGE 计算跳过：...` 若出现，检查 `rouge_score`/`nltk` 是否安装，或直接以人工对比为准。

#### 注意事项

- **改动 1 的顺序是命门**：基线 evaluate 和基线生成必须在 `trainer.train()` 之前，
  顺序一旦调换，"基线"就变成"微调后"了；
- `generate_answer` 里的 `model_lora.eval()` 不能省——Trainer 训练循环会把模型切回
  train 模式，`lora_dropout=0.1` 会让 greedy 也产生随机性；
- 逐条生成 5 条 × 2 次最多几分钟；想提速可自行研究 batch 生成（需 left padding）；
- 方案 C 不在本组合内；想加训练曲线，把 `report_to="none"` 改为 `"tensorboard"` 即可。

---

## 十一、test_Lora_qwen3-0.6b.py 现状代码评审（2026-09-11）

> 对 A+B 组合落地后的脚本逐行检查，并实证了缓存目录与训练产物。结论：**核心训练链路正确，
> 可以放心跑**；发现 2 个中等优先级问题（数据采样、评估/保存步数空转）和若干注释/卫生问题，
> 均为改进建议，按需手动修改。

### 11.1 实证检查通过项 ✅

| 检查项 | 证据 | 结论 |
|---|---|---|
| 分词缓存指纹机制 | `train.arrow` 42GB → **668K**（val 70K / test 56K，17:08 更新） | `select(100)` 改变指纹后 `map` 自动重算并**覆盖**旧缓存——8.6 节"先删旧缓存"的建议是保险而非必需 |
| labels mask 逻辑 | 74–122 行：user 部分 -100、回答+`<|im_end|>` 参与 loss | 正确 |
| 基线评估/生成顺序 | 235/240 行都在 `trainer.train()` 之前 | B=0 基线成立 |
| `generate_answer` | 只拼指令、greedy、`model_lora.eval()`、切提示、传 `pad_token_id` | 五个要点齐全 |
| eos 停止 | `generation_config.json` 的 `eos_token_id: [151645, 151643]` | 提示只含指令后能正常停止 |
| ROUGE 防护 | try/except + 中文逐字切分 | 缺依赖不崩溃 |
| 适配器保存 | `adapter_model.safetensors` ≈ 20MB | ≈500 万 LoRA 参数 × fp32（peft 默认 fp32 初始化 A/B），合理；tokenizer/config 随 `processing_class` 一并保存 |

### 11.2 问题 1（中）：`select` 前没有 shuffle，有主题聚集风险

第 39 行 `select(range(100))` 取的是数据集**最前面的 100 条**。爬取类语料常按话题/时间组织，
前 100 条可能高度同质（比如全是某几类问题），导致 80 条训练 + 10 条验证都分布偏窄，
微调效果和 eval_loss 结论都会带偏。

```python
# 第 39 行，改为先 shuffle 再 select：
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(100))  # 打乱后取100条
```

`seed=42` 保证每次运行取到同一批（可复现性不破坏）。

### 11.3 问题 2（中）：`eval_steps=1000`/`save_steps=200` 对 20 步训练形同虚设

当前数据规模：select 100 → 80/10/10 切分 → **训练集只有 80 条**，有效批量 4 →
**每 epoch 仅 20 步、默认 1 epoch 共 20 步**。于是：

- `eval_steps=1000` → 训练中途**永远不会触发评估**（现在靠手动前后 evaluate 兜底，结果不受影响）；
- `save_steps=200` → **中途不会有任何 checkpoint**（好在 286 行 `trainer.save_model` 兜底保存了适配器）。

这两个参数目前是"空转"的。建议改为按 epoch，并顺手加 epochs 观察过拟合：

```python
    # 212-213 行，改为：
    eval_strategy="epoch",     # 每 epoch 评估一次，配合下方 epochs 能看到 eval_loss 曲线
    # eval_steps=1000,         # epoch 模式下不再需要
    save_strategy="epoch",     # 每 epoch 存一次 checkpoint（save_total_limit=3 自动滚动）
    num_train_epochs=3,        # 可选：80条数据跑3个epoch，观察 eval_loss 是否"先降后升"（过拟合信号）
```

这与方案 A 正好呼应：**"微调前 vs 微调后"只能给两个点，"每 epoch 评估"才能给出曲线**，
是判断"训练多久合适"的直接依据。

### 11.4 问题 3（低）：注释勘误表（学习仓库里错误注释比代码错误更误导）

| 行号 | 当前注释 | 实际情况 |
|---|---|---|
| 39 | `# 100条训练` | select 100 后还要 80/20 切分，**训练集实际 80 条** |
| 55 | `# train ~805k；validation ~100k；test ~100k` | 现在是 80 / 10 / 10 |
| 92–93 | `# 固定长度4096`、`# padding="max_length"：不足4096补pad` | `max_length=1024`，且**根本没有 padding**——变长存储，动态 padding 由 collator 完成（这正是缓存不膨胀的原因） |
| 204 | `gradient_checkpointing=False` 行尾注释"✅ 梯度检查点，大幅降低激活显存" | 已关闭，注释应改为"已关闭（1024长度+bs=1下显存足够，省30~40%时间）；OOM则改回True" |

### 11.5 问题 4（低）：训练产物与当前脚本可能不同步，旧 checkpoint 需清理

实证发现 `output/` 下三个 checkpoint 时间戳分属两批：

- `checkpoint-600` / `checkpoint-800`（16:41/16:47）——**2.4 万条旧配置的残留**，与当前代码无关；
- `checkpoint-60`（17:10）与 `lora_adapter/` 同刻——最近一次运行。**但注意**：当前配置
  （80 条 × 1 epoch）总共只有 20 步，60 步说明**当次运行的配置与现在文件里的不完全一致**
  （比如当时设了 `num_train_epochs=3`，或训练集还不是 80 条）。

两个习惯建议：

```bash
# 1) 清理旧 checkpoint，避免以后加载/对照时混淆
rm -rf ./training/qwen3-0.6b_Lora/output/checkpoint-600 ./training/qwen3-0.6b_Lora/output/checkpoint-800

# 2) 修改脚本后重跑前，先清掉 output/ 旧产物（保留 lora_adapter 另行处理）
```

以及一个学习点：**`checkpoint-60/trainer_state.json` 里保存着当次运行的完整超参快照和
loss 日志（log_history）**——`cat` 一下就能核对当时到底用了什么配置、loss 怎么变化的，
比猜可靠得多（这也是不开 TensorBoard 看训练曲线的最简途径）。

### 11.6 问题 5（低）：ROUGE 计算未剔除特殊 token

`generate_answer` 用 `skip_special_tokens=False` 解码（人工对比时能看到 `<|im_end|>` 是优点），
但这些特殊标记文本混进了 ROUGE 预测文本，成为噪声。最小修改（第 275 行）：

```python
    zh = lambda texts: [" ".join(t.replace("<|im_end|>", "").replace("<|im_start|>", "")) for t in texts]
```

影响很小（特殊标记只占个别 token），属严谨性改进。

### 11.7 修改优先级建议

1. **先改 11.2（shuffle）+ 11.3（eval/save 按 epoch + epochs=3）**——直接影响实验结论质量；
2. 11.4 注释勘误顺手改掉；
3. 11.5 清理旧 checkpoint、重跑前清 output；
4. 11.6 可改可不改。

改完后预期行为：训练 3 epoch × 20 步 = 60 步，每 epoch 末评估并保存 checkpoint（保留最近 3 个），
训练结束打印 eval_loss 前后对比、5 条前后生成对比和 ROUGE——**如果 eval_loss 在第 2~3 个
epoch 回升，就是 100 条数据下过拟合的直接证据**（这正是 11.3 想让你观察到的现象）。

---

## 十二、`ValueError: expected sequence of length 62 at dim 1 (got 1024)` 修复

> 报错位置：`baseline_eval = trainer.evaluate()`，崩在 `DataCollatorForLanguageModeling`
> 内部的 `tokenizer.pad()` → 张量转换。本节一并挖出了**比崩溃更严重的两个隐藏 bug**。

### 12.1 报错根因（源码级）

报错信息本身已经指明方向：`input_ids`/`attention_mask` 都转换成功了（说明 pad 对它们生效），
唯独 **`labels` 长度参差（62 vs 1024）**——即 pad 根本没处理 labels。查已安装的 transformers 源码：

**证据 1 —— `tokenization_utils_base.py` 的 `_pad` 只填充这些键：**

```python
if return_attention_mask: ...            # attention_mask 补 0
if "token_type_ids" in encoded_inputs: ...
# input_ids 补 pad_token_id
# special_tokens_mask 补 1
# —— 没有任何 labels 的分支，labels 原样透传
```

**证据 2 —— `data_collator.py` 的 `DataCollatorForLanguageModeling.torch_call`：**

```python
if isinstance(examples[0], Mapping):
    batch = pad_without_fast_tokenizer_warning(
        self.tokenizer, examples, return_tensors="pt", ...   # 含自定义labels的features直接丢给pad
    )
```

链条：数据集自定义的 `labels` 列 → 原样进入 `tokenizer.pad` → pad 不填充它 →
eval 批量 bs=8 里混着 62 token（短答案）和 1024 token（截断长答案）的样本 →
`torch.tensor(参差列表)` 崩溃。

### 12.2 为什么现在才炸 + 两个更严重的隐藏 bug

**为什么之前训练没崩**：训练 `bs=1`，每个批只有 1 条样本，labels 长度天然一致，永不触发；
而 `eval_steps=1000` 大于训练总步数，**训练中途的 eval 从未真正运行过**——今天加的
`baseline_eval = trainer.evaluate()`（bs=8）才是第一次多样本评估，第一次就踩中。

**隐藏 bug ②（静默！）：数据集的 labels 会被丢弃覆盖。** 同一个 `torch_call` 的 `mlm=False` 分支：

```python
else:
    labels = batch["input_ids"].clone()          # ← 用 input_ids 重建 labels
    if self.tokenizer.pad_token_id is not None:
        labels[labels == self.tokenizer.pad_token_id] = -100
    batch["labels"] = labels                     # ← 无条件覆盖，数据集的自定义 labels 没了
```

也就是说：**到目前为止的所有训练，`tokenize_func` 里精心构造的"prompt=-100 只学回答"mask
根本没有生效**——模型实际在"完整序列（问题+回答）"上计算 loss（等价于续练语料，而不是 SFT）。
bs=1 下 labels 长度一致、不崩，所以这个覆盖一直是**静默**发生的。

**隐藏 bug ③（呼应 10.1！）：停止符的监督被抹掉。** 本脚本 `pad_token = eos = <|im_end|>`，
于是 `labels == pad_token_id → -100` 把**所有 `<|im_end|>` 位置的监督清零**——模型从头到尾
没被训练过"在回答结束处输出 `<|im_end|>`"。这正是 10.1 观察到"生成停不下来"的深层原因
之一：即使 prompt 只含指令，模型对停止符的可预测性也没被强化过。

**对 eval_loss 口径的影响**：之前的 baseline/after eval_loss 都是"全序列 loss"而非设计的
"仅回答部分 NLL"。修复后口径变化，**前后对比需要用修复后的代码重新跑一遍**。

### 12.3 修复：换回自定义 collator（替换第 146–149 行）

讽刺的是，脚本最初被三引号注释掉的 `custom_collate_fn` 思路才是对的——它自己处理三个键的
padding，不依赖 `tokenizer.pad`。完整替换代码：

```python
# ===================== 训练 =====================
# ⚠️ 不能用 DataCollatorForLanguageModeling，三个坑（详见 docs 第十二节）：
#   ① tokenizer.pad 不填充 labels，eval批量>1时长度参差会崩
#   ② mlm=False 分支用 input_ids 覆盖 labels，prompt=-100 的 mask 静默失效
#   ③ labels==pad_token_id(即<|im_end|>)被改成-100，模型学不到停止符
def custom_collate_fn(features):
    """动态padding到batch内最长：input_ids补pad_token_id、attention_mask补0、labels补-100"""
    max_len = max(len(f["input_ids"]) for f in features)
    return {
        "input_ids": torch.tensor(
            [f["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(f["input_ids"])) for f in features],
            dtype=torch.long),
        "attention_mask": torch.tensor(
            [f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features],
            dtype=torch.long),
        "labels": torch.tensor(
            [f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features],
            dtype=torch.long),
    }

data_collator = custom_collate_fn
```

要点：

- `labels` 用 **-100** 补齐：pad 位置不参与 loss；数据集里已有的 -100（prompt 部分）原样保留；
- `<|im_end|>` 的 label（151645）**不再被抹**，模型重新获得停止符监督；
- Trainer 默认 `remove_unused_columns=True` 会把 INSTRUCTION/RESPONSE 等非模型输入列剔除，
  collator 实际只收到三个键，无需额外处理。

### 12.4 临时验证手段（不是修复！）

把 `per_device_eval_batch_size` 改回 1 能让崩溃消失（单样本批没有参差问题），但 bug ②③
在训练中**依然静默生效**——只用于快速确认诊断，不要当成解决方案。

### 12.5 修复后的验证清单

1. `baseline_eval = trainer.evaluate()` 不再崩溃；
2. eval_loss 数值与修复前**不可比**（口径从全序列变为仅回答部分），前后对比用修复后代码重跑；
3. 训练后生成应能在 `<|im_end|>` 处正常停止（这次停止符真的被训练了）——可与 10.6 的
   前后对比输出互相印证；
4. 对比一个 eval batch 的形状：`batch["input_ids"].shape == batch["labels"].shape`，
   且 labels 在 pad 位置全为 -100、在 prompt 位置全为 -100、在回答位置为真实 token id。

### 12.6 落地检查（第二轮评审，2026-09-11）

对照第十二节逐行复查修改后的脚本：

| 项 | 状态 |
|---|---|
| 12.3 自定义 collator | ✅ 实现正确：labels 用 -100 补齐、`<|im_end|>` 监督恢复、attention_mask 补 0 |
| 11.2 shuffle | ✅ `.shuffle(seed=42).select(range(100))` 已加 |
| 11.3 eval/save 按 epoch | ✅ 已改；但 **`num_train_epochs` 未加**（默认 1 epoch → 整个训练只在末尾评估/保存一次，看不到"eval_loss 曲线"）；想观察过拟合需 ≥2 epochs |
| 11.6 ROUGE 剔除特殊 token | ✅ 已加 `<|im_end|>`/`<|im_start|>` replace |
| **`import torch`** | ❌ **丢失！第 16 行原 `import torch` 被 `import evaluate as ev` 替换掉了** |

**阻断性问题（必须修）**：`custom_collate_fn`（160–168 行）和 `generate_answer`（202 行
`torch.no_grad()`）都直接使用 `torch`，但全文件已无 `import torch`。第一次调用
`trainer.evaluate()` 进 collator 时必然抛 `NameError: name 'torch' is not defined`。

```python
from datasets import load_dataset, DatasetDict
from typing import List, Dict
import torch                 # ← 补回这一行
import evaluate as ev
```

其余小项（可选，不影响运行）：

1. 第 92 行注释仍残留"固定长度4096"（实际 max_length=1024 且无 padding）；
2. 第 39 行注释"100-20条训练"表述不清，建议改"shuffle后取100条，切分后训练集80条"；
3. 未使用的 import 可清理：第 10 行 `DataCollatorForLanguageModeling`（已注释弃用）、
   第 15 行 `List, Dict`（旧版 collator 的类型标注，新 collator 未用）；
4. `zh` lambda 可再补 `replace("<|endoftext|>", "")`——eos 列表里有 151643，万一模型以它
   结束，该字样会混进 ROUGE 文本（影响极小）。

---

## 十三、eval 阶段 CUDA OOM（Tried to allocate 4.64 GiB）修复

> 报错位置：`baseline_eval = trainer.evaluate()` 一路走到
> `loss_utils.ForCausalLMLoss → fixed_cross_entropy → F.cross_entropy` 处申请显存失败。
> 关键信息：`Tried to allocate 4.64 GiB`，`total capacity of 6.00 GiB`，
> `8.99 GiB is allocated by PyTorch`。

### 13.1 报错解读：崩在 loss 计算，不是模型前向

栈底是**交叉熵**而不是层前向——这正是 15.2 万词表的 CLM 特有的显存大头：

```
logits 张量 [bs, seq, 151936]，loss 计算路径的峰值显存 ≈ bs × seq × 151936 × 10 B：
  bf16 logits            bs × seq × 151936 × 2 B
  CE 前 logits.float()   bs × seq × 151936 × 4 B
  CE 内部 log_softmax    ≈ 再来一份 fp32

eval bs=8、批内有一条 ~1024 长样本把整批 pad 到 1024：
  8 × 1024 × 151936 × 2B ≈ 2.5 GB   （bf16 logits）
  + 5 GB                （fp32 上转副本）
  + ~4.6 GB             （CE 内部——正是报错要分配的 4.64 GiB）
  ≈ 12 GB → 6G 卡必死
```

两个顺带说明：

- 报错里"6GB 卡却已分配 8.99GB"是 **WSL2 共享内存回溢**现象：驱动先把溢出部分放进系统
  内存，速度骤降并最终 OOM，不是显存真的超过 6G；
- "eval 无梯度所以批量可比训练大"只对**激活/反向**成立——**loss 计算的 logits/CE 显存与
  batch 严格成正比**，而它在 15 万词表模型里占绝对大头。

> **勘误声明**：第八节 8.3/8.4 推荐的 `per_device_eval_batch_size=8` 是估算错误
> （漏算了词表维度），已在原文处加 ⚠️ 标注，正确做法见本节。

### 13.2 修复（一行，第 229 行）

```python
    per_device_eval_batch_size=1,   # ⚠️ 15万词表下 bs=8 的 loss 计算需 ~12GB；bs=1 仅 ~1.6GB
```

验证集只有 10 条，bs=1 也就 10 次前向，速度完全无感。修复后 eval 峰值 ≈
权重 1.2GB + KV cache 0.12GB + loss 路径 ~1.6GB ≈ 3.5GB，稳定。

（训练侧 bs=1 无需动：同样的 loss 路径 bs=1 只要 ~1.6GB，之前 60 步已验证跑得通。）

### 13.3 可选加固

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 缓解显存碎片（报错提示同款）
```

不建议为此把 `model.config.use_cache=False`：eval 的 teacher-forcing 前向里 KV cache
确实纯浪费（bs=8 时约 0.9GB），但全局关闭会拖慢脚本末尾 `generate_answer` 的逐 token
生成（256 token 需要复用 KV cache），改来改去顾此失彼——bs=1 修复后已无必要。

### 13.4 经验总结：CLM 显存的第一大头是"词表 × seq × batch"的 loss 计算

1. 以后估 batch 上限：**loss 路径峰值 ≈ bs × seq × 词表 × 10 B**（bf16 权重 + fp32 loss
   路径）。6G 卡、seq=1024、词表 15.2 万 → bs 上限约 2~3（还要留出权重和激活）；
2. 模型权重（bf16 1.2GB）反而只是小头——"0.6B 的小模型"显存压力不在参数量在词表量；
3. 若未来验证集变大、需要评估吞吐，再考虑按长度分桶（`group_by_length`）或分片评估；
   当前 10 条规模无需任何额外手段。

---

## 十四、训练结果评估（80 条 × 3 epoch × 60 步）与 generation flags 警告检查

> 首次完整跑通：基线评估 → 基线生成 → 3 epoch 训练（每 epoch 评估）→ 微调后评估 →
> 前后生成对比 → ROUGE（跳过）→ 保存适配器。本节回答两个问题：训练到底有没有效？
> 中间反复出现的 "generation flags are not valid" 是不是错误？

### 14.1 结论：微调有效，学到的是"格式与风格"，不是知识

**定量证据（eval_loss，仅回答部分 NLL）：**

| 阶段 | eval_loss | 说明 |
|---|---|---|
| 基线（B=0 ≡ base） | 4.8659 | 困惑度 e^4.87 ≈ 130 |
| epoch 1 | 4.5854 | **主要收益在这一步拿到** |
| epoch 2 | 4.5400 | 边际递减 |
| epoch 3 | 4.5191 | 收敛迹象；累计 **-7.1%** |

- eval_loss **三个 epoch 单调下降**：没有过拟合（11.3 想观察的曲线形态是"先降后升"，
  80 条数据 3 epoch 还没到过拟合点）；
- 其他健康信号：`grad_norm` 稳定在 1.0~1.4、学习率线性衰减正常、微调后手动 evaluate 的
  数值（4.519139766693115）与 epoch 3 自动评估**逐位一致**——管线自洽；
- 收益结构：epoch 1 拿走了大部分提升 → **想再涨要加数据，不是加 epoch**。

**定性证据（这才是 SFT 生效的直接展示）：**

1. **`<|im_end|>` 正常停止**（样本 0、4 生成到 `<|im_end|>` 干净收尾）——基线 5 条全部
   跑满 256 token 不停。这是第十二节 collator 修复（停止符监督恢复）的**直接验证**；
2. **`</think>` 学会闭合**：基线是"一直 think 到被截断"，微调后能"想完 → 给出短答"；
3. **输出形态从说教长文变为知乎式短答**，与训练数据的风格方向一致。

**同样重要的反面结论：知识没有被学到（也不应该期待）：**

- 样本 4"三种开球方法：单手开球、双手开球和三手开球"是**胡编**（斯诺克没有这种分类）；
- 样本 0"念念不忘的文章是《一个》"是退化答案；
- 80 条 × LoRA r=8 × 3 epoch 的容量，本来就只够学"输出长什么样"，学不了"内容对不对"。
  这正是小型学习实验的正常边界。

**已知瑕疵（非 bug）：**

- 样本 1/3 的**重复坍塌**（"回答要分点，不要用任何其他格式"×20、"女学生"×20）——
  贪心解码 + 小数据 LoRA 的典型病：一旦掉进高概率循环就出不来；
- 样本 2 仍有 think 模式泄漏（训练数据没有 think 块，但没完全压掉）；
- ROUGE 跳过是**预期行为**（离线环境缺 `rouge_score`/`nltk`，try/except 按设计兜住）。

**顺带一个背景认知**：基线生成以 `<think>` 开头——说明本地 `Qwen3-0.6B` 是 HF 上的
**instruct（thinking）版**（`Qwen3-0.6B-Base` 才是纯基座）。这解释了基线的说教风格，
也意味着本次微调相当于"把 thinking 模型往知乎短答风格拉"。

### 14.2 "generation flags are not valid" 警告：不是错误，解码行为正确

**现象**：每次 `generate_answer` 前打一条，共 10 次（5 样本 × 微调前后两轮）：

```
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']
```

**根因**：本地模型的 `generation_config.json` 带着官方推荐采样参数
（`do_sample=True, temperature=0.6, top_p=0.95, top_k=20`）。`generate_answer` 显式传了
`do_sample=False`（贪心）后，transformers 的配置校验器发现这 3 个**只在采样模式下有意义**
的参数仍然有值，于是提示"将被忽略"。已核对源码（`configuration_utils.py`）触发条件：
`do_sample=False` 且 `temperature≠1.0` / `top_p≠1.0` / `top_k≠50` → 记入警告列表。

**解码行为是正确的**：参数确实被忽略、贪心在生效——证据是同指令前后两次输出完全稳定可复现。
这属于"提示你配置里有冗余参数"的良性警告。

**修复（可选，消除警告）**——在 `generate_answer` 的 `generate` 调用里显式传"中性值"，
它们恰好是校验器的跳过阈值，贪心模式下本就不参与计算，**行为完全不变**：

```python
        output = model_lora.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,     # ← 中性值，消除警告；贪心模式下无效参数
            top_p=1.0,           # ←
            top_k=50,            # ←
            pad_token_id=tokenizer.pad_token_id
        )
```

不想改也完全可以——学习仓库里知道"为什么会有这条警告"比消除它更有价值。

### 14.3 本次实验的一句话总结

> 评估链路（基线 B=0 → 训练 → 前后对比）第一次给出了"微调是否有效"的**可量化答案**：
> eval_loss -7.1%、停止符与短答格式习得、知识不可期。实验设计（第十二节 collator +
> 第十节前后对比）至此完整闭环。

---

## 十五、"有已训练权重就直接用，否则才训练"——实现方案与代码

> 目标行为：`lora_adapter/` 目录里已经有训练过的适配器 → 加载它、跳过训练，直接进入评估/
> 生成对比；没有 → 走完整训练流程并在结束时保存。

### 15.1 方案设计（三个关键决策）

1. **用什么判断"保存过"**：判断 `lora_adapter/adapter_config.json` 是否存在，而不是只判断
   目录存在——目录可能在保存中途被杀掉（残缺），配置文件才是"保存完成"的标志；
2. **加载用 `PeftModel.from_pretrained`，而不是 `get_peft_model`**：前者读回 adapter_config.json
   （含 r、target_modules、task_type=CAUSAL_LM）并载入**训练好的 A/B 权重**；后者是新建一个
   B=0 的空适配器。CLAUDE.md 的 `test_Lora-load_bert.py` 用的正是前者；
3. **加载模式下怎么拿 base 基线**：没有"B=0 时刻"可用了，但 PEFT 提供了
   **`disable_adapter()` 上下文管理器**——`with` 块内临时把适配器置零（数学上等价 base 模型），
   出块自动恢复。不用往 6G 显存里再塞一份 base 模型。这是本方案最值得学的技巧。

两条路径在后半段（微调后评估 → 前后对比 → ROUGE）**完全共用**：`baseline_eval` /
`after_eval` / `*_generations` 变量语义保持一致，下游代码零改动。

### 15.2 改动 1：文件顶部 imports（补 `os` 和 `PeftModel`）

```python
import os

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,        # 新增：加载已保存的适配器
)
```

### 15.3 改动 2：检测 + 分支（替换原 173–192 行的 peft_config/model_lora 段）

```python
# ===================== 训练 =====================
lora_save_path = "./training/qwen3-0.6b_Lora/lora_adapter"   # ← 从文件末尾上移到此处定义
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))

if has_adapter:
    # ---------- 模式一：加载已训练权重，跳过训练 ----------
    print(f"检测到已保存的LoRA适配器：{lora_save_path}，跳过训练")
    model_lora = PeftModel.from_pretrained(model, lora_save_path)
    # 默认 is_trainable=False：适配器权重 requires_grad=False，仅推理/评估
    # 此时 print_trainable_parameters 会显示 0% 可训练——这是预期的
    model_lora.print_trainable_parameters()
else:
    # ---------- 模式二：首次运行，完整训练流程 ----------
    peft_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        inference_mode = False,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias = "none",
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.1
    )
    model_lora = get_peft_model(model, peft_config)
    model_lora.enable_input_require_grads()   # 只在训练分支需要（配合梯度检查点）
    model_lora.print_trainable_parameters()
```

`generate_answer`、`TrainingArguments`、`Trainer` 三段**不改**——加载模式同样用
`trainer.evaluate()`，保证两种模式评估口径一致（同一 collator、同一验证集）。

### 15.4 改动 3：基线评估/训练段条件化（替换原 256–265 行）

```python
# 固定同一批测试样本（微调前后必须完全一致）
N_COMPARE = 5
test_instructions = [raw_datasets["validation"][i]["INSTRUCTION"] for i in range(N_COMPARE)]
test_references   = [raw_datasets["validation"][i]["RESPONSE"]    for i in range(N_COMPARE)]

if has_adapter:
    # 加载模式：用 disable_adapter() 临时关闭适配器得到 base 基线
    print("==== 基线评估（disable_adapter 临时关闭适配器 ≙ base 模型）====")
    with model_lora.disable_adapter():
        baseline_eval = trainer.evaluate()
    print(baseline_eval)
    with model_lora.disable_adapter():
        baseline_generations = [generate_answer(ins) for ins in test_instructions]
    print("基线生成完成（已训练权重模式）")
else:
    # 训练模式：B=0 数学上等价 base，直接评估/生成
    print("==== 预训练模型原始基线 ====")
    baseline_eval = trainer.evaluate()
    print(baseline_eval)
    baseline_generations = [generate_answer(ins) for ins in test_instructions]
    print("基线生成完成,开始 SFT 微调：")
    trainer.train()
```

其后的"微调后评估 → 对比打印 → ROUGE"整段**共用、零改动**（`after_eval = trainer.evaluate()`
在加载模式下评估的就是载入的权重）。

### 15.5 改动 4：保存段条件化（替换原 306–311 行）

```python
# ============ 保存LoRA适配器 ============
if not has_adapter:
    trainer.save_model(lora_save_path)
    print(f"LoRA适配器已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")
```

### 15.6 知识点与注意事项

| 点 | 说明 |
|---|---|
| `from_pretrained` vs `get_peft_model` | 前者=载入训练好的 A/B + 读回配置；后者=新建 B=0 空适配器 |
| `is_trainable=False`（默认） | 加载的适配器 `requires_grad=False`，只能推理/评估；**想"从上次继续训练"**：`PeftModel.from_pretrained(model, path, is_trainable=True)` 并让它走训练分支（可作为扩展练习） |
| `disable_adapter()` | `with` 块内适配器输出置零 ≙ base，出块恢复；是"同一份显存里的双面模型"，6G 卡上拿基线的正道 |
| 加载模式下 `eval_strategy="epoch"` | 不会触发（没有 `trainer.train()`），评估全部来自手动调用，无副作用 |
| dtype 细节 | 适配器权重载入后挂载在 bf16 主干上，LoRA 前向以更高精度计算后并回，无需手动处理 |
| 残缺目录 | 若上次保存被 Ctrl+C 打断，可能出现"有目录无 adapter_config.json"→ 本方案会正确地判定为未保存、重新训练 |

### 15.7 验证清单

1. **强制重训**：`mv ./training/qwen3-0.6b_Lora/lora_adapter lora_adapter.bak`（或删除）→
   运行 → 应走训练分支（60 步）→ 结束保存；
2. **立即重跑**：第二次运行 → 日志显示"跳过训练"，加载模式的 `eval_loss` 应与上次
   epoch 3 的 **4.5191 逐位一致**（同一权重、同一验证集、确定性评估）；
3. **加载模式下的基线**：`disable_adapter()` 的 eval_loss 应与上次**基线 4.8659 一致**——
   证明关闭适配器数学上等价 base；
4. **生成复现**：加载模式下样本 0/4 的生成与上次训练后输出**逐字一致**（greedy + 同权重）。

### 15.8 落地检查（2026-09-12）

对照第十五节逐行复查修改后的脚本，**实现正确、两种模式均可运行，无阻断性 bug**：

| 检查项 | 状态 |
|---|---|
| imports（`PeftModel` / `torch, os`） | ✅ |
| `has_adapter` 检测（`adapter_config.json` 存在性） | ✅ |
| 分支：加载模式 `PeftModel.from_pretrained` / 训练模式原流程 | ✅ `enable_input_require_grads` 已正确收进训练分支 |
| `generate_answer` 中性值（14.2 的警告修复） | ✅ 顺带落地，下次运行 generation flags 警告会消失 |
| 两条路径后半段共用，`baseline_eval` 两分支均有定义 | ✅ 第 303 行不会 NameError |
| 保存段条件化 | ✅ |

**一个值得知道的发现：磁盘上 `adapter_config.json` 里 `"inference_mode": true`**，
与训练时 `LoraConfig(inference_mode=False)` 不同——这不是异常：**PEFT 保存适配器时会统一
把标记写成 `inference_mode: true`**（"保存的适配器用于推理"的语义），加载后适配器正常生效，
也不影响 `disable_adapter()` 的切换。训练时写 False、保存后变 True 是 PEFT 的设计行为。

**非阻断的小瑕疵（建议顺手改）**：

1. 第 255–257 行注释已过期：仍写着"B=0 直接评估"（那是训练分支的行为），且挂在
   `generate_answer` 上方、描述的其实是下方 if/else。建议改为中性说明：
   `"基线评估在下方按 has_adapter 分两种模式：训练模式 B=0≡base 直接评；加载模式用 disable_adapter() 切回 base"`；
2. 第 286–287 行：`else:` 行尾多余空格、注释行缩进 5 格（Python 不报错，风格不一致）；
3. 第 95 行注释"固定长度1024"仍不准确——实际是**变长、上限 1024、不 padding**（padding 由 collator 动态做）；
4. `output/` 下旧 checkpoint（上轮 3-epoch 的产物）仍在，不影响运行，强制重训前可清理。

**当前磁盘状态与下次运行的预期**：`lora_adapter/` 存在（今天 16:00 保存，来自 collator
修复后的那次完整训练）→ **下次运行将进入加载模式**，可直接验证 15.7 全部判据：
打印"跳过训练" + 0% 可训练；`disable_adapter` 基线 = 4.8659（逐位）；适配器 eval_loss =
4.5191（逐位）、delta -7.1%；样本 0/4 生成逐字复现；警告消失；结尾"无需重复保存"。
想重新训练：`mv ./training/qwen3-0.6b_Lora/lora_adapter ./training/qwen3-0.6b_Lora/lora_adapter.bak`。

---

## 十六、全程经验教训与知识点总结

> 每条一句话，`§` 指本文对应小节。

### A. LoRA / PEFT 核心

1. PEFT 部分参数微调 + 梯度检查点（reentrant）必须打通梯度入口：`enable_input_require_grads()` 或 `use_reentrant=False`，否则 loss 没有 grad_fn（§1–7）
2. LoRA 的 B=0 初始化数学等价 base——训练前的评估/生成本身就是基线；`disable_adapter()` 让同一份显存在"base / 微调后"之间双面切换（§10、§15）
3. LoRA 训练算力是 4ND 不是全量的 6ND：省的是冻结权重的 dW 与优化器状态显存，前向和激活一点没省（§9.2）
4. 小数据 LoRA 学到的是**格式与风格**（停止符、短答、闭合 think），不是知识——样本量决定上限（§14.1）
5. `get_peft_model` 新建 / `PeftModel.from_pretrained` 载入；保存时 PEFT 自动把 `inference_mode` 置 true（正常现象）；`is_trainable=True` 可续训（§15.6）

### B. Trainer / 数据管道陷阱

6. SFT 带自定义 labels 时**必须用自定义 collator**：`DataCollatorForLanguageModeling` 有三坑——不 pad labels（eval 批量>1 崩）、无条件覆盖 labels（prompt mask 静默失效）、`pad_token==eos` 时把停止符监督抹成 -100（§12）
7. eval_loss 的口径由 collator 决定——collator 改动前后的数值不可比，对比要用同一套代码重跑（§12.2）
8. 改 `tokenize_func`/`max_length` 必须删 arrow 缓存；`select` 改变指纹会自动重算，删缓存是保险（§8.6、§11.1）
9. 参数名用 `eval_strategy`（新版），别写旧版 `evaluation_strategy`

### C. 显存与速度

10. CLM 显存第一大头是 loss 计算：峰值 ≈ **bs × seq × 词表 × 10B**——0.6B 小模型的压力在 15 万词表，不在参数量（§13）
11. eval 虽无梯度，loss 显存仍与 batch 严格成正比；定 batch 上限先套上面公式，再留权重和激活的余量（§13.4）
12. 时间 ∝ token 总量：缩数据 > 一切调参；提速前先 `nvidia-smi -l 1` 判断是否已到算力瓶颈（§9）
13. datasets 的 mmap 页缓存会顶满 WSL2 内存配额——`free -h` 里 `buff/cache` 高属无害，缩数据是根治（§8.2）
14. WSL2 三特性：CUDA 直通慢 5~10%、共享内存回溢（6G 卡可报"已分配 8.99G"）、`.wslconfig` 可封顶与回收（§8.2、§13.1）

### D. 评估方法论

15. 生成式前后对比三前提：**prompt 只含指令（不含答案）、greedy 或固定 seed、同批样本**——缺一结论无效（§10.2）
16. 训练有效性看三点：eval_loss 相对变化、格式/停止符是否习得、以及"知识不可期"的预期管理；收益通常集中在第一个 epoch（§14）
17. vendored ROUGE 是英文分词，中文要逐字加空格才非零；只看相对变化不看绝对值（§10.4）
18. 贪心解码的重复坍塌是小数据典型病，不是代码 bug；采样可缓解但对比实验必须保持 greedy（§14.1）

### E. 工程习惯

19. 错误注释比错误代码更害学习型仓库：注释必须与代码同步演进（§11.4）
20. 产物与代码版本可能不同步：checkpoint 时间戳 + `trainer_state.json` 能考古当次超参与 loss 曲线（§11.5）
21. 报错日志里常藏直接证据（"None of the inputs have requires_grad" 一行即根因）；良性警告也要查明原因再决定忽略（§14.2）
22. 离线环境对可选依赖用 try/except 优雅降级，不阻塞主流程（§10.4）

### 一句话总纲

> 本次踩过的所有坑几乎都来自三个交叉点：**冻结与建图**（PEFT × 检查点 × collator 里 labels
> 的流转）、**词表与规模**（loss 显存、数据量、页缓存）、**口径与复现**（基线怎么定、对比怎么
> 才可比、产物对不对得上代码版本）。下次换模型/数据/方法时，先按这三条线自查一遍。
