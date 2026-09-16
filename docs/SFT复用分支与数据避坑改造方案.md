# SFT 复用分支与数据避坑改造方案（test_SFT_qwen3-0.6b.py）

> 需求：给 `test_SFT_qwen3-0.6b.py` 补上①自定义数据集避坑（对齐 LoRA 线的采样做法）②复用上次训练成果分支（参考 `test_Lora_qwen3-0.6b.py` 的 has_adapter 双模式）。
> 现状：L240-242 是未完成的空分支（`if has_adapter:` / `else` / 空——直接运行会 SyntaxError）。
> **核心认知先行：SFT 的"复用"和 LoRA 的"复用"机制完全不同——SFT 没有适配器，保存/加载的是完整模型权重**。照抄 LoRA 的 `PeftModel.from_pretrained` 是行不通的，这正是卡住的原因。

---

## 一、SFT vs LoRA：复用机制的本质差异

| | LoRA 线（参考对象） | SFT 线（本脚本） |
|---|---|---|
| 训练时改什么 | 冻结主干，只训 adapter（~20MB） | **全部参数都在训** |
| `trainer.save_model(path)` 存什么 | 只存 adapter（`adapter_config.json` + safetensors 几十 MB） | **完整权重**（`model.safetensors` ≈ **1.2GB** + config + tokenizer） |
| 完成标志（has 判断） | `adapter_config.json` 存在 | `model.safetensors` 存在 |
| 加载方式 | `PeftModel.from_pretrained(base_model, path)`——base + 适配器两层 | **`AutoModelForCausalLM.from_pretrained(path)`**——直接重载整个微调后模型，无包装概念 |
| "切回 base 做对比" | 加载模式可用 `disable_adapter()` | **不可能**——全量微调直接覆盖了权重，没有开关；所以**基线评估必须在重载之前完成**（当前脚本的顺序已经满足：baseline 在前、加载分支在后，这是设计使然，不要调整顺序） |

> 变量命名建议同步正名：`lora_save_path` → `sft_save_path`、`has_adapter` → `has_model`、目录 `lora_adapter` → `full_model`。避免重蹈 BERT 线"Lora_bert 目录混写"的覆辙。

---

## 二、逐段实现方案

### 2.1 数据避坑（对齐 LoRA 线口径，三处改动）

**① L34 前加采样**（与 LoRA 线逐字一致，否则缓存无法共享）：

```python
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(100))
first_split = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
```

同时把 L49 注释 `# train ~805k；validation ~100k；test ~100k` 改为 `# 采样后 train:80 validation:10 test:10（与 LoRA 线同口径，可横比）`。

**为什么必须加**：SFT 训练全部参数，80 万条 × 6G 卡完全不可行；且实测发现**缓存乒乓**问题（见 2.1②）。

**② L112 的 max_length 对齐**：`tokenized_user` 的 4096 改 1024：

```python
    tokenized_user = tokenizer(user_parts, truncation=True, max_length=1024)   # 原4096，与tokenized_full及LoRA线对齐
```

两个理由（第二个是真 bug）：

- **缓存共享**：`datasets` 按"输入数据指纹 + 函数指纹"校验缓存。实测当前 `cache/train.arrow` 是 **80 行**（LoRA 系 9 月 10 日最后写入的 100 采样变体）。若 SFT 保持全量数据 + 函数与 LoRA 不同，每次运行都会指纹不匹配 → 重算并把缓存翻写成 805k 行 → 下次 LoRA 运行又翻写回来——**每交替一次就全量重分词一遍**。加了采样并对齐函数后，SFT 加入共享缓存家族，**零重算直接复用 80 行缓存**；
- **labels 长度破坏 bug**：user 部分若超过 1024 token（长指令样本存在），`tokenized_user` 截到 4096 而 `tokenized_full` 截到 1024 → `user_len(如1100) > len(label)(1024)` → `label[:user_len] = [-100]*user_len` 在 Python 切片赋值语义下会把 1024 长的 label **撑长到 1100** → labels 与 input_ids 形状错位 → 前向崩溃。对齐后 user_len ≤ 1024，此路径封死。

**③ L69-81 的旧 tokenize_func 注释块**（docstring 包着的第一个版本）：可选删除，减少干扰。

### 2.2 加载分支（核心，L235-242 整块替换）

```python
# ===================== 加载 =====================
sft_save_path = "./training/qwen3-0.6b_SFT/full_model"   # 全量微调成果目录（完整权重，非adapter）
# 判断"保存完成"：model.safetensors 存在（SFT保存的是完整模型权重，没有adapter_config.json）
has_model = os.path.exists(os.path.join(sft_save_path, "model.safetensors"))

if has_model:
    # ---------- 模式一：加载已训练的全量权重，跳过训练 ----------
    print(f"检测到已保存的全量微调权重：{sft_save_path}，跳过训练")
    # SFT没有adapter概念：直接from_pretrained重载整个微调后模型（config+tokenizer已随权重保存）
    # 注意基线评估已在上方完成——全量微调覆盖了权重，无法像LoRA那样disable_adapter()切回base
    del trainer_baseline, model          # 释放旧引用：否则旧base权重(~1.2G)仍被trainer_baseline.model
                                         # 持有，重载后显存里同时存在两份0.6B模型
    model = AutoModelForCausalLM.from_pretrained(
        sft_save_path,
        torch_dtype="auto",
        local_files_only=True
    )   # 重载后在CPU；下方主Trainer构造时会自动迁移到cuda，无需手动.to()
else:
    # ---------- 模式二：首次运行，保持当前 base model 直接进入训练 ----------
    pass
```

四个要点解释（对应 LoRA 脚本里没有的细节）：

1. **`del trainer_baseline, model`**：与 LoRA 不同，LoRA 的 `PeftModel.from_pretrained(model, ...)` 复用已加载的 base（不新增大块显存）；SFT 重载是**全新一份权重**。旧 model 虽然被 `model = ...` 重新绑定，但 `trainer_baseline.model` 还攥着它——不 del 的话 6G 卡上两份 1.2GB + eval 激活很紧张。baseline 此时已用完（评估+生成都做完了），del 安全；
2. **不需要 device_map / .to("cuda")**：主 Trainer 构造时自动把模型迁到 GPU（transformers 的 `_move_model_to_device`），这正是当初删掉 `device_map="auto"` 的同一机制在起作用；
3. **tokenizer 不用重载**：`trainer.save_model()` 在传了 `processing_class=tokenizer` 时会一并保存 tokenizer 文件到目标目录，重载目录是自包含的；
4. **else 分支是 `pass`**：SFT 不需要像 LoRA 那样在 else 里做 `get_peft_model` 包装——训练的就是裸 model 本身。

### 2.3 训练参数适配（采样后必须同步，否则评估/保存全部失效）

**这是本次检查发现的最隐蔽问题**：当前 `eval_steps=200` / `save_steps=100`，但采样后全程只有 `80条 ÷ 4累积 × 3epoch = 60 步`——**两个阈值都不触发，训练全程一次 eval、一个 checkpoint 都没有**（这两组参数是当年全量 80 万条口径下设的，那时几十万步，200/100 很合理）。

L250-251、L263-264 改为与 LoRA 线一致的 epoch 策略：

```python
    # save_strategy="steps",
    # save_steps=100,
    save_strategy="epoch",
    save_total_limit=3,
    ...
    # eval_strategy="steps",
    # eval_steps=200,
    eval_strategy="epoch",
```

（原参数注释保留作历史参照即可。）另建议显式写出 `learning_rate=2e-5`（全量微调常用量级；默认 5e-5 也行，但显式写出来是仓库从 AdaLora 报告吸取的教训——adapter 系与全量系的 lr 直觉不同，写明口径避免误读）。

### 2.4 保存分支（L323-330 整块替换）

```python
# ============ 保存全量微调权重（关键代码） ============
if not has_model:
    # SFT保存完整模型权重(~1.2GB bf16)——与LoRA只存adapter(几十MB)完全不同
    trainer.save_model(sft_save_path)     # 同时保存config+tokenizer(因传了processing_class)
    print(f"全量微调权重已保存至：{sft_save_path}")
else:
    print(f"已训练权重来自：{sft_save_path}（无需重复保存）")
```

配套把 L280 的守卫 `if not has_adapter:` 改为 `if not has_model:`（两处：训练守卫 L280、保存守卫 L324）。

磁盘账：`full_model/` ≈ 1.2GB + `output/` 下 3 个 epoch checkpoint × 1.2GB ≈ 共 4.8GB——确认 WSL 磁盘余量；旧的全量口径 checkpoint（9 月 8 日产物）与新采样口径不可比，建议确认后清理。

---

## 三、其它问题检查清单

| # | 位置 | 问题 | 级别 |
|---|---|---|---|
| 1 | L240-242 | 未完成分支是**语法错误**，脚本当前无法运行 | 🔴 即 2.2 |
| 2 | L263-264 | 采样后 `eval_steps=200 > 全程60步`，评估与保存均不触发 | 🔴 即 2.3 |
| 3 | L112 | `tokenized_user` 4096 不对齐 → 缓存乒乓 + labels 撑长 bug | 🔴 即 2.1② |
| 4 | L34 | 缺采样，80 万条对 6G 卡不可行 | 🔴 即 2.1① |
| 5 | L236-238、L323-330 | `lora_save_path`/`has_adapter`/"LoRA适配器"文案均为 LoRA 语汇残留 | 🟡 即 2.2/2.4 |
| 6 | L207 | 注释"加载模式用 disable_adapter() 切回 base"——SFT 不适用（无 adapter 可禁用），建议改写为"基线须在重载前评完" | 🟡 |
| 7 | L69-81 | 旧版 tokenize_func 注释块 | ⚪ 可选清理 |
| 8 | L212 | `generate_answer` 注释提到 `lora_dropout` ——SFT 无此参数，无害但可顺手改 | ⚪ |

**已核对无需改的部分**：基线 Trainer 的 output_dir 已指向 `qwen3-0.6b_SFT` ✓；`custom_collate_fn` 三坑规避 ✓（与全仓库一致）；`report_to="tensorboard"` + `logging_steps=10` ✓；`trainer` 顶格定义（CLAUDE.md 坑 6）✓；`device_map` 已注释 ✓；fp16+gc 的 OOM 组合是刻意设置 ✓。

---

## 四、修改后自查清单

1. `python -m py_compile test_SFT_qwen3-0.6b.py` 通过（分支不再悬空）；
2. 首次运行：分词阶段**秒过**（复用 80 行缓存，无 "Map..." 长进度条——若出现长 Map 说明函数指纹仍不一致，回头查 2.1②）；
3. 训练 3 个 epoch，每 epoch 出现 1 次 eval + 1 个 checkpoint（TB 曲线有 3 个 eval 点）；
4. 结束后 `training/qwen3-0.6b_SFT/full_model/` 出现 `model.safetensors`(~1.2GB) + config + tokenizer 文件；
5. **二次运行**：打印"检测到已保存的全量微调权重…跳过训练"，直接进评估；`after_eval` 与上次训练结束的 eval_loss 一致；生成对比正常；
6. 想强制重训：删除（或改名）`full_model/` 目录即可。

---

## 五、方法论一句话

**"复用训练成果"不是一种模式，而是两种机制**：参数高效系（LoRA/AdaLora/prompt 系）复用的是「base + 增量」的组合，加载走 `PeftModel.from_pretrained`；全量系复用的是「最终权重」本身，加载就是普通的 `from_pretrained`。复制脚本跨这两个世界时，保存路径的判断文件（`adapter_config.json` vs `model.safetensors`）、加载 API、显存账（是否新增一份权重）三件事都要跟着换——这也是本仓库"论文名 ≠ 实现名"之外的第二类"名实分离"陷阱。

---

## 六、复查结果（第二轮：方案落实后 + 缓存机制勘误）

> 复查方式：① 通读修改后全文；② `py_compile` 语法验证；③ 用真实脚本源码段 + 真实缓存路径实测缓存复用行为（先备份，实测未污染缓存）。

### 6.1 落实确认（第二节方案的关键项全部到位）

| 方案项 | 位置 | 状态 |
|---|---|---|
| 采样 100 条（对齐 LoRA 线口径） | L33 | ✅ |
| `tokenized_user` max_length 4096→1024 | L99 | ✅ |
| 加载分支（`model.safetensors` 判断 + `del` 旧引用 + `from_pretrained` 重载 + else pass） | L222-241 | ✅ 结构与方案一致 |
| save/eval 策略改 epoch | L250-255 | ✅ `eval_strategy="epoch"` 已激活（第一轮曾漏改，本轮确认已换位） |
| 保存分支（全量权重语义） | L323-331 | ✅ |
| 语法 | 全文 | ✅ py_compile 通过 |
| 基线注释（重载前评完） | L194 | ✅ |

### 6.2 【勘误】缓存机制：本文 2.1② 的"缓存乒乓"说法不成立，真实机制更值得警惕

**源码 + 实测双重证据**：datasets 的 `_map_single` 缓存复用判断是（`arrow_dataset.py`）：

```python
if os.path.exists(shard_kwargs["cache_file_name"]) and load_from_cache_file:
    return Dataset.from_file(...)        # ← 只看文件是否存在，不校验指纹
```

实测：用 SFT 改后的真实数据管线（真实缓存路径）跑了一遍——计算出的指纹 `db5f479d…` 与缓存内存储的 `be20ddc4…` **并不相同**，但三个 arrow 的 mtime 纹丝未动（直接加载、零重算、零翻写）。

**修正后的正确认知**：

1. **map 的缓存复用从不校验指纹**——"存在即加载"。所以不存在本文 2.1② 设想的"交替翻写/全量重分词"；改后 SFT 下次运行将**秒级复用**现有 80 行缓存；
2. 真正的风险方向恰好相反（也更阴险）：**函数或数据变了、但缓存文件还在 → 静默加载旧数据**。这正是 CLAUDE.md 坑 1（"改 tokenize_func 必须手删 arrow"）的根源。设想未对齐的 pre-fix SFT（全量 80 万条口径）在缓存存在时运行：它会**拿着 80 条数据安静地训练**，日志毫无异常；
3. 因此 2.1 的对齐工作依然完全必要，只是理由修正：从"避免乒乓重算"改为"**保证共享缓存的内容语义正确**"——采样口径、max_length、mask 逻辑三样与 LoRA 线一致后，加载这 80 行缓存才是对的；
4. 附带发现：map 计算出的指纹依赖执行上下文（heredoc 执行 `6357…` / 脚本文件执行 `db5f…` / 缓存存储 `be20…` 三者互不相同）——**指纹不是跨脚本共享缓存的判据，内容一致才是**；好在复用根本不看它。

### 6.3 遗留修改清单（不阻塞运行，按需手动清理）

**🟡 命名正名（功能正确，仅语义误导——建议做）**：

| 位置 | 现状 | 建议 |
|---|---|---|
| L223 | `lora_save_path = ".../lora_adapter"` + 行尾旧注释"← 从文件末尾上移到此处定义"（LoRA 复制残留） | 改 `sft_save_path = "./training/qwen3-0.6b_SFT/full_model"`，删旧注释；同步 L225/L235/L326/L329/L331 |
| L225 | `has_adapter`（查的却是 model.safetensors） | 改 `has_model`；同步 L227/L280/L324 |
| L224 | 注释"adapter_config.json 存在"与代码检查的 `model.safetensors` 不符 | 注释改为"model.safetensors 存在（全量权重保存完成的标志）" |

**⚪ 可选清理**：L87-88 注释"固定长度4096 / padding=max_length"已不实（现为 1024 + 动态 padding）；L59"还没有做 labels mask"不实（函数做了）；L199 `lora_dropout` 注释是 LoRA 语汇；L4 `DataCollatorForLanguageModeling` import 仅剩注释块引用；L17 `#    model_name,` 残留。另可选：`training_args` 显式写 `learning_rate=2e-5`（全量微调常用量级；默认 5e-5 也能跑，写明口径便于跨线解读）。

### 6.4 运行前自查清单（最终版）

1. ✅ `py_compile` 通过（已验）；
2. 首次运行分词阶段秒过（6.2 已实测缓存可复用）；
3. 训练 3 个 epoch（60 步）：每 epoch 1 次 eval + 1 个 checkpoint，TB 里 3 个 eval 点、6 个 train 点（logging_steps=10）；
4. 结束后 `full_model/`（或现名 `lora_adapter/`）出现 `model.safetensors`（~1.2GB）+ config + tokenizer 文件；
5. **二次运行**：打印"检测到已保存的全量微调权重…跳过训练"，直接评估；`after_eval` 与上次训练结束的 eval_loss 一致，生成对比正常；
6. 强制重训：删除该权重目录即可。

---

## 七、实战复盘：训练第一步 `NotImplementedError ... for 'BFloat16'`（fp16 AMP × bf16 权重冲突）

> 触发：加载分支改造完成后首次训练，第 0/60 步即崩：
> `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`
> 基线评估正常（10/10 完成）、"开始 SFT 微调"打印后才崩——**只有训练路径会踩**。

### 7.1 报错链路与根因

```
trainer.train() → clip_grad_norm_ → scaler.unscale_(opt)                    ← fp16=True 启用的 GradScaler
  → torch._amp_foreach_non_finite_check_and_unscale_ → NotImplementedError for BFloat16
```

根因是**混精三件套失配**：

| 组件 | 脚本现状 | 说明 |
|---|---|---|
| 权重 dtype | **bf16**（L19 `torch_dtype="auto"` → Qwen3 config 是 bfloat16） | 梯度随权重也是 bf16 |
| AMP 开关 | **fp16=True**（L263） | 启用 fp16 autocast + **GradScaler**（loss 放大器） |
| GradScaler 对 bf16 梯度 | ❌ | fp16 才需要 loss scaling（动态范围窄）；**bf16 从不需要**，所以 torch 根本没为 BFloat16 实现这个 CUDA foreach unscale 核——撞上即 NotImplementedError |

**注释暴露的矛盾**：L263 注释写着"不要用 torch_dtype=auto 带来的 bf16"，但 L19 恰恰就是 `torch_dtype="auto"`——fp16 与 auto 同时生效是**两个时期编辑叠加**的自相矛盾组合，在这套环境（torch 2.8）下必然崩。基线评估没炸是因为 baseline 用的是 `bf16=True`（无 scaler），评估也不走梯度路径。

### 7.2 修复方案（一行）：`fp16=True` → `bf16=True`

```python
    bf16=True,                         # ★ 原fp16=True与torch_dtype="auto"(bf16权重)冲突：fp16的GradScaler
                                     #   对bf16梯度调CUDA foreach unscale核 → NotImplementedError。
                                     #   bf16 AMP无需GradScaler，scaler路径整个消失；与全仓库Qwen线口径一致
```

三个理由：**机制上** bf16 AMP 不需要 loss scaling，报错的整条链路消失；**实证上** 全仓库 Qwen 线（LoRA/AdaLora/Prefix/Prompt/PTuningV2）清一色 `bf16=True + torch_dtype="auto"` 全部跑通；**口径上** 与其他线对齐后 eval_loss 才可跨线比较。顺手删掉误导注释和 L264 的 `# bf16=False,` 残留。

### 7.3 替代方案（学习向，不推荐）

若坚持 fp16 AMP，必须让梯度不是 bf16，二选一：

- `torch_dtype=torch.float32` + `fp16=True`：fp32 权重 → fp32 梯度，scaler 可用。代价：权重显存 1.2→2.4GB，6G 卡上徒增压力；
- `torch_dtype=torch.float16` + `fp16=True`：fp16 权重 → fp16 梯度，scaler 可用。但纯 fp16 权重训练数值稳定性差（这正是 bf16 被发明的原因）。

两者都不如 `bf16=True` 干净；"6G 卡优先 fp16"的旧直觉在权重已是 bf16 的前提下不成立（bf16 AMP 的激活显存与 fp16 相当，RTX 4050 完整支持 bf16）。

### 7.4 日志里那行 use_cache 警告：良性，与 Prefix 线的假跑警告是两回事

```
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
```

这是 transformers 自己处理的良性提示（开 gc 时自动关 KV 缓存，一行即止）。**不要**与 Prefix 线的 `"Caching is incompatible..."` 刷屏假跑警告混淆——那个是每层 `GradientCheckpointingLayer` 丢弃 `past_key_value` 的真故障信号。SFT（非 KV 注入方法）开着 gc 完全正常。想消除这行提示，可在训练分支加 `model.config.use_cache = False`（仓库其他线的做法），不加也无妨。

### 7.5 修复后预期

训练 60 步正常推进，每 epoch 1 次 eval + 1 个 checkpoint，TB 曲线 3 个 eval 点；结束保存 ~1.2GB 全量权重；二次运行走加载分支。

### 7.6 方法论沉淀

**混精匹配律：权重 dtype 决定 AMP 开关**——bf16 权重配 `bf16=True`（无 GradScaler），fp16/fp32 权重才配 `fp16=True`（有 GradScaler）；`torch_dtype="auto"` + `fp16=True` 是"自相矛盾组合"的典型样本，报错点在第一步 `clip_grad_norm_` 的 unscale 环节， traceback 里出现 `grad_scaler.py` + `BFloat16` 字样即可秒判。

---

## 八、实战复盘：ROUGE-L 归零——SFT 反而成功了，是评分管线失效

> 实跑结果：eval_loss 4.8659 → 4.4278（**-9.0%**，训练健康）；ROUGE-L 0.0174 → **0.0000**。
> 先给结论：**SFT 是成功的**（加载保存的权重实测生成了流畅、切题的中文回答，无乱码无空输出）；ROUGE 归零是**评分管线的字符集缺陷**，且"归零"恰恰是成功的旁证。

### 8.1 证据链（三步定位，全部实测）

**① 加载保存的 SFT 权重实际生成**（只读诊断）——三条验证集指令的输出：

```
样本0（"念念不忘的文章"）：'我之前写过一篇《一个》的改版前的文章，是关于…'（连贯中文，有重复句式）
样本1（"钢炼的伏笔"）：'首先，我需要明确用户的问题。用户询问的是《钢之炼金术师》中…'（切题、有结构）
样本2（"胸肌怎么练"）：'我之前在健身群里看到有人问过这个问题…先做胸肌的训练…'（切题）
```

→ **无乱码、无空输出、无循环崩坏**——与基线（葡语式乱码碎片循环）天壤之别。eval_loss -9% 与生成质量互相印证。

**② vendored rouge 的分词器实测**：`DefaultTokenizer.tokenize` 对纯中文文本返回 **`[]`**——`rouge-score` 底层只认 `[a-z0-9]`，**汉字逐字加空格的 `zh()` hack 对中文完全无效**（空格分不出"英文词"来，汉字本身就不在字符集里）。本文 3 节和 LoRA 文档里"逐字加空格伪装英文词"的说法需要修正。

**③ 0.0174 的真面目**：基线乱码碎片含**拉丁字母**（如 "eração"），与参考里偶尔的数字/拉丁字符碰巧 LCS 重叠 → 0.0174 是**伪信号**；SFT 后输出变成纯中文 → 拉丁交集消失 → 0.0000。**讽刺的结论：ROUGE 归零恰恰证明 SFT 把"拉丁乱码"修成了"正常中文"**。

### 8.2 历史数字解读修正（重要）

按同一机制回看：此前 Prompt 线（0.0174→0.0586）、Prefix 线（→0.0258）的"ROUGE 提升"同样是**拉丁噪声的起伏**，不反映中文生成质量的真实变化——那些线的 eval_loss 变化仍是可靠指标（NLL 不经过分词），但 ROUGE 部分请以本节认知为准。

### 8.3 修复方案：给 rouge.compute 传字符级 tokenizer（已实测）

vendored `rouge.py` 的 `_compute` 原生支持 `tokenizer` 参数（L129-132 会包装成 rouge-score 的 Tokenizer），传一个**按字符切分**的函数即可绕开 `[a-z0-9]` 过滤。ROUGE 对比段（L313-319）改为：

```python
try:
    rouge = ev.load("./datasets/evaluate/metrics/rouge/rouge.py")
    zh = lambda t: t.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "").strip()
    # ★ 字符级 tokenizer：按字符切分，绕开 rouge-score 的 [a-z0-9] 英文分词（对中文是必须的）
    char_tok = lambda s: list(s)
    kw = dict(rouge_types=["rougeL"], tokenizer=char_tok)
    r_before = rouge.compute(predictions=[zh(g) for g in baseline_generations],
                             references=[zh(r) for r in test_references], **kw)
    r_after  = rouge.compute(predictions=[zh(g) for g in after_generations],
                             references=[zh(r) for r in test_references], **kw)
    print(f"ROUGE-L：微调前 {r_before['rougeL']:.4f} -> 微调后 {r_after['rougeL']:.4f}")
except Exception as e:
    print("ROUGE 计算跳过：", e)
```

注意三点：① `zh()` 里**不再加空格**（字符级 tokenizer 自己切）；② 字符级 LCS 会给中文常用字带来基线偏置（零重叠文本也能得 ~0.11），**只看相对变化与量级**；③ 实测区分度：高重叠中文→0.68、零重叠中文→0.11、拉丁乱码 vs 中文→0.00，符合直觉。此修复同样适用于 Qwen 其他五条线的 ROUGE 段。

### 8.4 SFT 生成本身的质量评估（非缺陷，是预期）

实测生成的两点小毛病：**重复句式**（样本 0 整句循环）与**内容偏泛**（没有真正"答到点上"）。这在 80 条数据 × 3 epochs × lr 默认的全量微调下是正常水平——eval_loss -9% 已说明模型有效吸收了数据分布。若想进一步改善（均非 bug）：加数据量、加 epochs + 早停、或采样解码（`do_sample=True, temperature=0.7, top_p=0.9`，注意前后对比的可复现性会被破坏，对比实验保持 greedy）。

### 8.5 方法论沉淀

1. **指标归零先分"生成坏了"还是"评分坏了"**——把生成内容原文打印出来（`repr()` 看转义字符）是第一步，永远不要只盯着数字推理；
2. **英文评测管线对中文的静默失效**：分词器返回空列表 → 所有样本得 0，**不报错、不警告**。任何英文原生的评测指标（rouge/bleu/bertscore）用于中文前都要先做一次"正样本能否得非零分"的烟囱测试；
3. 伪信号比零信号更危险：0.0174 这种"看起来正常的小数字"掩盖了指标根本无效的事实，还让 Prompt/Prefix 线误读了"提升"。**指标为 0 时问一句"它对好样本能打非零分吗"，指标非零时问一句"这个分数有语义吗"**。

---

## 九、实战复盘（续）：为什么 LoRA/AdaLora/Prefix 的 ROUGE"没问题"，SFT 才归零？

> 问题：如果是评分管线缺陷，为什么此前各线的 ROUGE 都有数字，唯独 SFT 归零？
> 方法：逐线加载已保存的 adapter 权重（base/LoRA/AdaLora/Prefix/Prompt/PTuningV2），同款 prompt、greedy 64 token 实测生成，对同一批文本分别用「默认分词」和「字符级分词」打 ROUGE-L。**QLoRA 线因 bnb 4bit 仅支持 CUDA、CPU 无法复现，未实测**。

### 9.1 直接答案：前提不成立——所有线都有同样的问题，"没问题"是错觉

实测大表（同一批 3 条验证集指令）：

| 线 | 生成特征（实测摘录） | 拉丁字母数 | 默认分词 ROUGE-L | 字符级 ROUGE-L |
|---|---|---|---|---|
| base | 以 **`<think>
嗯，用户问的是…`** 开头（思考模式续写） | 33 | **0.0000** | 0.1188 |
| LoRA | 与 base 几乎同款（仅 think 标签消失，样本1混入 "mere"） | 13 | **0.0000** | 0.1222 |
| AdaLora | 同 base（带 `<think>`，行为基本没变） | 27 | **0.0000** | 0.1161 |
| Prefix(proj=True) | **无 think、直接正文、有 `<\|im_end\|>` 收尾**（内容有编造） | 8 | **0.0000** | **0.1401** |
| Prompt | 样本0 直接**复述问题**（学坏），样本2 中文回答 | 8 | **0.0000** | 0.1127 |
| PTuningV2 | 全乱码（阿拉伯文/拉丁碎片循环） | 430 | **0.0000** | **0.0000** |
| SFT（全量，上节） | 流畅切题中文正文 | ~0 | 0.0000 | （未测，与 Prefix 同级预期） |

**"默认分词"口径下，六条线全是 0**——你当年看到的非零数字（0.0174/0.0258/0.0586/0.0052）来自脚本里「逐字加空格」的 `zh()` 口径，其成因是：`skip_special_tokens=False` 让生成保留 **`<think>`（decode 后是 5 个拉丁字母）**、`《One.一个》` 的 "One"、以及乱码碎片的拉丁词，这些 token 与参考里偶见的数字/拉丁（"10.27"、"03版"、"FA"）**碰巧 LCS 重叠**——纯噪声，与生成质量无任何正相关（PTuningV2 全乱码也有 0.0052 为证）。

**为什么唯独 SFT 归零**：SFT 是唯一一条"整个模型被推向纯中文输出"的线（全量微调改所有权重，60 步就跳出 think 模式写正文）——拉丁噪声源消失 → 伪信号归零。**不是 SFT 出了问题，而是 SFT 的成功让一直无效的指标露了馅**。prompt 系各线参数少（0.3M~10M）、200 步内改不动整体输出风格，生成保留基座的 think/One 等拉丁字符 → 噪声在 → "看起来正常"。

### 9.2 附带收获：六线生成的横向对比（字符级 ROUGE + 肉眼）

- **字符级 ROUGE 全线挤在 0.11~0.14**（PTuningV2 除外）——这是中文常用字重叠的基线偏置区，区分度有限，只适合看相对变化；
- 肉眼判断更有信息量：**base/LoRA/AdaLora 生成全部是 `<think>` 思考模式续写**（基座 chat 行为，没学到 zhihu 数据的"直接回答"风格）；**Prefix(proj=True)、Prompt、SFT 三条线跳出了 think 模式**直接写正文——Prefix/SFT 还学会了 `<|im_end|>` 收尾。这与 eval_loss 的结论（SFT -9.0% / Prompt -11.4% > Prefix -9.1% > LoRA -7.1% > AdaLora -4.4%）大体自洽：**学得多的线，行为改变可见**；
- PTuningV2 的乱码（阿拉伯文循环）与 +119% 互证，维持"封存"结论。

### 9.3 修复清单：7 个脚本统一改（含 QLoRA）

`grep -l "rouge.compute"` 确认 **7 个脚本**共用同一段缺陷代码：`test_Lora / test_AdaLora / test_QLoRA / test_Prefix / test_Prompt / test_PTuningV2（qwen） / test_SFT`。统一把 try 块内的 ROUGE 段替换为 8.3 节的字符级版本（各脚本只有 `zh`/生成变量名差异，模式一致）：

```python
    rouge = ev.load("./datasets/evaluate/metrics/rouge/rouge.py")
    zh = lambda t: t.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "").strip()
    char_tok = lambda s: list(s)                      # ★ 字符级分词，绕开 [a-z0-9] 英文分词
    kw = dict(rouge_types=["rougeL"], tokenizer=char_tok)
    r_before = rouge.compute(predictions=[zh(g) for g in baseline_generations],
                             references=[zh(r) for r in test_references], **kw)
    r_after  = rouge.compute(predictions=[zh(g) for g in after_generations],
                             references=[zh(r) for r in test_references], **kw)
    print(f"ROUGE-L：微调前 {r_before['rougeL']:.4f} -> 微调后 {r_after['rougeL']:.4f}")
```

改动要点（相对旧版）：① `zh()` 去掉逐字加空格（无用且误导）；② 传入 `tokenizer=char_tok`；③ 阅读口径：**字符级 0.11~0.14 是"没有实质重叠"的基线区**，超过 ~0.2 才算有字面参考价值，且永远优先看 eval_loss 与生成原文。

### 9.4 方法论沉淀

1. **"别的线没问题"要靠实测复核，不能靠印象**——本轮发现所谓"没问题"的各线在同一口径下同样是 0；跨脚本复制的同一段代码，缺陷是**成批**的，修一处必须 grep 全仓库；
2. **伪信号的存在性有条件**：同一个坏指标，在不同数据分布上会"时好时坏"（拉丁噪声在/不在），这比恒定为 0 更迷惑——它让人生出"SFT 之前都正常"的错误归因；
3. 生成质量肉眼审读不可省：本轮顺带发现 base/LoRA/AdaLora 全在 `<think>` 思考模式里打转、Prefix/Prompt/SFT 才学会直接作答——这类行为差异是任何单一指标都看不出来的。

---

## 十、实战复盘（终）：「我实测全 0」与「你运行非零」的矛盾消解（口径对齐复现）

> 用户提供的新证据：6 条线用原脚本跑出的 ROUGE-L 非零且多数"提升"（LoRA 0.0174→0.0478、QLoRA→0.0467、Prompt→0.0433、Prefix→0.0258、AdaLora→0.0154、PTuningV2→0.0036）；SFT 换字符级后 0.1028→0.0959。
> 这与我第九节"六条线默认分词全 0"表面矛盾。**消解方式：对齐口径复现**——结果证明两轮数据都对，差别全在口径。

### 10.1 两轮实测的口径差异表（矛盾的根源）

| 因素 | 第九节诊断（得全 0） | 你的脚本运行（得非零） |
|---|---|---|
| 样本数 | **3 条** | **5 条**（N_COMPARE=5） |
| 生成长度 | **max_new_tokens=64** | **256** |
| zh() 版本 | 无空格（只清洗标记） | **逐字加空格** |
| 精度/设备 | CPU fp32 | GPU bf16 |

**口径对齐复现**（N=5、256 token、逐字空格版 zh，CPU fp32）：

```
[base] 用户口径(逐字空格+默认分词) ROUGE-L=0.0441 | 字符级=0.1049 | 拉丁字母数=109
[LoRA] 用户口径(逐字空格+默认分词) ROUGE-L=0.0353 | 字符级=0.1059 | 拉丁字母数=28
```

base 的"默认分词"从 0.0000 变 **0.0441**——非零复现（与你的 0.0174 量级不同属正常：CPU fp32 与 GPU bf16 的贪心生成本就有内容差异，**这本身又是一个指标不稳定性的证据**）。拉丁字母数 109 vs 28 的悬殊直接展示了生成长度对噪声量的放大。

### 10.2 机制最终版（修正第九节的表述）

rouge-score 分词器只提取 `[a-z0-9]+`，**汉字无论"逐字空格"与否都不参与评分**——两种 `zh()` 版本对汉字等价（逐字空格并不能让汉字变成"英文词"）。因此：

- 所有非零数字 = **拉丁/数字 token 的碰巧重叠**。256 token 的长生成里 `<think>` 思考正文含大量英文单词（实测 base 一轮就 109 个拉丁字母），与参考里偶见的数字/英文重叠概率大增；64 token 短生成 + 3 样本时交集可以为空 → 全 0；
- 同一配置换个精度（fp32/bf16）数字就能从 0.0174 波动到 0.0441——**这个指标在中文任务上方差大到不可用**；
- 第九节"六条线全 0"的表述应修正为："小口径（3 样本/64 token）下全 0；全量口径（5 样本/256 token）下为拉丁噪声值"——两种口径都不是有效测量，只是失真方式不同（恒零 vs 随机噪声）。

### 10.3 各线"提升"的正确解读

| 线 | 你实测的"提升" | 字符级实测（微调后） | 解读 |
|---|---|---|---|
| LoRA | +0.030 | 0.1222 | "提升"= 微调后生成更长更流畅 → `<think>` 正文等英文 token 更多 → 噪声重叠更多，**与中文质量无正相关** |
| QLoRA | +0.029 | 未实测（同机制） | 同上 |
| Prompt | +0.026 | 0.1127 | 同上 |
| Prefix | +0.008 | **0.1401**（六线最高） | Prefix 学会直接正文+收尾，噪声反而少 → "提升"小 ≠ 效果差 |
| AdaLora | -0.002 | 0.1161 | 行为基本没变（与 -4.4% 一致） |
| PTuningV2 | -0.014 | 0.0000 | 乱码是阿拉伯字符，不在 `[a-z0-9]` → 分数最低——**乱码反而"得分最低"纯属字符集巧合** |
| SFT（字符级） | 0.1028→0.0959 | 0.0959 | 微调前后都处 ~0.10 的常用字基线区；微调后微降 = 输出"自己的话"而非模仿参考句式，正常 |

结论不变但更完整：**默认分词口径下所有非零与"提升"都是噪声；字符级口径下各线挤在 0.10~0.14 基线区**——这个指标（两种口径）都无法区分各线的生成质量，评估请以 eval_loss（NLL 不经分词）+ 生成原文审读为准。

### 10.4 最终修复清单（不变，补充解读规则）

7 个脚本的 ROUGE 段统一按 8.3 节改为字符级 `tokenizer`；解读规则补充：**字符级 ROUGE-L 在 0.10~0.14 = 常用字基线区（无实质重叠），>0.2 才有字面参考价值**；跨线/跨精度比较该指标无意义（方差 > 信号）。

### 10.5 方法论沉淀（本轮最重要的一课）

1. **对照实验必须先对齐口径**——样本数、生成长度、解码精度、预处理函数，任何一项不同都可能把"有效信号"变成"表面矛盾"。我自己上一轮的診断（3 样本/64 token/无空格）就栽在这上面，得出"六线全 0"的局部真相；用户拿着全量口径的数字质疑，才逼出完整机制。**两份"互相矛盾"的实测数据往往都对，拼起来才是全图**；
2. **复现是消解矛盾的终极手段**：把口径逐项对齐后数字立刻复现（0.0441 vs 你的 0.0174，同量级非零），比任何推理都硬；
3. 指标方差 > 信号时果断弃用：同一配置跨精度波动（0.0174→0.0441）比各线之间的差异还大，这样的指标只剩"报警器"价值（全 1 退化解检测靠预测类别数，不靠它）。

---

## 十一、收尾确认：LoRA/AdaLora 换字符级后"变化不大"——合理，且这正是健康的标志

> 实测：LoRA 0.1028 → 0.1081（+0.005）、AdaLora 0.1028 → 0.1060（+0.003）。
> 对照修复前（默认分词）：LoRA 0.0174 → 0.0478（"提升很大"）、AdaLora 0.0174 → 0.0154。
> 结论：**修复后的结果合理且健康；修复前的"大提升"是拉丁噪声的放大，修复后回归真实——指标在噪声区，模型的真实进步体现在 eval_loss 上。**

### 11.1 三个佐证

1. **三线微调前分数一字不差 = 0.1028**（SFT/LoRA/AdaLora）：基线是同一个 base、同样 5 条样本、同样 greedy、修复后同样的字符级 tokenizer → 确定性分数必须一致。对比默认分词时代 base 的"分数"在 0.0000~0.0441 间乱跳（随精度/长度/样本数漂移）——**这是修复后指标可复现性的直接实证**；
2. **0.1028~0.1081 全在第十节划定的"常用字基线区"（0.10~0.14）**：中文常用字的偶合重叠就能贡献 ~0.10，LoRA/AdaLora 在 80 条数据上学到的风格不足以让生成与"那 5 条特定参考"字面接近，+0.003~0.005 属噪声级波动；
3. **与第九节的独立实测对上**：当时用 64 token 口径实测 LoRA 微调后字符级 0.1222、AdaLora 0.1161——与本次 256 token 口径的 0.1081/0.1060 同区，互相印证。

### 11.2 为什么 LoRA"之前提升大、现在不大"（对照表）

| | 修复前（默认分词） | 修复后（字符级） |
|---|---|---|
| LoRA | +0.030（"提升很大"） | +0.005 |
| AdaLora | -0.002 | +0.003 |
| 分数的构成 | **拉丁 token**（`<think>` 的 5 个字母、One、乱码拉丁词）与参考数字/英文的碰巧 LCS——LoRA 微调后保留更多 think 风格英文结构 → 噪声分涨 | **全部汉字参与**的真实字面重叠——80 条数据学到的风格变化对"与特定参考的字面重合"贡献极小 |
| 与质量的关系 | 无（乱码线 PTuningV2 也能"得分"） | 弱（只反映字面模仿，不反映风格/事实学习） |

一句话：**修复前 LoRA 的"+0.030"和 AdaLora 的"-0.002"都是噪声的方向随机；修复后两线都收敛到基线区的小正值，才是真实信号。**

### 11.3 与 eval_loss 的分工（最终解读框架）

| 指标 | 度量什么 | 本仓库实测的角色 |
|---|---|---|
| eval_loss（NLL） | 逐 token 预测能力（学到数据分布的程度） | **效果主指标**：LoRA -7.1% > AdaLora -4.4%，与训练投入一致 |
| 字符级 ROUGE-L | 与特定参考的字面 LCS 重叠 | **辅助参考**：基线区 ~0.10-0.14；全 0/骤降可作退化解旁证；**不做跨线效果比较** |
| 预测类别数（分类线） | 是否退化为多数类 | 退化解硬报警器 |
| 生成原文审读 | 风格/行为改变（think 模式 → 直接正文 + 收尾） | 最有信息量，肉眼不可省 |

ROUGE 类指标天生度量"与参考的字面重合"，而 instruction 微调学的是**风格与分布**（模型输出"自己的话"），两者本就弱相关——80 条数据下尤其如此。想让 ROUGE 真正涨起来需要的不是更多训练，而是**让任务变成"逼近特定参考"的形态**（如摘要/改写），那不是本仓库 zhihu 问答任务的语义。

### 11.4 给后续脚本的建议

其余 5 个脚本（QLoRA/Prefix/Prompt/PTuningV2/SFT 已改或待改）换字符级后，预期同样出现"数字变小且变化不大"——**那不是退步，是显影真实**。判定标准固定为：微调前分数应严格等于 0.1028（同 base 同样本同解码，不等即有随机性泄漏，查 seed/greedy/样本选取）；微调后的解读交给 eval_loss 与生成审读。
