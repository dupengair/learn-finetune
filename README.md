# 大模型微调（PEFT）学习实验仓库

本仓库是大模型参数高效微调（PEFT）的学习与实验仓库，围绕两种典型架构、两种任务形态，覆盖从数据准备、微调训练到部署的完整流程。**当前规划的全部微调实验已完成**，踩坑与分析文档见 `docs/`（文末索引）。

## 两条实验线

| | **Qwen3-0.6B** | **BERT-base-uncased** |
|---|---|---|
| 架构 | Decoder-Only（`AutoModelForCausalLM`） | Encoder-Only（`AutoModelForSequenceClassification`） |
| 任务类型 | 生成（CAUSAL_LM） | 判别（SEQ_CLS，二分类） |
| 数据集 | zhihu-kol（INSTRUCTION/RESPONSE，约 100 万条，采样 100 条） | GLUE MRPC（句对语义等价判断，全量） |
| 实验方法 | 全量 SFT、LoRA、AdaLora、QLoRA（4bit NF4）、Prefix-Tuning、Prompt Tuning、P-Tuning v2（机制对照）、推理量化、vLLM 部署 | 全量微调、LoRA、AdaLora、P-Tuning v2 |

## 实验成果速览

> 数字均为本地实测（Qwen 线：zhihu-kol 采样 train 80 / val 10 / test 10；BERT 线：MRPC 全量）。**基线**：Qwen `eval_loss = 4.8659`（纯 base）、BERT `macro_f1 = 0.406`（随机分类头全预测多数类）。数值统一以磁盘现存 `checkpoint-*/trainer_state.json` 为准；详细分析与踩坑过程见 docs 对应文档与 [复盘总集](docs/大模型微调学习方案.md)。

| 实验线 | 方法 | 状态 / 关键结论 | 详见 |
|---|---|---|---|
| Qwen | SFT（全量） | eval_loss 4.8659 → **4.4278**（**-9.0%**）；产物为完整权重 1.19GB | — |
| Qwen | LoRA | eval_loss 4.8659 → **4.5222**（**-7.1%**，3 epochs） | [LoRA 分析](docs/LoRA微调requires_grad错误分析与修复.md) |
| Qwen | AdaLora | 调参后表观 4.8659 → 4.6509（-4.4%），**剥离关不掉的正交正则后 CE 4.6476（-4.5%）**；核心教训：**adapter 训练必须显式设 lr**，秩预算需回调驱动 `update_and_allocate` | [AdaLora 审查报告](docs/AdaLora微调代码审查报告.md) |
| Qwen | QLoRA（4bit NF4） | 已跑通，4.8659 → 4.5212（**-7.1%**，与 LoRA 线相对幅度持平）；权重显存 1.12→0.82 GiB；**QLoRA 线 eval_loss 与 LoRA 线不可横比**（量化底座不同） | [QLoRA 改造方案](docs/QLoRA微调改造方案.md) |
| Qwen | Prompt Tuning | eval_loss 4.8659 → **4.3326**（**-11.0%**，Qwen 线最佳）；仅 16,384 可训练参数（0.0027%）；**lr=5e-3**（embedding 系需大步长） | [PromptTuning 检查报告](docs/PromptTuning实现检查报告.md) |
| Qwen | Prefix Tuning（重参数化） | eval_loss 4.8659 → 4.4212（**-9.1%**），无乱码；`prefix_projection=True` 是小数据下的**稳定性前提**（同配置关掉则 +119% 崩坏，见下行） | [PrefixTuning 报错分析](docs/PrefixTuning实现检查与报错分析.md) |
| Qwen | P-Tuning v2（论文形态） | +119.4%/乱码——**封存为机制对照**，非 bug：v2 = peft 的 `PrefixTuningConfig(projection=False)`，去重参数化在小模型×小数据下不成立（论文前提是 300M~10B + 大数据） | [PTuningV2 检查报告](docs/PTuningV2实现检查报告.md) |
| Qwen | 4bit NF4 推理 / vLLM 部署 | 已验证 / 脚本就绪（OpenAI 兼容服务） | — |
| BERT | SFT / LoRA | macro_f1 0.406 → **0.833** / **0.8225**（2751 步，几分钟跑完） | [LoRA 分析](docs/LoRA微调requires_grad错误分析与修复.md) |
| BERT | AdaLora | 2751 步训练完成，macro_f1 0.406 → **0.7995**（ep2 峰值 0.8009）；核心教训：`load_best_model_at_end` 与 AdaLora 的 rank_pattern 演化**不兼容**（size mismatch） | [AdaLora BERT 改造方案](docs/AdaLora微调BERT改造方案.md) |
| BERT | P-Tuning v2 | macro_f1 0.406 → **0.6731**（lr=1e-3、5 epoch，保 v2 纯度）；含全 1 退化解的定位与拯救，七组对照实验 E1-E7；关键结论：**prompt 系参数 lr 需 1e-3 量级起步**（1e-4 会卡在"全预测多数类"退化解），且**「重参数化 + 足够步长」两个必要条件缺一不可**；BERT+SEQ_CLS+PREFIX_TUNING 组合在本环境实测可用（修正了旧认知） | [PTuningV2-BERT 检查报告](docs/PTuningV2_BERT实现检查报告.md) |

## 学习成果与经验总结

> 以下是全部实验沉淀的可迁移结论，按主题归组；每条都有实测证据，细节见对应文档。

### 1. Prompt 系方法谱系与 peft 实现的映射（论文名 ≠ 实现名）

| 论文方法 | 机制 | peft 配置类 |
|---|---|---|
| P-Tuning v1 | 输入层伪 token + LSTM/MLP 重编码 | `PromptEncoderConfig` |
| Prompt Tuning (Lester) | 输入层纯 embedding，无编码器 | `PromptTuningConfig` |
| Prefix Tuning (Li & Liang) | 每层注入 KV 前缀 + **MLP 重参数化** | `PrefixTuningConfig(prefix_projection=True)` |
| **P-Tuning v2** | 每层注入 KV 前缀，**去重参数化** | **`PrefixTuningConfig(prefix_projection=False)`（默认）** |

- v2 论文自述"是 Deep Prompt Tuning（即 Prefix Tuning 机制）的 NLU 适配版"——peft 只实现了一套每层注入代码，`prefix_projection` 开关区分两篇论文的形态；
- 实证：同一脚本仅翻转该开关，Qwen 线从 +119%/乱码 变为 -9.1%/正常——**"是哪个方法"最终由这个超参决定**。

### 2. PEFT 方法选型与调参判据

- **起点中性是第一判据**：LoRA 的 B=0 初始化让"包装后未训练"严格等价 base（黄金标准）；Prefix 系存在机制性起点罚金（KV 注入门槛 +4~9 loss，与注入值无关，全零同样触发）。**新方法第一步先评"包装后未训练的验证集 loss"**；
- **lr 按"冻结彻底程度"重校**：adapter 系（LoRA/AdaLora）1e-4~2e-4 够用；embedding/prompt 系需 1e-3~5e-3（本仓库实测：Qwen PromptTuning 5e-3、BERT PTuningV2 1e-3）；
- **重参数化是小数据/小模型下 Prefix 系的稳定性前提**：Qwen 线 False→+119%、True→-9.1%；BERT 线单开 True 不够，还必须配大 lr——「重参数化 + 足够步长」缺一不可；
- **跨方法对比先对齐起点**：起点不同的两条线（LoRA 4.87 vs Prefix 9.8+），终点数字不可比。

### 3. 训练设施与方法的兼容性（最阴险的一类 bug：静默假跑）

- **gradient_checkpointing 与 KV 注入路径结构性冲突**：transformers 的 `GradientCheckpointingLayer` 会把每层 `past_key_value` 置 None → Prefix 系的可训练参数脱离计算图。Qwen 侧表现为 loss 恒定的"假跑"（仅刷警告），BERT 侧连警告都没有、更静默（实测 3 个可训练参数只剩 2 个有梯度）。**用 prompt 系方法必关 gc**；
- **"良性警告"必须查明含义再忽略**：判定标准是警告是否修改了你方法的**核心机制对象**（如 `past_key_value=None`）；
- 换 PEFT 方法先问"**可训练参数走哪条前向路径**"：LoRA 走权重旁路、Prefix 走 KV 注入、Prompt Tuning 走输入拼接——路径不同，兼容的训练设施（gc / use_cache / collator）就不同。

### 4. 评估与排障方法论

- **全 1 退化解检测器**：分类任务 `len(set(preds))==1` = "没学动"的硬信号，比 loss 曲线可靠（loss 微降可能只是偏置滑动）；BERT v2 线的定位即由此开始；
- **归因靠对照实验，不靠猜测**：本仓库多次用"绕过 peft 原生注入""剂量曲线（prefix 长度 1→256）""linear probe 下限""同口径多配置对照（E1-E7）"把责任定位到方法/环境/数据层；
- **跨方法复制脚本时，加载分支比训练分支更易踩雷**：`is_trainable` 三分规则——prompt learning **禁止传 True**（peft 直接 raise）；LoRA 系传了无意义（可删）；**AdaLora 必须传 True**（否则评估前向访问 `trainable_adapter_name` 崩溃）。同类坑：`trainer` 定义在 else 分支内导致二次运行 NameError；
- 评估细节：`DataCollatorForLanguageModeling` 对 instruction 微调有**三个坑**（labels 不 pad → eval 批量>1 崩；无条件覆盖自定义 labels → `-100` mask 静默失效；`pad_token==eos` 时抹掉停止符监督 → 生成停不下来），**换自定义 collator 一次解决三个**；且 **collator 改动前后的 eval_loss 不可比**，对比要用同一套代码重跑；
- **ROUGE 在中文上曾长期给出伪信号**：vendored `rouge-score` 的分词器只认 `[a-z0-9]`，对纯中文返回空列表（全 0、不报错）；而"逐字加空格伪装英文词"的 hack **对汉字同样无效**（汉字不在字符集里）。此前各线的"ROUGE 提升"实为 `<think>`、乱码拉丁词与参考里偶见数字的**碰巧重合**，与生成质量零相关。正确做法是给 `rouge.compute` 传字符级 tokenizer；**主指标请用 eval_loss（NLL 不经过分词）**。

### 5. 环境与工具

- TensorBoard 全线集成（`report_to="tensorboard"` + `logging_steps=10`；注意默认 500 步大于小实验全程会一条不记）；历史 checkpoint 可从 `trainer_state.json` 的 `log_history` 补转 event 文件（见操作指导文档）；
- BERT 兼容性认知修正：transformers 4.55 重构版 BERT 的 `BertSelfAttention` 已实现 KV 路径，**prompt learning 系列（含 PREFIX_TUNING）在 BERT/SEQ_CLS 上实测可用**——旧资料"BERT 不支持"的说法对当前版本不成立；
- 离线环境要点：模型/数据 `local_files_only=True`；分词缓存 `.arrow` 不随 `tokenize_func` 改动自动失效，改分词逻辑必须手删缓存。

## 运行环境

- **硬件**：RTX 4050 Laptop（6GB 显存）+ WSL2
- **Conda 环境**：`ai-gpu`
- **关键版本**：torch 2.8.0 / transformers 4.55.0 / peft 0.18.1 / datasets 4.5.0 / evaluate 0.4.6 / vllm 0.10.2 / bitsandbytes 0.50.2 / tensorboard 2.20.0

> 脚本中的 `batch_size=1`、`gradient_accumulation_steps=4`、`gradient_checkpointing` 等参数是为 6GB 显存刻意设置的 OOM 规避方案（prompt 系方法除外——必须关 gc，见上文）；所有模型/数据集加载均使用 `local_files_only=True`（离线环境）。

## 目录与文件结构

```
.
├── test_*.py                    # 自包含实验/训练脚本（顶层顺序执行，非 pytest 测试）
├── CLAUDE.md                    # AI 助手项目配置（含架构约定与踩坑记录，亦可作项目说明阅读）
├── docs/                        # 总复盘 + 各实验线的踩坑/方案文档（见文末索引）
├── model/                       # ❌ 不进 git：本地 HF 模型权重（约 5G）
├── datasets/
│   ├── evaluate/                # ✅ 进 git：vendored 的 HF evaluate 库源码（脚本依赖其本地路径加载指标）
│   ├── zhihu-kol/               # ❌ 不进 git：SFT 原始数据 1.4G + 分词缓存（仅提交 README 数据卡片）
│   ├── glue/                    # ❌ 不进 git：GLUE 各子集 + 分词缓存（仅提交 README）
│   └── .cache/                  # ❌ 不进 git：下载缓存
└── training/                    # ❌ 不进 git：训练产物（脚本自动生成）
    ├── qwen3-0.6b_SFT / _Lora / _AdaLora / _QLora / _Prefix / _Prompt / _PTuning-V2
    └── bert-base-uncased_SFT / _Lora / _AdaLora / _PTuning-V2
    （每个目录下：output/=checkpoint、logs/=TensorBoard、lora_adapter/=适配器成果）
```

### 脚本清单

`test_` 前缀**不是 pytest 测试**，是自包含的实验脚本（无 main 函数，顶层顺序执行）。运行单个实验即 `python` 执行对应脚本：

| 脚本 | 内容 |
|---|---|
| `test_qwen3-0.6b.py` / `test_Bert.py` | 模型加载与结构探测（chat template、generate） |
| `test_transformers_qwen3-0.6b.py` / `test_transformers_bert.py` | 推理基础；Qwen 侧含 4bit NF4 量化推理 |
| `test_dataset_qwen3-0.6b.py` | zhihu-kol 数据加载/切分/分词/labels mask 流程 |
| `test_SFT_qwen3-0.6b.py` / `test_SFT_bert.py` | 全量微调（Trainer） |
| `test_Lora_qwen3-0.6b.py` / `test_Lora_bert.py` | LoRA 微调（peft）；含"复用已训练成果"双模式（has_adapter 判断）与包装前基线评估 |
| `test_Lora-load_bert.py` | 从 `training/.../lora_adapter` 加载适配器推理 |
| `test_AdaLora_qwen3-0.6b.py` / `test_AdaLora_bert.py` | AdaLora 微调：秩预算回调（`update_and_allocate`）、三阶段调度、`is_trainable=True` 加载（AdaLora 专属必需） |
| `test_QLora_qwen3-0.6b.py` | QLoRA 微调：BitsAndBytes 4bit NF4 加载 + `prepare_model_for_kbit_training` + paged 优化器 |
| `test_Prefix_qwen3-0.6b.py` | Prefix Tuning（原版论文形态：`prefix_projection=True` + 每层 KV 注入；v2 形态的对照见 PTuningV2 脚本） |
| `test_Prompt_qwen3-0.6b.py` | Prompt Tuning（TEXT 初始化虚拟 embedding，lr=5e-3） |
| `test_PTuningV2_qwen3-0.6b.py` | P-Tuning v2 论文形态（`prefix_projection=False`，机制对照线） |
| `test_PTuningV2_bert.py` | BERT×MRPC 分类上的 P-Tuning v2（含全 1 退化解拯救方案） |
| `test_vllm.py` | vLLM 0.10.2 起本地 OpenAI 兼容服务 |

### 运行方式

所有脚本必须**从仓库根目录运行**（脚本内均为相对路径）：

```bash
conda activate ai-gpu
python test_Lora_qwen3-0.6b.py
```

> 各微调脚本均有双模式行为：检测到 `training/<线>/lora_adapter/adapter_config.json` 则加载已训练权重跳过训练；删除该目录（或改名）即强制重训。
> 训练中/后查看 loss 曲线：`tensorboard --logdir=./training/<实验线>/logs --port=6006`，浏览器开 `http://localhost:6006`（详见 [TensorBoard 操作指导](docs/TensorBoard查看loss曲线操作指导.md)）。

## 操作说明

### 环境

- 工具（HuggingFace 镜像下载）：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

- 下载模型：

```bash
hf download Qwen/Qwen3-0.6B --local-dir ./model/Qwen3-0.6B
hf download google-bert/bert-base-uncased --local-dir ./model/bert-base-uncased
```

- 下载数据集：

```bash
hf download --repo-type dataset cong11ccc/Zhihu-KOL  --local-dir ./datasets/zhihu-kol
hf download --repo-type dataset nyu-mll/glue  --local-dir ./datasets/glue
```

- 下载 Evaluate（vendored 到仓库内，离线加载指标）：

```bash
git clone https://github.com/huggingface/evaluate.git ./datasets/evaluate
# 使用时要 load .py：
# metric = evaluate.load("./datasets/evaluate/metrics/glue/glue.py", config_name="mrpc")
```

> 训练产物（`training/**/output/`、`logs/`、`lora_adapter/`）由本地训练脚本自动生成，无需下载。

### 库

| 库 | 版本 | 用途 |
|---|---|---|
| torch | 2.8.0 | 训练框架（CUDA / bf16） |
| transformers | 4.55.0 | 模型加载、Trainer、量化推理 |
| peft | 0.18.1 | LoRA / AdaLora / QLoRA / Prefix / Prompt Tuning / P-Tuning 全系 |
| datasets | 4.5.0 | 数据加载与分词缓存（arrow） |
| evaluate | 0.4.6 | 评估指标（配合 vendored 源码本地加载） |
| bitsandbytes | 0.50.2 | 4bit NF4 量化（QLoRA） |
| vllm | 0.10.2 | OpenAI 兼容推理服务 |
| tensorboard | 2.20.0 | loss 曲线可视化（`torch-tb-profiler` 为可选性能分析插件，看曲线不需要） |

## Git 使用说明

### 不提交远程的目录（见 `.gitignore`）

| 目录 | 原因 |
|---|---|
| `model/` | 模型权重约 5G，可从官方渠道重新下载 |
| `datasets/zhihu-kol/`、`datasets/glue/`（README 除外） | 原始数据 + 分词缓存，可再生 |
| `training/*` | checkpoint、日志体积大且可再生；适配器成果仅保留本地 |
| `.codegraph/`、`__pycache__/`、`tmp_trainer/` | 本地索引、编译缓存、临时产物 |

### 日常提交流程

```bash
git add -A                      # 暂存所有变更（新增/修改/删除），幂等可重复执行
git commit -m "feat: 描述本次改动"
git push                        # 首次推送用 git push -u origin main
```

> `git add` 是幂等覆盖：暂存后若又修改了文件，再次 `git add -A` 即可让暂存区拿到最新版本。
> 只想撤销暂存（不动磁盘文件）：`git restore --staged <文件>`；只从暂存区移除已跟踪文件：`git rm -r --cached <路径>`。

### 首次关联远程仓库（已完成，留档备查）

```bash
git init
git remote add origin https://github.com/dupengair/learn-finetune.git
git push -u origin main         # -u 建立跟踪关系，之后可省略 origin main
```

## 更多文档

**规划与总复盘（建议先读这篇）**
- 实验总体规划 + 全部实验复盘 + 35 条踩坑总集 + 排障工具箱：[docs/大模型微调学习方案.md](docs/大模型微调学习方案.md)
  —— 含「预判 vs 实测」逐条对照：原方案里"peft 不支持 CausalLM 做 prompt 系""BERT 不能用 peft 的 PrefixTuningConfig""P-Tuning v2 需要第三方实现"等旧认知**均被实测证伪**。

**Qwen 线（生成）**
- LoRA 线踩坑全集（requires_grad / collator / 评估方法 / 复用模式基线陷阱）：[docs/LoRA微调requires_grad错误分析与修复.md](docs/LoRA微调requires_grad错误分析与修复.md)
- AdaLora(Qwen) 代码审查与三轮实战复盘（调度回调 / 加载崩溃 / eval_loss 口径 / 效果评估）：[docs/AdaLora微调代码审查报告.md](docs/AdaLora微调代码审查报告.md)
- QLoRA 改造方案（bnb 配置检查 / 显存实测 / 量化损失）：[docs/QLoRA微调改造方案.md](docs/QLoRA微调改造方案.md)
- Prefix Tuning 检查与报错分析（gc 假跑 / KV 注入门槛 / `prefix_projection` 由败转胜复盘）：[docs/PrefixTuning实现检查与报错分析.md](docs/PrefixTuning实现检查与报错分析.md)
- Prompt Tuning 检查报告（peft 自动对齐机制 / lr / Position ids 告警分析）：[docs/PromptTuning实现检查报告.md](docs/PromptTuning实现检查报告.md)
- P-Tuning v2(Qwen) 检查报告（v2=PrefixTuningConfig 的实证 / 与 Prefix 线的等价性）：[docs/PTuningV2实现检查报告.md](docs/PTuningV2实现检查报告.md)

**BERT 线（判别）**
- AdaLora(BERT) 改造方案与 size mismatch 复盘：[docs/AdaLora微调BERT改造方案.md](docs/AdaLora微调BERT改造方案.md)
- P-Tuning v2(BERT) 检查报告（BERT 可用性验证 / gc 静默假跑 / 全 1 退化解七组对照拯救 / `is_trainable` 三分规则 / trainer 作用域修复）：[docs/PTuningV2_BERT实现检查报告.md](docs/PTuningV2_BERT实现检查报告.md)

**工具**
- TensorBoard 查看 loss 曲线操作指导（历史 checkpoint 补转 event / WSL2 访问 / 曲线判读）：[docs/TensorBoard查看loss曲线操作指导.md](docs/TensorBoard查看loss曲线操作指导.md)
