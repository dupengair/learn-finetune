# Prefix Tuning 实现检查与报错分析（test_Prefix_qwen3-0.6b.py）

> 环境：peft 0.18.1 / transformers 4.55.0，关键结论均有源码级证据（transformers `modeling_layers.py`、peft `peft_model.py`）。
> 本文只分析不改源码，修改请对照文中代码手动进行。

## 一、结论速览

| # | 问题 | 位置 | 级别 |
|---|---|---|---|
| 1 | `gradient_checkpointing=True` 与 Prefix Tuning **结构性不兼容**：prefix KV 每层被置 None → 可训练参数脱离计算图 → **训练假跑**（loss 恒定、梯度为 None、200 步白跑） | L264 | 🔴 根因（"Caching is incompatible" 刷屏的真实含义，见第二节） |
| 2 | `generate_answer(model_lora, ...)`：变量名 `model_lora` **不存在**（本脚本叫 `model_trained`）→ 训练跑完后必抛 NameError | L301 | 🔴 致命 |
| 3 | `enable_input_require_grads()` 冗余（修掉 #1 后完全无意义） | L242 | 🟡 清理 |
| 4 | `bf16=False`：能跑但与其他实验线口径不一、更慢 | L266 | 🟡 建议 |
| 5 | 文案残留 3 处（"LoRA 适配器"字样） | L229、L293、L333 | ⚪ 文案 |
| 6 | `prefix_projection` 默认关闭（纯 Embedding，无重参数化 MLP）——与论文做法不同，可选增强 | L236-240 | ⚪ 可选 |

**重要认知先行**：那 25+ 条 "Caching is incompatible..." **不是无害警告**。它意味着每层的 `past_key_value` 被强制置 `None`——对 LoRA 无所谓（LoRA 不用 KV 缓存），但 Prefix Tuning 的全部可训练参数就活在 `past_key_values` 里。**你现在的训练是"假跑"**。

---

## 二、根因解析：Prefix Tuning × 梯度检查点为什么结构性冲突

### 2.1 Prefix Tuning 的工作机制（与 LoRA 的本质区别）

LoRA/AdaLora 改的是**权重**（给目标层挂低秩旁路），前向走普通 attention，不涉及 KV 缓存。
Prefix Tuning **不碰任何权重**：它训练一段虚拟 token 的 K/V 向量，每层 attention 计算时把 prefix KV 拼接在真实 token 的 KV 前面。载体就是 `past_key_values` 参数——peft 源码（`peft_model.py` `get_prompt`）确认：训练模式下 prefix 经 `prompt_encoder` 算出、**带梯度**、cast 到主干 dtype 后以 `past_key_values` 传入 `model.forward`。**这条 KV 注入链就是 Prefix Tuning 的全部可训练路径**（你打印的 `trainable params: 917,504` = `num_virtual_tokens 16 × 28 层 × 2(K/V) × 8 kv头 × 128 head_dim`）。

### 2.2 transformers 4.55 的源码证据：置 None 与 use_cache 无关

警告来自 `transformers/modeling_layers.py` L54-74（`GradientCheckpointingLayer.__call__`，Qwen3 的每个 DecoderLayer 都继承它）：

```python
def __call__(self, *args, **kwargs):
    if self.gradient_checkpointing and self.training:      # ← 触发条件就这两个
        ...
        if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
            kwargs["past_key_value"] = None                # ← Prefix 的 KV 在这里被丢弃
            do_warn = True
        ...
```

两个关键点：
1. 触发条件只有 `gradient_checkpointing and self.training`——**`use_cache=False` 救不了它**（第一行警告 "`use_cache=True` is incompatible... Setting `use_cache=False`" 只是连带处理，关掉 use_cache 后 `past_key_value` 照样被置 None）。你日志里 28 条 layer 警告 = 28 个 DecoderLayer 各丢弃一次；
2. `logger.warning_once` 也没拦住刷屏（消息随层构造），每个 step 重复 28 条 × 200 步。

### 2.3 后果链：为什么说这是"假跑"

```
gc=True → 每层 past_key_value=None → prefix KV 不参与任何 attention 计算
        → prompt_encoder 的输出不在计算图上 → backward 时 prefix embedding 的 grad = None
        → optimizer.step() 对这些参数是空转 → loss 恒定在 base 水平（≈4.87）
        → 200 步训练完成，但什么都没学（还不会报任何错！）
```

这是这类 bug 最阴险的地方：**不崩、只刷警告、loss 看起来"正常"（不变）**。唯一的破案线索就是你看到的警告本身。

### 2.4 为什么 LoRA / AdaLora 没这个问题

它们的前向不传 `past_key_value`（kwargs 里没有这个键，置 None 分支不触发），gc 只影响激活重算。**冲突是 Prefix Tuning 特有的**——但注意 LoRA 脚本当前 `gradient_checkpointing=False`，即使开了也不会有这个问题。

### 2.5 认知纠偏：PREFIX_TUNING 其实**支持** CAUSAL_LM

CLAUDE.md 已知坑 2 写的"`PrefixTuningConfig` 仅支持 Seq2Seq（T5/BART），不能直接用于 Qwen（CausalLM）"**对 Qwen 侧不成立**——本次运行已经证明：`get_peft_model` 包装成功、`trainable params: 917,504` 打印正常、训练启动。peft 0.18.1 的 `PeftModelForCausalLM` 明确支持 PREFIX_TUNING（`get_prompt` 就定义在它的 forward 路径里）。该认知中**仍然正确**的部分是：BERT（SEQ_CLS）不支持 prompt learning 系列——`PeftModelForSequenceClassification` 没有 KV 注入机制。建议顺手更正 CLAUDE.md 的坑 2 表述（本文档不改它）。

---

## 三、其余问题清单

### 3.1 🔴 L301：`NameError: name 'model_lora' is not defined`（必崩）

```python
after_generations = [generate_answer(model_lora, ins) for ins in test_instructions]   # L301
```

本脚本变量名是 `model_trained`（L230/L241）。这是从旧脚本复制时没改全——训练即使假跑成功，到这一行也会崩，评估/ROUGE/保存全部不会执行。改成 `model_trained`。

### 3.2 🟡 L242：`enable_input_require_grads()` 删除

它存在的唯一意义是配合梯度检查点（让冻结 embedding 的输出可求导）。修掉根因后 gc 已关，这行彻底无意义——prompt tuning 的可训练参数是 prefix embedding（本来就 `requires_grad=True`），不经这条路径。

### 3.3 🟡 L266：`bf16=False` 建议改 `bf16=True`

能跑（peft 的 `get_prompt` 会把 prefix KV cast 成主干 dtype，不会 dtype 报错），但：① 与 LoRA/AdaLora 线的 `bf16=True` 口径不一致，跨线对比时多一个变量；② 无 autocast 更慢。`optim` 里 `adamw_torch_fused` 也建议保留（你没动，✓）。`remove_unused_columns=False` 无害，可不动。

### 3.4 ⚪ 文案残留 3 处

- L229 `"检测到已保存的LoRA适配器"` → Prefix；
- L293 `"==== LoRA 微调后 ===="` → Prefix；
- L333 `"LoRA适配器已保存至"` → Prefix。

### 3.5 ⚪ 可选增强：`prefix_projection=True`（论文的 MLP 重参数化）

当前配置是**直接训练 Embedding**（`prefix_projection` 默认关闭），917,504 参数。Prefix Tuning 论文实际用的是 **MLP 重参数化**（小 Embedding → MLP → 实际 prefix），训练更稳定、效果通常更好：

```python
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=16,
    prefix_projection=True,        # ← 可选：启用重参数化 MLP
    # prefix_projection_hidden_size 默认 512
)
```

代价：可训练参数从 0.9M 涨到约 30M（MLP 输出层 512×57344 ≈ 29M），显存占用依然很小（6G 卡完全装得下）。建议：先按最小改动跑通基线版，再开这个做对比实验。

### 3.6 你做对的部分（确认无需改）

- 数据管线 / labels mask / collator / 双模式结构 / 包装前基线：全部正确沿用 ✓；
- `num_train_epochs=10` + `learning_rate=1e-4` + `warmup_ratio=0.1`：正确吸收了 AdaLora 第十节的教训 ✓；
- 目录已改为 `qwen3-0.6b_Prefix` ✓；
- 保存/加载链路：`trainer.save_model` 对 prompt tuning 存 prompt encoder 权重（~3.7MB），`PeftModel.from_pretrained` 加载分支兼容 ✓。

---

## 四、修复方案

### 4.1 核心修复（一行）：关闭梯度检查点

```python
    gradient_checkpointing=False,        # ★ Prefix Tuning 必须关！gc 会把 past_key_value 置 None，
                                         #   prefix 参数被移出计算图 → 训练假跑（见文档第二节）
```

**显存可行性核算**（为什么敢关）：Prefix Tuning 冻结全部主干，可训练参数只有 0.9M（激活显存与 LoRA 线完全相同——同数据、同 batch、同 max_length，LoRA 线 `gc=False` 已实测跑通）；Prefix 额外引入的 KV 仅 `16 token × 28 层 × 2 × 1024 × 2字节(bf16) ≈ 3.7MB`。结论：6G 卡毫无压力，gc 本来就是多余的。

配套清理：

```python
    model_trained = get_peft_model(model, peft_config)
    # model_trained.enable_input_require_grads()   # ← 删除（3.2）
```

### 4.2 NameError 修复（L301）

```python
after_generations = [generate_answer(model_trained, ins) for ins in test_instructions]
```

### 4.3 可选优化：训练时关 KV 缓存省显存

gc 关掉后，训练 forward 若 `use_cache=True` 会白白缓存每层 KV（约 100+ MB @ 1024 token）。在训练分支加一行（不影响 generate——生成时由 generate 自己管理缓存）：

```python
    model_trained.config.use_cache = False   # 训练时不缓存 KV，省 ~100MB；generate 不受影响
```

### 4.4 文案修正 + 可选 `prefix_projection=True`

见 3.4 / 3.5。

---

## 五、修复后自检清单

1. **警告消失**：日志不再出现任何 "Caching is incompatible"（若还在，说明 gc 没关干净）；
2. **真跑判据（最重要）**：训练 loss 应从 ≈4.87 的 base 水平**明显变化**——假跑时 loss 恒定不动。也可在训练前加一步梯度探针：
   ```python
   batch = data_collator([tokenized_datasets["train"][0]])
   loss = model_trained(**batch).loss; loss.backward()
   g = model_trained.prompt_encoder["default"].embedding.weight.grad
   print("prefix 梯度:", None if g is None else f"{g.norm().item():.4f}")   # None = 假跑，非 None = 真学习
   ```
3. `trainer.train()` 跑完 200 步 → **L301 不再 NameError**，评估/生成/ROUGE/保存全流程走通；
4. 微调后 eval_loss 相对基线 4.8659 应有变化；跨线对比时注意：Prefix 线与 LoRA 线都是 bf16 底座（修掉 3.3 后），eval_loss **可以**横比——这点与 QLoRA 不同；
5. 保存后 `training/qwen3-0.6b_Prefix/lora_adapter/` 下是 prompt encoder 权重（约 3.7MB，远小于 LoRA 的 20MB——0.9M 参数 × fp32）；
6. 二次运行复用分支：`PeftModel.from_pretrained` 加载 prompt tuning 适配器正常、生成对比可用；
7. （可选，做完 3.5 后）对比 prefix_projection=False/True 两版的 loss 曲线与生成质量。

---

## 六、方法论沉淀

1. **"良性警告"必须查明含义再忽略**——这批警告的真相是"训练已失效"。判定标准：警告消息里出现对你方法的**核心机制对象**做修改的字样（这里是 `past_key_value=None`），就不是良性；
2. **换 PEFT 方法时先问"它的可训练参数走哪条前向路径"**：LoRA 走权重旁路、AdaLora 走权重+KV 打分、Prefix 走 KV 注入——路径不同，与之兼容的训练设施（gc / use_cache / collator）就不同。这个检查比跑通更重要，因为这类冲突往往**静默失效**；
3. 复制脚本改造时，用 `grep -n "model_lora\|LoRA" 新脚本` 扫一遍残留——本次的 NameError 和 3 处文案都是同一类遗漏。

---

## 七、实战复盘：跑通后 loss 反涨 +119%、生成乱码、Position ids 警告

> gc 修复后训练确实"真跑"了（loss 从 11.65 缓降到 10.66，参数在动），但出现三个新现象。
> 本节用三组对照实验（GPU 实测，复现代码见 7.6）把责任定位到方法与环境组合层面。

### 7.1 现象与因果链

| 现象 | 数据 |
|---|---|
| eval_loss 起点/终点 | 基线 4.8659 → epoch1 已 11.65 → 终点 10.66（+119.1%） |
| ROUGE-L | 0.0174 → 0.0038 |
| 生成 | 全部样本乱码（"eração..." 葡语碎片循环） |
| 警告 | `Position ids are not supported for parameter efficient tuning. Ignoring position ids.` |

关键观察：**epoch 1 的 loss（11.65）就已经远高于基线**——问题发生在"包装后未训练"的时刻，训练只是在糟糕的起点上缓慢适应（-1.0），根本回不到 4.87。三个现象同根：**prefix KV 一注入，模型的 attention 就被系统性破坏**。

### 7.2 实验定音（三组对照）

**实验 1：peft 包装后、未训练的 loss**（10 条验证集，bf16）

| 状态 | 验证集 loss |
|---|---|
| 纯 base | 6.4988 |
| `get_peft_model` 包装后（未训练，peft 默认初始化） | **9.7949**（+3.3） |
| 手动把 prefix embedding 置零后 | **9.6483**（几乎没恢复！） |
| 置零后梯度探针 | grad 范数 0.157（训练可启动） |

→ **零初始化也救不了**——破坏不来自初始化值。

**实验 2：绕过 peft，transformers 原生注入同样的 KV**（单条样本）

| 注入内容 | loss |
|---|---|
| 无 prefix | 4.3575 |
| 原生注入 16 个零 KV | **8.7149** |
| 原生注入 N(0,1) KV | 13.0068 |
| 原生注入 N(0,0.05²) 小尺度 KV | 8.6979 |

→ **peft 没有实现 bug**：不经 peft、原生传零 KV 同样翻倍。破坏是"Qwen3 + KV 缓存路径"的固有行为。

**实验 3：零 KV 剂量曲线**（prefix 长度 1→256）

| 长度 | 1 | 2 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|
| loss | 8.50 | 8.49 | 8.69 | 8.71 | 7.92 | 7.19 |

→ **1 个零 KV 就 +4.1，且与长度基本无关（更长反而略好）**。这排除了"softmax 概率稀释"假设（那种机制应随长度单调恶化），说明破坏与**"是否走了 KV 缓存路径"绑定**（position 平移 / cache mask 对齐等路径切换效应），而非 prefix 的内容或数量。

### 7.3 定性与修复决策

1. **peft 0.18.1 的 Prefix Tuning × transformers 4.55 × Qwen3 的组合，存在机制性的注入门槛代价**（loss +4 左右的起点罚金）：只要给 Qwen3 传 `past_key_values`，无论值是什么，attention 行为就系统性改变。Prefix Tuning 的可训练参数**只有** KV 注入这一条路径——门槛过高时，小数据训练填不平：你的 200 步只从 11.65 恢复到 10.66（起点 +6.8 的坑），LoRA 线 -7.1% 的效果在 Prefix 线不可能复现；
2. **peft 默认 N(0,1) 初始化雪上加霜**：`PrefixEncoder`（`peft/tuners/prefix_tuning/model.py`）对 `nn.Embedding` **没有任何显式初始化**——默认 N(0,1)，其尺度（实测 std=1.0、absmax=4.7）远大于真实 K/V（~0.05 量级），额外造成 8.7→13.0 的恶化。对比 LoRA 的 `B=0` 黄金初始化（起点严格等价 base），差距是方法设计年代（GPT-2 时代）的产物；
3. **乱码的成因**：N(0,1) 的 prefix K 抢占 softmax 质量 + 训练未收敛 + 下述 position 缺陷，三重叠加下条件分布完全崩坏，输出退化为碎片循环（"eração"）；**不是**"输出了拼接的 prefix"——prefix 是 KV 向量，解码不出来文本，你看到的乱码是模型在坏 hidden 状态下的产物。

### 7.4 Position ids 警告的解释

`peft/peft_model.py` L2141：prompt learning 的 forward **无条件丢弃** `position_ids`（因为部分 prompt 方法会改变序列长度，peft 干脆忽略）。generate 时 transformers 会构造并传入 position_ids → 触发警告。Qwen3 随后 fallback 到 cache_position——本就不完全可靠（7.2 实验 2 已证明 cache 路径本身有问题）。这是 peft prompt learning 对现代 decoder 适配粗糙的一个侧面，**用户侧无法修复**，知晓即可。

### 7.5 修复方案（按目标分级）

**先想清楚目标再选**——这条线在当前环境组合下的诚实定位是"**跑通流程 + 理解机制**"，不是追求效果：

- **目标 A（理解 Prefix Tuning 机制，推荐）**：接受现状，把 7.2 的三组实验跑一遍当作学习材料——这一轮你学到的（KV 注入路径、初始化影响、cache 路径效应）比"调出一个 -5%"多得多。可选的收尾实验：把初始化改小尺度（见下）观察起点从 11.65 → ~9 的变化，验证 7.2 的结论；
- **目标 B（减少初始化伤害）**：训练分支加零/小尺度初始化（省掉 8.7→13.0 的部分）：
  ```python
  model_trained = get_peft_model(model, peft_config)
  with torch.no_grad():
      model_trained.prompt_encoder["default"].embedding.weight.zero_()
  # 或小尺度：.normal_(0, 0.05)（实验 2 显示小尺度与零相当，但梯度信号更丰富）
  ```
- **目标 C（追求效果）**：本环境不建议。若坚持：数据 100 条 → 数千条 + 大幅加训练（填平 +4~6 的起点坑后才有机会谈增益），且 `prefix_projection=True`（MLP 重参数化，输出尺度受 Linear 初始化控制）。**效果线用已验证的 LoRA/AdaLora**；
- 修掉上轮遗留：L301 的 `model_lora` → `model_trained`（NameError 仍会崩）、文案 3 处。

### 7.6 复现实验代码

```python
# 实验 2/3：原生 KV 注入对照（绕过 peft，定位责任）——GPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = "./model/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
user = "<|im_start|>user\n斯诺克一般有几种开球方法？<|im_end|>\n<|im_start|>assistant\n"
tf = tok(user + "主要分三种：红球开球、四球开球以及防守开球。<|im_end|>", return_tensors="pt")
ul = len(tok(user)["input_ids"])
labels = tf["input_ids"].clone(); labels[:, :ul] = -100
batch = {"input_ids": tf["input_ids"].cuda(), "labels": labels.cuda()}

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto", local_files_only=True).eval()
cfg = m.config
def inject(n, std):
    k = torch.zeros(1, cfg.num_key_value_heads, n, cfg.head_dim).to(m.dtype).cuda()
    if std: k = torch.randn_like(k) * std
    cache = DynamicCache()
    for i in range(cfg.num_hidden_layers):
        cache.update(k.clone(), k.clone(), i)   # v 同 k（zeros 或同随机）
    return cache
with torch.no_grad():
    print("base:", m(**batch).loss.item())
    print("零KV×16:", m(**batch, past_key_values=inject(16, 0)).loss.item())
```

（peft 路径的实验 1 代码：`get_peft_model` 包装后先跑一次验证集 loss，再 `prompt_encoder["default"].embedding.weight.zero_()` 对比——见 7.2 表。）

### 7.7 方法论沉淀

1. **"起点中性"不是 PEFT 的理所当然**：LoRA 的 B=0（起点严格等价 base）是黄金标准；Prefix Tuning 的起点罚金在本环境组合下高达 +4 以上。**新 PEFT 方法的标准第一步应该是"包装后未训练的验证集 loss"**——它量化"这个方法破坏预训练知识的程度"，一步定位很多玄学问题（本次三现象全部由此预见）；
2. **跨方法对比时先对齐"起点"**：起点不同的两条线，终点数字不可比（LoRA 起点 4.87、Prefix 起点 9.8+，终点再怎么比都是噪声）；
3. **归因靠剂量实验**：长度 1→256 的剂量曲线（7.2 实验 3）一次排除了 softmax 稀释假设——"改一个变量看响应"比读十篇博客快。

### 7.8 总结
 1. **责任归属：不全是 peft 一家的锅**

  - 实验证据显示 peft 没有实现 bug（绕过 peft、原生注入零 KV 同样翻倍）。准确的归因是三层叠加：
    - transformers 4.55 的 KV 缓存路径与 Qwen3 组合的注入门槛（主因，+4 以上）；
    - peft 的适配粗糙：N(0,1) 默认初始化（额外 +4.3，源码确认无显式 init）、丢弃 position_ids——这是 GPT-2 时代的设计没跟上现代架构（RoPE + GQA + QK-Norm + DynamicCache）；
    - 也就是说，是 “旧方法设计 × 新模型架构”的组合问题。理论上不是绝对无解（换旧版 transformers、给 prefix K 手动过 RoPE、数据量硬填），但 6GB
    学习环境里投入产出比完全不成立——“不值得解决”比“解决不了”更准确。

  1. **“loss 缓降说明有作用”——对，但要说清是哪种作用**

  - 11.65 → 10.66 证明的是训练管线是通的（prefix 参数有梯度、在优化、方向正确）——这恰好验证了我们修掉 gc 假跑的价值。但它在 +6.8 的起点坑里 200 步只爬了 1.0，这不是“Prefix Tuning
  - 有用”的证据，而是“优化器在工作”的证据。起点坑不填平，永远到不了“验证方法效果”的阶段。

  3. **业界判断：对，可以再精确一层**

  - Prefix Tuning（2021）与 LoRA 同期出道，如今 LoRA 系（LoRA/QLoRA/DoRA）确实是工业界绝对主流——效果、易用性、与推理部署的兼容性全面胜出（LoRA 可以 merge 回权重零开销推理，Prefix
  永远要背着 KV 注入）。Prefix 系至今的价值主要在学术思想层面：它的 KV 注入思路活在 P-Tuning v2（ChatGLM 生态用过）、prefix caching、attention sink
  - 等后续工作里。所以你的结论成立：做实验理解机制就够了，不必在它身上追求效果。

4.  **收尾建议**

  - 定位封存：Prefix 线就停在“流程跑通 + 机制理解”——这正好是文档 7.5 节“目标 A”，不需要再投入；
  - 收个尾：如果想让脚本完整留档可复跑，把 L301 的 model_lora → model_trained（NameError 会挡住评估流程）和 3 处"Lora"文案顺手修掉；
  - README 的成果表可以把 Prefix 行从“脚本就绪”改为“已跑通流程；机制性起点罚金，效果线不采用（详见 docs）”——一句结论留给未来的自己；
  - 下一步精力建议放在：三条已验证线的横向对比总结（LoRA -7.1% / AdaLora -4.4% / QLoRA 口径独立，同数据同超参下的相对效果 + 显存 +
  速度），这比再开一条新方法线更符合学习仓库的收束节奏。
