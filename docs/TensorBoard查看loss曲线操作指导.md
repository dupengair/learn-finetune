# TensorBoard 查看 eval_loss 曲线操作指导

> 适用场景：确认 `test_Prompt_qwen3-0.6b.py` 训练的 eval_loss 曲线是否健康（单调下降、无回升）。
> 本指导经过实际环境验证（ai-gpu 环境，tensorboard 2.20.0），转换脚本已实测跑通。

---

## 一、先回答包的问题：你已经装好了，但装多的那个用不上

| 包 | 状态 | 作用 | 看曲线是否需要 |
|---|---|---|---|
| `tensorboard` 2.20.0 | ✅ **环境里本来就有** | TensorBoard 本体，Scalars 面板看 loss 曲线 | **需要，且已满足** |
| `tensorboardX` 2.6.4 | ✅ 已有 | 第三方写 event 的库 | 备用，不需要 |
| `torch-tb-profiler` 0.4.3 | 你新装的 | **性能分析**插件（算子耗时、显存 profiling） | ❌ 不需要（无害，留着也行） |
| `tensorboard_plugin_profile` 2.21.4 | ✅ 已有 | 同上，profiling 插件 | ❌ 不需要 |

结论：**看 loss 曲线不需要额外安装任何包**，`torch-tb-profiler` 是查性能瓶颈用的，与本任务无关。

---

## 二、两个必须先知道的事实（为什么你现在看不到曲线）

### 坑 1：本次训练没有写 TensorBoard 日志

脚本里 `report_to="none"`（OOM 修复时顺手关掉了所有上报），Trainer 就不会创建 TensorBoardCallback——**`training/qwen3-0.6b_Prompt/logs/` 和 `log/` 目录都是空的**，没有任何 event 文件可看。

### 坑 2：训练数据其实没丢，但只有 eval_loss

每个 checkpoint 里都有 `trainer_state.json`，其中的 `log_history` 完整记录了每次评估的结果。实测你这次训练的 10 条记录：

```
epoch 1: 4.4672   epoch 2: 4.3702   epoch 3: 4.3569   epoch 4: 4.3130   epoch 5: 4.3283
epoch 6: 4.3099   epoch 7: 4.2970 ←最低   epoch 8: 4.3091   epoch 9: 4.3136   epoch 10: 4.3118
```

但**没有 train_loss**：`logging_steps` 默认 500，而全程只有 200 步 → 训练 loss 一次都没被记录。所以无论怎么改，本次都只有 eval_loss 一条曲线（想同时看 train_loss，见第四节路线二的两行改动）。

---

## 三、路线一（推荐）：把已有数据转成 TensorBoard 日志，立即查看本次结果

不用重新训练。分两步：**转换 → 启动 TensorBoard**。

### 第 1 步：创建转换脚本

在仓库根目录新建 `tb_export_from_trainer_state.py`（已验证可运行）：

```python
# tb_export_from_trainer_state.py
# 用途：把 checkpoint 里 trainer_state.json 的 log_history 转成 TensorBoard event 文件
# 用法：从仓库根目录运行  python tb_export_from_trainer_state.py
import json
import os
from torch.utils.tensorboard import SummaryWriter

ckpt_dir = "./training/qwen3-0.6b_Prompt/output/checkpoint-200"  # 最新的 checkpoint
log_dir = "./training/qwen3-0.6b_Prompt/logs"                    # 与训练脚本的 logging_dir 保持一致

with open(os.path.join(ckpt_dir, "trainer_state.json"), encoding="utf-8") as f:
    log_history = json.load(f)["log_history"]

writer = SummaryWriter(log_dir=log_dir)
count = 0
for h in log_history:
    # tag 用 "分组/名称" 格式，TensorBoard 会自动按前缀分组显示
    if "eval_loss" in h:
        writer.add_scalar("eval/loss", h["eval_loss"], h["step"])
        count += 1
    if "loss" in h:
        writer.add_scalar("train/loss", h["loss"], h["step"])
        count += 1
writer.close()
print(f"已写入 TensorBoard 日志到 {log_dir}，共 {count} 条记录")
```

运行（务必从仓库根目录，相对路径才能对上）：

```bash
python tb_export_from_trainer_state.py
# 预期输出：已写入 TensorBoard 日志到 ./training/qwen3-0.6b_Prompt/logs，共 10 条记录
```

### 第 2 步：启动 TensorBoard 并在浏览器查看

```bash
tensorboard --logdir=./training/qwen3-0.6b_Prompt/logs --port=6006
```

- 看到提示 `TensorBoard 2.20.0 at http://localhost:6006/` 即启动成功；
- **WSL2 环境**：直接在 Windows 浏览器打开 `http://localhost:6006`（WSL2 默认自动转发 localhost）；
- 如果提示端口被占用，换 `--port=6007`；
- 用完在终端按 `Ctrl+C` 停止服务。

> 若 localhost 转发不通（少数 WSL 配置），改用 `tensorboard --logdir=... --port=6006 --bind_all`，然后在 Windows 浏览器访问 `http://<WSL的IP>:6006`（WSL IP 用 `hostname -I` 查询）。

### 第 3 步：浏览器里的操作

1. 打开 `http://localhost:6006`，顶部导航切到 **Scalars** 面板；
2. 左侧只勾选 **eval/loss**（本次只有这一条曲线）；
3. 右上角 **Smoothing（平滑）** 滑块：只有 10 个点的稀疏曲线建议拖到 **0.3~0.5**，太高（>0.8）会掩盖真实波动；
4. 曲线横轴默认是 **STEP**（= optimizer step，每 20 步一个 epoch）；可用左栏 X-axis 切换成 **RELATIVE**（按 wall 时间）或 **WALL**，学习时看 STEP 即可；
5. 鼠标悬停曲线上的点可看精确数值（step、epoch、eval_loss）。

### 第 4 步：对照着确认什么（本次数据的预期形态）

打开曲线后，你应该看到与第二节列出数值一致的形态，据此做三个判断：

| 判断项 | 本次实际结论 |
|---|---|
| 最低点在哪 | **epoch 7（step 140）= 4.2970**，前 7 个 epoch 基本单调下降（仅 epoch 5 微抖） |
| 是否回升（过拟合迹象） | epoch 7 之后 3 个 epoch 持平微升 ~0.4%（4.3091/4.3136/4.3118），幅度接近噪声，**属平台期，无严重过拟合** |
| 训练是否还要加 | 不必加 epoch——已进入平台期，加训练量收益趋零；想再降应改数据量或方法（这正是两条 PEFT 实验线对比的意义） |

**一个附带教训**：最优权重在 checkpoint-140（epoch 7），但 `save_total_limit=3` 只保留最近 3 个 checkpoint（160/180/200），**140 已被轮换删除**。下次想「训练完拿最优而不是最新的」，需要两个改动配合：`save_total_limit` 放宽 + `load_best_model_at_end=True`（Trainer 会按 eval_loss 自动加载最优 checkpoint，配合 `metric_for_best_model="eval_loss"`）。

---

## 四、路线二：下次训练让 Trainer 直接写 TensorBoard（治本）

下次跑训练脚本前，在 `test_Prompt_qwen3-0.6b.py` 的 `training_args` 里改两行：

```python
training_args = TrainingArguments(
    output_dir="./training/qwen3-0.6b_Prompt/output",
    logging_dir="./training/qwen3-0.6b_Prompt/logs",
    report_to="tensorboard",      # ★ 原 "none" 改这里：启用 TensorBoard 上报
    logging_steps=20,             # ★ 新增：默认 500 > 全程 200 步，不改则 train_loss 一条都不记；
                                  #   20 = 每个 epoch 记一次，与 eval 对齐，方便两曲线对照
    ...
)
```

改完后正常训练，训练过程中/后执行：

```bash
tensorboard --logdir=./training/qwen3-0.6b_Prompt/logs --port=6006
```

 Scalars 面板会同时出现两条曲线（训练时实时刷新，页面右上角可开自动刷新）：

- **train/loss**：训练过程每个 logging 点的损失（含噪声，看整体趋势）；
- **eval/loss**：验证集损失（干净的泛化指标，判断过拟合就靠它）。

**健康曲线的判读标准**：eval_loss 随 train_loss 同步下降为健康；eval_loss 下降转回升而 train_loss 继续下降 = 开始过拟合（本次实验已在 epoch 7 附近出现平台，属于「正常收敛」而非过拟合）。

---

## 五、常见问题速查

| 现象 | 原因与处理 |
|---|---|
| 浏览器打开 Scalars 是空白 | event 文件路径不对：确认 `--logdir` 指向的目录里有 `events.out.tfevents.*` 文件 |
| 曲线只有一个点 | 正常：本次只有 10 条 eval 记录；想要密曲线 → 路线二里把 `eval_strategy`/`logging_steps` 改成 `"steps"` + 更小间隔 |
| `tensorboard: command not found` | conda 环境没激活：`conda activate ai-gpu`，或用全路径 `/home/dupengair/shared/conda/anaconda3/envs/ai-gpu/bin/tensorboard` |
| Address already in use | 换端口 `--port=6007`，或先 `pkill -f tensorboard` |
| 重复跑了转换脚本出现多条同名曲线 | event 文件是追加式的，清空 `logs/` 后重跑转换脚本即可 |
