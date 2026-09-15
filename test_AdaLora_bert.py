from peft import (
    AdaLoraConfig,     # ← 替换 LoraConfig
    TaskType,
    get_peft_model,
    # get_peft_config,
    PeftModel,        # 新增：加载已保存的适配器
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from datasets import load_dataset, DatasetDict
import evaluate, numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import random, torch, os       # ← 追加 os（has_adapter 检测要用）
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True


# ===================== 加载模型 =====================
model_path = "./model/bert-base-uncased" 
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",  # 自动选择 GPU/CPU
    local_files_only=True,
    num_labels=2,       # MRPC二分类
)

print("模型加载成功！")
print(model)

# ===================== 加载数据集 =====================
raw_datasets = load_dataset("./datasets/glue","mrpc")
print("原始数据集：", raw_datasets)
raw_train_dataset = raw_datasets["train"]
print("训练集样本数：", len(raw_train_dataset))
print("第一条样本：", raw_train_dataset[0])


# ======================= 分词 ========================
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)

def tokenize_func(examples):
    # 对每条样本的两个句子进行分词
    return tokenizer(
        examples["sentence1"], 
        examples["sentence2"], 
        truncation=True, 
        max_length=128, 
        padding=False
    )

# tokenized_datasets = raw_datasets.map(tokenize_func, batched=True)
# print("分词后数据集：", tokenized_datasets)

# 分开map，每个split指定独立cache_file_name
tokenized_datasets = DatasetDict()

tokenized_datasets["train"] = raw_datasets["train"].map(
    tokenize_func,
    batched=True,
    batch_size=2000,
    cache_file_name="./datasets/glue/cache-mrpc/train.arrow"
)

tokenized_datasets["validation"] = raw_datasets["validation"].map(
    tokenize_func,
    batched=True,
    batch_size=2000,
    cache_file_name="./datasets/glue/cache-mrpc/val.arrow"
)

tokenized_datasets["test"] = raw_datasets["test"].map(
    tokenize_func,
    batched=True,
    batch_size=2000,
    cache_file_name="./datasets/glue/cache-mrpc/test.arrow"
)

print("分词后数据集：", tokenized_datasets)


# ================== MRPC评估指标 ================
metric = evaluate.load("./datasets/evaluate/metrics/glue/glue.py", config_name="mrpc")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    res = metric.compute(predictions=predictions, references=labels)
    res["macro_f1"] = f1_score(labels, predictions, average="macro")
    return res

# ================== 训练&评估 ===================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# baseline 评估用，不需要保存/eval策略，最简配置
training_args_baseline = TrainingArguments(
    output_dir="./training/bert-base-uncased_AdaLora/output",
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
    bf16=True
)

trainer_baseline = Trainer(
    model=model,
    args=training_args_baseline,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# 训练前基线评估（原始预训练权重）
baseline_result = trainer_baseline.evaluate()
print("==== 预训练模型原始基线 ====")
print(baseline_result)


# ===================== 加载 =====================
lora_save_path = "./training/bert-base-uncased_AdaLora/lora_adapter"
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))

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
                            # 平均每次砍 (432−288)/200 ≈ 0.7 个三元组，节奏平缓   
        lora_alpha = 16,                 # 一般为2r    
        lora_dropout = 0.1               # 防止过拟合
    )
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()  # 打印可训练参数

training_args = TrainingArguments(
    output_dir="./training/bert-base-uncased_AdaLora/output",
    logging_dir="./training/bert-base-uncased_AdaLora/logs",
    report_to="tensorboard",       # 显式指定：默认"all"会顺带探测wandb等集成；配合logging_dir查看曲线
    logging_strategy="steps",
    logging_steps=10,
    # save_strategy="steps",
    # save_steps=100,
    save_strategy="epoch",
    save_total_limit=3,
    # eval_strategy="steps",        # 旧版本用这个，不要写evaluation_strategy
    # eval_steps=100,
    eval_strategy="epoch",
    per_device_train_batch_size=4,   # 从8降到4，降低激活显存
    per_device_eval_batch_size=8,    # eval也降
    num_train_epochs=3,
    learning_rate=5e-4,
    bf16=True,
    gradient_checkpointing=True,     # ✅开启梯度检查点，大幅降低激活显存，代价训练速度变慢
    load_best_model_at_end=False,    # ✅关闭自动加载最优模型，避免与 AdaLora 的秩预算调度冲突
    # metric_for_best_model="f1",
    metric_for_best_model="macro_f1",
    weight_decay=0.01    
)

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

# AdaLora微调
if not has_adapter:
    trainer.train()

# 训练结束完整跑一次验证集
eval_res = trainer.evaluate()
print("==== 最终验证集结果 ====")
print(eval_res)


# ================== 预测 ===================
predictions = trainer.predict(tokenized_datasets["validation"])
print("==== 验证集预测结果 ====")
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)
print("预测标签：", preds)
labels = predictions.label_ids
print("混淆矩阵：\n", confusion_matrix(labels, preds))


# ============ 保存LoRA适配器（关键代码） ============
if not has_adapter:
    # 保存LoRA权重、adapter配置，不保存BERT主干
    trainer.save_model(lora_save_path)
    # 等价：model.save_pretrained(lora_save_path)
    print(f"LoRA适配器已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")

# ======= 检查 mask 是否生效（lora_E 的零值比例应从 0 升至约 1/3）=======
for n, p in model_lora.named_parameters():
    if "lora_E" in n:
        print(n, f"{(p == 0).float().mean().item():.2%}")