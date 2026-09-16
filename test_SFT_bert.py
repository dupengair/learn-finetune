from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
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
    num_labels=2,  # 二分类
    local_files_only=True
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


# ===================== 微调前基线评估 ==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args_baseline = TrainingArguments(
    output_dir="./training/bert-base-uncased_SFT/output",
    per_device_eval_batch_size=8,     # 与主 Trainer 一致（6G 卡 OOM 规避）
    bf16=True,
    do_train=False,
    do_eval=True,
    report_to="none",
)

trainer_baseline = Trainer(
    model=model,                      # ← 原始 model，未包装
    args=training_args_baseline,
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics      # ★ 补上：基线也要有 acc/f1/macro_f1
)

# 训练前基线评估（原始预训练权重）
baseline_result = trainer_baseline.evaluate()
print("==== 预训练模型原始基线 ====")
print(baseline_result)


# ===================== 加载 =====================
lora_save_path = "./training/bert-base-uncased_SFT/lora_adapter"   # ← 从文件末尾上移到此处定义
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "model.safetensors"))

if has_adapter:
    # ---------- 模式一：加载已训练的全量权重，跳过训练 ----------
    print(f"检测到已保存的全量微调权重：{lora_save_path}，跳过训练")
    # SFT没有adapter概念：直接from_pretrained重载整个微调后模型（config+tokenizer已随权重保存）
    # 注意基线评估已在上方完成——全量微调覆盖了权重，无法像LoRA那样disable_adapter()切回base
    del trainer_baseline, model          # 释放旧引用：否则旧base权重(~1.2G)仍被trainer_baseline.model
                                         # 持有，重载后显存里同时存在两份0.6B模型
    model = AutoModelForSequenceClassification.from_pretrained(
        lora_save_path,
        torch_dtype="auto",
        local_files_only=True
    )   # 重载后在CPU；下方主Trainer构造时会自动迁移到cuda，无需手动.to()
else:
    # ---------- 模式二：首次运行，保持当前 base model 直接进入训练 ----------
    pass


# ================== 训练 ===================
training_args = TrainingArguments(
    output_dir="./training/bert-base-uncased_SFT/output",
    logging_dir="./training/bert-base-uncased_SFT/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    eval_strategy="steps",        # 旧版本用这个，不要写evaluation_strategy
    eval_steps=100,
    per_device_train_batch_size=4,   # 从8降到4，降低激活显存
    per_device_eval_batch_size=8,    # eval也降
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=True,     # ✅开启梯度检查点，大幅降低激活显存，代价训练速度变慢
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="tensorboard",       # 显式指定：默认"all"会顺带探测wandb等集成；配合logging_dir查看曲线
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

if not has_adapter:
    print("开始 SFT 微调：")
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


# ============ 保存全量微调权重（关键代码） ============
if not has_adapter:
    # SFT保存完整模型权重(~1.2GB bf16)——与LoRA只存adapter(几十MB)完全不同
    trainer.save_model(lora_save_path)
    # 等价：model.save_pretrained(lora_save_path)
    print(f"全量微调权重已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")