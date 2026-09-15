from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset, DatasetDict
import evaluate, numpy as np


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
# tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    return metric.compute(predictions=predictions, references=labels)


# ================== 训练&评估 ===================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    metric_for_best_model="f1",
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

# 训练前基线评估（原始预训练权重）
baseline_result = trainer.evaluate()
print("==== 预训练模型原始基线 ====")
print(baseline_result)

# SFT微调
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