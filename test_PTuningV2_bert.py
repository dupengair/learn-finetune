from peft import (
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    PeftModel
)

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


# ===================== 微调前基线评估（peft 包装前：原始 model）==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# baseline 评估用，不需要保存/eval策略，最简配置
training_args_baseline = TrainingArguments(
    output_dir="./training/bert-base-uncased_PTuning-V2/output",
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
lora_save_path = "./training/bert-base-uncased_PTuning-V2/lora_adapter"
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))

if has_adapter:
    # ---------- 模式一：加载已训练权重，跳过训练 ----------
    print(f"检测到已保存的P-TuningV2适配器：{lora_save_path}，跳过训练")
    # ★ is_trainable=True 必须：evaluate/predict 的前向都带 labels，
    #   AdaLoraModel.forward 正则段访问 trainable_adapter_name（默认 False 时该属性
    #   不存在 → AttributeError，Qwen 报告第八节的坑在 BERT 侧同样成立）
    model_trained = PeftModel.from_pretrained(model, lora_save_path)
    # 此时 print 显示 ~0.5%（55万上下）而非 0%——rank_pattern 缩形重建的新参数
    # 逃过了 _freeze_adapter（Qwen 报告 8.4）。无害：本分支不会 trainer.train()，
    # evaluate/predict 都在 no_grad 下执行，不会发生任何参数更新
    model_trained.print_trainable_parameters()
else:
    # ---------- 模式二：首次运行，完整训练流程 ----------

    peft_config = PrefixTuningConfig(
        peft_type="PREFIX_TUNING",       # P-Tuning v2 = Deep Prompt Tuning（v2论文：Prefix Tuning机制的NLU适配版）    
        task_type = TaskType.SEQ_CLS,    # 序列分类，不是seq2seq
        num_virtual_tokens=16,           # 每层注入的可学习前缀长度
        prefix_projection=False,         # ★ v2关键设计：去掉重参数化MLP；设True则回到原版Prefix Tuning
        inference_mode=False
    )
    model_trained = get_peft_model(model, peft_config)
    model_trained.config.use_cache = False      # 训练时不缓存 KV，省 ~100MB；generate 不受影响
    model_trained.print_trainable_parameters()     # 打印可训练参数

training_args = TrainingArguments(
    output_dir="./training/bert-base-uncased_PTuning-V2/output",
    logging_dir="./training/bert-base-uncased_PTuning-V2/logs",
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
    num_train_epochs=5,
    learning_rate=1e-3,
    bf16=True,
    gradient_checkpointing=False,        # ★ v2必须关！gc把past_key_values置None → backward直接RuntimeError
    load_best_model_at_end=True,
    # metric_for_best_model="f1",
    metric_for_best_model="macro_f1",
    weight_decay=0.01    
)

trainer = Trainer(
    model=model_trained,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# 微调
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


# ============ 保存P-TuningV2适配器（关键代码） ============
if not has_adapter:
    # 保存LoRA权重、adapter配置，不保存BERT主干
    trainer.save_model(lora_save_path)
    # 等价：model.save_pretrained(lora_save_path)
    print(f"LoRA适配器已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")