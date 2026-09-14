from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. 路径配置
#odel_path = "./model/bert-base-uncased"          # 原始BERT主干
lora_adapter_path = "./training/bert-base-uncased_Lora/lora_adapter" # LoRA适配器保存目录
peft_config = PeftConfig.from_pretrained(lora_adapter_path)
# 自动拿到base model路径
model_path = peft_config.base_model_name_or_path

# 2. 读取LoRA配置
peft_config = PeftConfig.from_pretrained(lora_adapter_path)

# 3. 加载原始BERT主干（num_labels必须和训练保持一致！）
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    torch_dtype="auto",
    local_files_only=True,
    device_map="auto"
)

# 4. 加载LoRA适配器，融合到主干模型
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.eval() # 推理评估务必切eval模式

# 加载tokenizer（tokenizer属于主干，单独加载）
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# ---------------- 评估验证集示例（和你训练时evaluate逻辑一致） ----------------
from datasets import load_dataset
from transformers import DataCollatorWithPadding, Trainer
import evaluate

raw_datasets = load_dataset("./datasets/glue","mrpc")
def tokenize_func(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=128,
        padding=False
    )
tokenized_val = raw_datasets["validation"].map(tokenize_func, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("./datasets/evaluate/metrics/glue/glue.py", config_name="mrpc")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 构造临时Trainer做评估
eval_trainer = Trainer(
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)
res = eval_trainer.evaluate(tokenized_val)
print("加载LoRA后验证集指标：", res)
