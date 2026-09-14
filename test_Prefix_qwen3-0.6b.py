from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
#   DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, DatasetDict
import torch

# ===================== 加载模型 =====================
# model_name = "Qwen/Qwen3-0.6B"
model_path = "./model/Qwen3-0.6B" 

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)
# 新增
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
#    model_name,
    model_path,
    torch_dtype="auto",
    # device_map="auto",  # ❗删掉这一行！Trainer会自动迁移model到device
    local_files_only=True
)

print("模型加载成功！")

# ===================== 加载数据集 =====================
raw_datasets = load_dataset("./datasets/data")
print("原始数据集：", raw_datasets)


# 严格切分三份 train:80%  validation:10%  test:10%
# 第一次切分：取出20%作为临时集，剩余80%为train
first_split = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
ds_train = first_split["train"]
temp_dataset = first_split["test"]

# 对临时20%做二次切分，对半分：validation 10%，test 10%
second_split = temp_dataset.train_test_split(test_size=0.5, seed=42)

# 组装为标准DatasetDict，三个key：train / validation / test
raw_datasets = DatasetDict({
    "train": ds_train,
    "validation": second_split["train"],
    "test": second_split["test"]
})

print("切分完成：", raw_datasets)
# train ~805k；validation ~100k；test ~100k

# 索引取第0条样本，方括号 []，不是 ()
sample = raw_datasets["train"][0]
print(sample)
print("指令：", sample["INSTRUCTION"])
print("回答：", sample["RESPONSE"])


# ===================== 分词 =====================
# tokenize函数（注意：这里还没有做labels mask，仅做分词
# batched=True，入参example是batch，字段值是list
'''
def tokenize_func(example):
    texts = []
    for ins, resp in zip(example["INSTRUCTION"], example["RESPONSE"]):
        prompt = f"<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n{resp}<|im_end|>"
        texts.append(prompt)
    return tokenizer(
        texts,
        truncation=True,
        max_length=4096,
        padding="max_length"
    )
'''
def tokenize_func(example):
    # full_texts：存放【完整对话】 user+assistant
    # user_parts：存放【仅用户prompt部分】，用来确定需要mask的token长度
    full_texts = []
    user_parts = []

    # batched=True：example["INSTRUCTION"] / example["RESPONSE"] 是一批样本的list
    # 循环batch内每一条样本，拼接Qwen‑IM标准对话模板
    for ins, resp in zip(example["INSTRUCTION"], example["RESPONSE"]):
        # user_str：<|im_start|>user + 用户指令 + <|im_end|> + <|im_start|>assistant\n
        # 注意末尾带上assistant起始标记，这一段全部要mask掉，不参与loss
        user_str = f"<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n"
        # full_str：完整样本 = 用户模板 + 模型回答 + assistant结束标记
        full_str = user_str + resp + "<|im_end|>"

        full_texts.append(full_str)
        user_parts.append(user_str)

    # 对完整文本做tokenize，得到input_ids、attention_mask；固定长度4096
    # padding="max_length"：不足4096补pad；超过则截断
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
 #      max_length=4096,
        max_length=1024,   # 由4096改为1024
 #      padding="max_length"
    )

    # 单独只对【用户prompt片段】做分词，得到用户部分的token数量 user_len
    # 不需要padding，只需要拿到真实token个数
    tokenized_user = tokenizer(user_parts, truncation=True, max_length=4096)

    labels = []
    # 遍历batch中每一条样本的input_ids 和 用户部分tokenId
    for input_ids, user_ids in zip(tokenized_full["input_ids"], tokenized_user["input_ids"]):
        # labels初始复制input_ids：因果LM标准做法，labels和input_ids形状完全一致
        label = input_ids.copy()
        # 用户prompt所占token长度
        user_len = len(user_ids)

        # ----------------核心mask逻辑----------------
        # 将user部分所有token label设置为‑100
        # CrossEntropyLoss会自动跳过值为‑100的位置，不对用户prompt计算loss
        # 保留从 user_len 往后：assistant回答 + <|im_end|>，这部分才计算loss
        label[:user_len] = [-100] * user_len

        labels.append(label)

    # 将构造好的labels加入分词结果字典，map输出会带上input_ids/attention_mask/labels
    tokenized_full["labels"] = labels
    return tokenized_full

# cache_dir给目录，map自动处理 train/validation/test 三个分片
tokenized_datasets = DatasetDict()
tokenized_datasets["train"] = raw_datasets["train"].map(
    tokenize_func, 
    batched=True, 
    batch_size=2000, 
    cache_file_name="./datasets/zhihu-kol/cache/train.arrow")
tokenized_datasets["validation"] = raw_datasets["validation"].map(
    tokenize_func,
    batched=True, 
    batch_size=2000, 
    cache_file_name="./datasets/zhihu-kol/cache/val.arrow")
tokenized_datasets["test"] = raw_datasets["test"].map(
    tokenize_func,
    batched=True, 
    batch_size=2000, 
    cache_file_name="./datasets/zhihu-kol/cache/test.arrow")

print("分词后数据集：", tokenized_datasets)

# ===================== 训练 =====================
'''
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # mlm=False 因果LM；True是bert掩码语言模型
)
'''
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./training/output",
    logging_dir="./training/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # ========== OOM修复参数 ==========
    per_device_train_batch_size=1,      # 6G卡，全量微调建议1；实在不行用 gradient_accumulation_steps
    gradient_accumulation_steps=4,       # 梯度累积，模拟 bs=4
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,         # ✅ 梯度检查点，大幅降低激活显存，速度会慢一点
    fp16=True,                           # 开启fp16混合精度，不要用torch_dtype=auto带来的bf16；6G卡优先fp16
    # bf16=False,

    # 评估不要太频繁，eval也占显存
    eval_strategy="steps",
    eval_steps=200,

    # 显存碎片优化
    optim="adamw_torch_fused",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer
)


print("开始 SFT 微调：")
trainer.train()


# ===================== 评估 =====================
print("开始评估：")
metrics = trainer.evaluate(tokenized_datasets["validation"])
print(metrics)

tokenizer.padding_side = "left" # generate必须left padding

def generate_sample(batch, idx=0):
    input_ids = torch.tensor(batch["input_ids"][idx:idx+1]).to(model.device)
    attention_mask = torch.tensor(batch["attention_mask"][idx:idx+1]).to(model.device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# 取validation前2条做生成演示，不要全量！
for i in range(2):
    text_out = generate_sample(tokenized_datasets["validation"], i)
    print(f"==== sample {i} ====")
    print(text_out)
