from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, DatasetDict

# ===================== 加载模型 =====================
# model_name = "Qwen/Qwen3-0.6B"
model_path = "./model/Qwen3-0.6B" 

# load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)
model = AutoModelForCausalLM.from_pretrained(
#    model_name,
    model_path,
    torch_dtype="auto",
    device_map="auto",  # 自动选择 GPU/CPU
    local_files_only=True
)

print("模型加载成功！")

# ===================== 加载数据集 =====================
raw_datasets = load_dataset("./datasets/zhihu-kol/data")
print("原始数据集：", raw_datasets)


# 严格切分三份 train:80%  validation:10%  test:10%
# 第一次切分：取出20%作为临时集，剩余80%为train
raw_datasets["train"] = raw_datasets["train"].select(range(100))      # 100条训练
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
        max_length=1024,
        padding="max_length"
    )

    # 单独只对【用户prompt片段】做分词，得到用户部分的token数量 user_len
    # 不需要padding，只需要拿到真实token个数
    tokenized_user = tokenizer(user_parts, truncation=True, max_length=1024)

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
