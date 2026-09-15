from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,        # 新增：加载已保存的适配器
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset, DatasetDict
import evaluate as ev
import torch, os


# ===================== 加载模型 =====================
model_path = "./model/Qwen3-0.6B" 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

print("模型加载成功！")
print(model)


# ===================== 加载数据集 =====================
raw_datasets = load_dataset("./datasets/zhihu-kol/data")
print("原始数据集：", raw_datasets)


# 严格切分三份 train:80%  validation:10%  test:10%
# 第一次切分：取出20%作为临时集，剩余80%为train
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(100))
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
# train:80%  validation:10%  test:10%

# 索引取第0条样本，方括号 []，不是 ()
sample = raw_datasets["train"][0]
print(sample)
print("指令：", sample["INSTRUCTION"])
print("回答：", sample["RESPONSE"])


# ===================== 分词 =====================
# tokenize函数（注意：这里还没有做labels mask，仅做分词
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)
# 新增
tokenizer.pad_token = tokenizer.eos_token

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

    # 对完整文本做tokenize，得到input_ids、attention_mask；不 padding（padding 由 collator 动态做）
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
        max_length=1024
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
        # 防止用户prompt过长被截断后越界
        user_len = min(user_len, len(input_ids))
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


# ===================== 微调前基线评估（peft 包装前：原始 model）==========
'''
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # 因果语言模型，不是掩码语言模型
)
'''

# ⚠️ 不能用 DataCollatorForLanguageModeling，三个坑（详见 docs 第十二节）：
#   ① tokenizer.pad 不填充 labels，eval批量>1时长度参差会崩
#   ② mlm=False 分支用 input_ids 覆盖 labels，prompt=-100 的 mask 静默失效
#   ③ labels==pad_token_id(即<|im_end|>)被改成-100，模型学不到停止符
def custom_collate_fn(features):
    """动态padding到batch内最长：input_ids补pad_token_id、attention_mask补0、labels补-100"""
    max_len = max(len(f["input_ids"]) for f in features)
    return {
        "input_ids": torch.tensor(
            [f["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(f["input_ids"])) for f in features],
            dtype=torch.long),
        "attention_mask": torch.tensor(
            [f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features],
            dtype=torch.long),
        "labels": torch.tensor(
            [f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features],
            dtype=torch.long),
    }

data_collator = custom_collate_fn

training_args_baseline = TrainingArguments(
    output_dir="./training/qwen3-0.6b_Lora/output",
    per_device_eval_batch_size=1,     # 与主 Trainer 一致（6G 卡 OOM 规避）
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
)

# 固定同一批测试样本（微调前后必须完全一致）
N_COMPARE = 5    # 验证集不足5条就调小
test_instructions = [raw_datasets["validation"][i]["INSTRUCTION"] for i in range(N_COMPARE)]
test_references   = [raw_datasets["validation"][i]["RESPONSE"]    for i in range(N_COMPARE)]

# 微调前基线评估 + 生成对比准备,基线评估在下方按 has_adapter 分两种模式：
# 训练模式 B=0≡base 直接评；加载模式用 disable_adapter() 切回 base
def generate_answer(m, instruction, max_new_tokens=256):
    """方案B核心：只用指令构造提示（不含答案！），greedy 解码保证前后对比可复现"""
    user_str = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(user_str, return_tensors="pt").to(m.device)
    m.eval()   # 关掉 dropout(lora_dropout=0.1)，否则 train 模式下 greedy 也不可复现
    with torch.no_grad():
        output = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # greedy：同输入必同输出，前后对比才有效
            temperature=1.0,     # ← 中性值，消除警告；贪心模式下无效参数
            top_p=1.0,           # ←
            top_k=50,            # ←
            pad_token_id=tokenizer.pad_token_id
        )
    # generate 返回 = 提示原样在前 + 新生成在后；切掉提示只解码新生成部分
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

print("==== 预训练模型原始基线 ====")
baseline_eval = trainer_baseline.evaluate()
print(baseline_eval)
# 微调前基线生成（同样是 B=0 的模型在生成） 
baseline_generations = [generate_answer(model, ins) for ins in test_instructions]
print("基线生成完成")


# ===================== 加载 =====================
lora_save_path = "./training/qwen3-0.6b_Lora/lora_adapter"   # ← 从文件末尾上移到此处定义
# 判断"保存完成"：adapter_config.json 存在（只看目录会误判保存中断的残缺目录）
has_adapter = os.path.exists(os.path.join(lora_save_path, "adapter_config.json"))

if has_adapter:
    # ---------- 模式一：加载已训练权重，跳过训练 ----------
    print(f"检测到已保存的LoRA适配器：{lora_save_path}，跳过训练")
    model_lora = PeftModel.from_pretrained(model, lora_save_path)
    # 默认 is_trainable=False：适配器权重 requires_grad=False，仅推理/评估
    # 此时 print_trainable_parameters 会显示 0% 可训练——这是预期的
    model_lora.print_trainable_parameters()
else:
    # ---------- 模式二：首次运行，完整训练流程 ----------
    peft_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,    
        inference_mode = False,   
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias = "none",
        r = 8,    
        lora_alpha = 16,                 # 一般为2r    
        lora_dropout = 0.1               # 防止过拟合
    )
    model_lora = get_peft_model(model, peft_config)
    model_lora.enable_input_require_grads()  # ✅ 新增：让冻结embedding的输出可求导，配合梯度检查点
    model_lora.print_trainable_parameters()  # 打印可训练参数


# ===================== 训练 =====================
training_args = TrainingArguments(
    output_dir="./training/qwen3-0.6b_Lora/output",
    logging_dir="./training/qwen3-0.6b_Lora/logs",
    logging_strategy="steps",
    logging_steps=10,
    #eval_strategy="steps",
    eval_strategy="epoch",
    #eval_steps=1000,
    #save_strategy="steps",
    save_strategy="epoch",
    #save_steps=200,
    save_total_limit=3,

    # ========== OOM修复参数 ==========
    per_device_train_batch_size=1,       # 6G卡，全量微调建议1；实在不行用 gradient_accumulation_steps
    gradient_accumulation_steps=4,       # 梯度累积，模拟 bs=4
    per_device_eval_batch_size=1,        # =8 会OOM
    gradient_checkpointing=False,        # ✅ 梯度检查点，大幅降低激活显存，速度会慢一点
    gradient_checkpointing_kwargs={      # ✅ 新增：非reentrant检查点，不依赖输入梯度
        "use_reentrant": False
    },
    #fp16=False,                          # 开启fp16混合精度，不要用torch_dtype=auto带来的bf16；6G卡优先fp16
    bf16=True,

    # 显存碎片优化
    optim="adamw_torch_fused",
    report_to="tensorboard",       # 启用TensorBoard上报loss曲线，用法见 docs/TensorBoard查看loss曲线操作指导.md
)

trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer
)

if not has_adapter:
    print("开始 Lora 微调：")
    trainer.train()


# ===================== 评估 =====================
print("开始评估：")
# 微调后 eval_loss
print("==== LoRA 微调后 ====")
after_eval = trainer.evaluate()
print(after_eval)
delta = (after_eval["eval_loss"] - baseline_eval["eval_loss"]) / baseline_eval["eval_loss"] * 100
print(f"eval_loss 对比：{baseline_eval['eval_loss']:.4f} -> "
      f"{after_eval['eval_loss']:.4f}（{delta:+.1f}%）")

# 微调后生成同一批指令
after_generations = [generate_answer(model_lora, ins) for ins in test_instructions]

# 前后对比打印 + 可选 ROUGE
for i in range(N_COMPARE):
    print(f"\n======== 样本 {i} ========")
    print(f"【指令】{test_instructions[i]}")
    print(f"【微调前 base】{baseline_generations[i]}")
    print(f"【微调后 LoRA】{after_generations[i]}")
    ref = test_references[i]
    print(f"【金标参考】{ref[:200]}{'...' if len(ref) > 200 else ''}")

# （可选）ROUGE-L 定量对比。
# 两个注意点：
# ① vendored rouge 指标内部是英文分词（非 a-z0-9 字符全部过滤），中文直接算全为 0，
#    必须逐字加空格"伪装"成英文词后再算（字符级 n-gram 粗略指标，仅看相对变化）；
# ② 依赖 rouge_score / nltk 包，离线环境未安装会报错，try 包住可优雅跳过。
try:    
    rouge = ev.load("./datasets/evaluate/metrics/rouge/rouge.py")
    # zh = lambda texts: [" ".join(t) for t in texts]   # 中文逐字切开
    zh = lambda texts: [" ".join(t.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "")) for t in texts]
    r_before = rouge.compute(predictions=zh(baseline_generations), references=zh(test_references))
    r_after  = rouge.compute(predictions=zh(after_generations),  references=zh(test_references))
    print(f"ROUGE-L：微调前 {r_before['rougeL']:.4f} -> 微调后 {r_after['rougeL']:.4f}")
except Exception as e:
    print("ROUGE 计算跳过：", e)


# ============ 保存LoRA适配器（关键代码） ============
if not has_adapter:
    # 保存LoRA权重、adapter配置
    trainer.save_model(lora_save_path)
    # 等价：model.save_pretrained(lora_save_path)
    print(f"LoRA适配器已保存至：{lora_save_path}")
else:
    print(f"已训练权重来自：{lora_save_path}（无需重复保存）")