from transformers import ( 
    AutoModelForCausalLM,  # BERT裸encoder，用于结构探测；做分类用AutoModelForSequenceClassification
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch

# ===================== 加载模型 =====================
model_path = "./model/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)

# 4bit NF4量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",  # 自动选择 GPU/CPU
    local_files_only=True,
    quantization_config=bnb_config  # 量化为4bit
) 

model = model.eval()  # 设置为评估模式


# ========= Qwen3 标准对话流程 =========
messages = [
    {"role": "user", "content": "晚上睡不着怎么办?"}
]

# 使用tokenizer应用对话模板，构建prompt
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True   # Qwen3思考模式开关
)

inputs = tokenizer(
    text, 
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8
    )

# 只取出新生成的部分，跳过输入prompt
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("模型回答：", response)
