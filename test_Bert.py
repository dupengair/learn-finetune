from transformers import (
    # AutoModelForCausalLM, 
    AutoModel,  # BERT裸encoder，用于结构探测；做分类用AutoModelForSequenceClassification
    AutoTokenizer, 
    AutoConfig
)
import json

'''
与QWen3 差异：
1. 模型类换成 `AutoModelForSequenceClassification`（分类）/ `AutoModel`（裸特征提取，推荐用来做模型结构探测）；**不能用 AutoModelForCausalLM**，BERT 没有 decoder 生成头。
2. BERT**没有 apply_chat_template、没有 generate、没有 thinking 标签`151668`**，这部分全部删除。
3. BERT 不使用 RoPE、GQA
'''

# ===================== 加载模型 =====================
# model_name = "./model/Qwen3-0.6B"
model_path = "./model/bert-base-uncased" 

# load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)
# model = AutoModelForCausalLM.from_pretrained(
# AutoModel：只输出hidden_states，无任务头；如果做分类任务：AutoModelForSequenceClassification
model = AutoModel.from_pretrained(
#    model_name,
    model_path,
    torch_dtype="auto",
    device_map="auto",  # 自动选择 GPU/CPU
    local_files_only=True
)

print("模型加载成功！")


# ===================== 测试模型 =====================
# prepare the model input
'''
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
'''

# 测试BERT前向推理（BERT不是生成模型，没有generate
# BERT输入普通句子，不使用chat_template
prompt = "Give me a short introduction to large language model."
model_inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=128
).to(model.device)


'''
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
'''
# 前向传播，获取encoder输出
outputs = model(**model_inputs)
# pooler_output：BERT的   输出向量，常用做分类
pooler_output = outputs.pooler_output
last_hidden_state = outputs.last_hidden_state

print(f"\nlast_hidden_state shape: {last_hidden_state.shape}")   # [batch, seq_len, hidden_size]
print(f"pooler_output shape: {pooler_output.shape}")             # [batch, hidden_size]


# ===================== 分析模型 =====================
print("Model Configuration:")
print(model.config)
print("\nModel Summary:") 
print(model)
print("\nFirst Layer:") 
# print(model.model.layers[0])
# BERT层访问路径： model.bert.encoder.layer ，不是 model.model.layers（Qwen的路径）
print(model.encoder.layer[0])

# 1. 加载配置文件
config = AutoConfig.from_pretrained(model_path)
config_dict = config.to_dict()

# 2. 打印并解读核心参数
print("--- bert‑base‑uncased 核心配置解读 ---")
config_dict = config.to_dict()
core_params = {
    "模型类型 (architectures)": config_dict.get("architectures"),
    "隐藏层维度 (hidden_size)": config_dict.get("hidden_size"),
    "中间层维度 (intermediate_size)": config_dict.get("intermediate_size"),
    "总头数 (num_attention_heads)": config_dict.get("num_attention_heads"),
    # "KV头数 (num_key_value_heads)": config_dict.get("num_key_value_heads"),
    "隐藏层(encoder)数量 (num_hidden_layers)": config_dict.get("num_hidden_layers"),
    "RMSNorm Epsilon (rms_norm_eps)": config_dict.get("rms_norm_eps"),
    # "RoPE Theta (rope_theta)": config_dict.get("rope_theta"),
    "词表大小 (vocab_size)": config_dict.get("vocab_size"),
    "max_position_embeddings": config_dict.get("max_position_embeddings"),
}
print(json.dumps(core_params, indent=2, ensure_ascii=False))

# --- 理论与现实的映射 ---
'''
num_q_heads = core_params["总头数 (num_attention_heads)"]
num_kv_heads = core_params["KV头数 (num_key_value_heads)"]
if num_q_heads != num_kv_heads:
    print(f"\n[分析]: Q头数({num_q_heads}) != KV头数({num_kv_heads}) -> 确认使用 GQA！")
    print(f"       GQA 分组大小: {num_q_heads // num_kv_heads}")
'''

# ff_ratio = core_params["中间层维度 (intermediate_size/d_ff)"] / core_params["隐藏层维度 (hidden_size/d_model)"]
ff_ratio = core_params["中间层维度 (intermediate_size)"] / core_params["隐藏层维度 (hidden_size)"]
print(f"[分析]: FFN 扩展比例 (d_ff / d_model): {ff_ratio:.2f}")

