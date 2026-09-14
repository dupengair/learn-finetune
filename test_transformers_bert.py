from transformers import (
    # AutoModelForCausalLM, 
    AutoModelForSequenceClassification,  # BERT裸encoder，用于结构探测；做分类用AutoModelForSequenceClassification
    AutoTokenizer
)
import json, torch

# ===================== 加载模型 =====================
model_path = "./model/bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 强制本地加载
)

raw_inputs = "Give me a short introduction to large language model."
inputs = tokenizer(
    raw_inputs,
    padding=True,
    return_tensors="pt",
    truncation=True
)
print(inputs)
print(tokenizer.decode([
    101, 2507, 2033, 1037, 2460, 4955, 2000, 2312, 
    2653, 2944, 1012, 102
]))

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",  # 自动选择 GPU/CPU
    local_files_only=True
).half().cuda()  # 转为半精度 float16

print("模型加载成功！")
print(model)

# ===================== 输出 =====================
inputs_on_gpu = {    
    key: value.to("cuda") for key, value in inputs.items()
}

outputs = model(**inputs_on_gpu)
#print(outputs.last_hidden_state.shape)  # 输出最后一层的隐藏状态
print(outputs.logits.shape)  # 输出logits

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)  # 输出类别标签映射

# ===================== 保存 =====================
model.save_pretrained("./model/bert-base-uncased-finetuned")
