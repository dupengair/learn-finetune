from openai import OpenAI

base_url = "http://localhost:8000/v1"
client = OpenAI(api_key="EMPTY", base_url=base_url)

response = client.chat.completions.create(
        model = "qwen3-0.6b-lora",
        messages = [
            {"role":"sysstem","content":"假设你是一名粤菜大厨"},
            {"role":"user","content":"东莞有哪些美食"}
        ]
    )
print(response.choices[0].message)