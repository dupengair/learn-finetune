import os
os.environ["VLLM_USE_V1"] = "0"

import uvicorn
from argparse import Namespace
from vllm import LLM
from vllm.entrypoints.openai import api_server

def main():
    llm = LLM(
        model="./model/Qwen3-0.6B",
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        served_model_name="qwen3-0.6b",
    )

    # 0.10.2 build_app入参是Namespace，没有llm_engine参数
    args = Namespace(
        llm_engine=llm.llm_engine,
        served_model_name="qwen3-0.6b",
        response_role="assistant",
        lora_modules=None,
        prompt_adapters=None,
        chat_template=None,
        return_tokens_as_token_ids=False,
        enable_tool_call_parser=False,
    )
    app = api_server.build_app(args)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()