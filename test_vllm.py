import os
os.environ["VLLM_USE_V1"] = "0"   # 必须在 import vllm 之前（envs.py 默认 V1，import 时固化）

import uvloop
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import run_server

def main():
    # 官方 serve 子命令同款组装：make_arg_parser 接收 parser 并注册全部前端+引擎参数
    parser = make_arg_parser(FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."))

    # 原脚本 LLM(...) 的全部意图平移为 CLI 风格参数；引擎由 run_server 内部创建
    args = parser.parse_args([
        "--model", "./model/Qwen3-0.6B",
        "--served-model-name", "qwen3-0.6b",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.7",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--disable-log-stats",
    ])
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))   # serve.py:50 官方同款；asyncio.run 亦可

if __name__ == "__main__":
    main()