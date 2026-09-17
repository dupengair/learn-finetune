# vLLM 部署报错分析与改造方案（test_vllm.py）

> 2026-09-17。运行 `test_vllm.py` 报 `AttributeError: 'Namespace' object has no attribute 'disable_fastapi_docs'`。
> 本文档只分析原因、指明改造方向，**不改动脚本**——改造留作手动练习（参考实现放第五节，先自己改再对照）。
> 版本事实基于本机 `ai-gpu` 环境实际安装的 **vLLM 0.10.2** 源码（`vllm/entrypoints/openai/`），所有行号均已核对。
> 硬件约束：RTX 4050 Laptop **6GB 显存** + WSL2，CPU 内存仅 7.76GB（日志可见 4GB 已划作 swap）。

---

## 一、现象与一句话结论

值得注意的时序：**引擎其实加载成功了**。日志里 `init engine ... took 19.47 seconds`、CUDA graph 捕获完毕、`Supported_tasks: ['generate']` 都已打印，说明 `LLM(...)` 本身没问题；崩溃发生在**之后的"服务组装"阶段**（`build_app`）。

**一句话结论**：vLLM 0.10.2 的服务端架构是"**从一份完整的命令行参数 Namespace 出发，由服务自己创建引擎**"，而不是"外部造好引擎塞给服务"。脚本手搓了一个只有 9 个字段的 `Namespace`，在 `build_app` 第一站就缺 `disable_fastapi_docs`；但就算补上这一个，后面还有两层更深的坑等着（见第二节）。正确修法是**删掉手工 `LLM(...)`，用官方 arg parser 生成完整 args，交给 `run_server`**。

---

## 二、逐层原因分析（三层坑，一层比一层深）

### 2.1 直接原因：`build_app` 的读取面远超 9 个字段

崩溃点 `api_server.py:1566`（`build_app` 第一行就访问缺失字段）：

```python
def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:      # ← 手搓 Namespace 里没有，当场 AttributeError
```

只补这一个字段没用。`build_app`（api_server.py:1565-1656）和 `init_app_state`（api_server.py:1660 起）沿路还要读：

| 来源 | 需要的 Namespace 字段 |
|---|---|
| `build_app` | `disable_fastapi_docs`、`root_path`、`allowed_origins`、`allow_credentials`、`allowed_methods`、`allowed_headers`、`api_key`、`enable_request_id_headers`、`middleware`、`enable_log_requests`、`max_log_len` |
| `init_app_state` | `served_model_name`、`model`、`disable_log_stats`、`chat_template`、`enable_log_requests`、`max_log_len` |

这就是"打地鼠"：每补一个字段，下一个 AttributeError 换个名字再来。脚本的 `llm_engine`、`response_role`、`lora_modules` 等 9 个字段恰好没有一个被这条路径用到。

### 2.2 第二层必炸点：`from_cli_args` 是"无默认值的硬 getattr"

就算把 `build_app` 需要的字段全补齐，服务路径更早的一步就会先崩——`build_async_engine_client`（api_server.py:145）拿到 args 后第一件事：

```python
engine_args = AsyncEngineArgs.from_cli_args(args)
```

而 `from_cli_args` 的实现（`vllm/engine/arg_utils.py:920-925`）是：

```python
@classmethod
def from_cli_args(cls, args: argparse.Namespace):
    attrs = [attr.name for attr in dataclasses.fields(cls)]
    engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})   # ← 没有 default！
    return engine_args
```

`getattr(args, attr)` **不带默认值**：`AsyncEngineArgs` 有一百多个 dataclass 字段，手搓 Namespace 缺任何一个（比如最核心的 `model`）都直接 `AttributeError`。所以**逐个手补字段的修法在机制上就走不通**——字段集由两个 dataclass + 前端参数共同决定，手搓永远赶不上读取面。

### 2.3 第三层（设计级）：`args.llm_engine` 是死代码，外挂引擎纯属白占显存

脚本的核心假设是"把 `llm.llm_engine` 塞进 Namespace 传给服务"。**0.10.2 的完整服务链路里没有任何一行代码读 `args.llm_engine`**：

```text
run_server(args)                        # api_server.py:1934
 └─ setup_server(args)                  # 绑定 socket
 └─ build_async_engine_client(args)     # ★ 引擎在这里、由 args 现场创建（from_cli_args）
 └─ build_app(args)                     # 纯粹建 FastAPI 路由
 └─ init_app_state(engine_client, ..., app.state, args)   # ★ 引擎注入点：app.state，不是 args
 └─ serve_http(app, sock, ...)
```

引擎的"注入通道"已经从 args 迁移到了 `app.state.engine_client`（init_app_state:1682）。这种 `Namespace(args.llm_engine=...)` 的写法是 vLLM 老版本（0.4~0.6 时代 `build_app`/`init_app_state` 还接受外部引擎参数时期）的遗留 hack。

**实测后果**（看日志算账）：脚本开头的同步 `LLM(...)` 已经完整初始化并占住了显存：

```text
model weights 1.12GiB + PyTorch activation peak 1.39GiB + CUDAGraph 0.23GiB + KV Cache 1.78GiB ≈ 4.5GiB
```

6GB 卡只剩 ~0.9GB。就算 2.1/2.2 两层都侥幸补全，`build_async_engine_client` 还会**再建第二个引擎**：权重 1.12GiB 都放不下，OOM 是必然结局。（若 Namespace 连 `model` 都没有，引擎会按默认值去找 `facebook/opt-125m` 联网下载——离线环境同样直接失败。）

> 学习要点：**报错在 `build_app` ≠ 引擎有问题**。第一层崩溃反而救了场——否则问题会以 OOM 的形式出现在更难定位的位置。

---

## 三、方向 A：逐步改造指引（推荐练习）

### 3.1 改造总览

原脚本 5 个组成部分，改完只剩 3 个：

```text
原脚本                              新脚本
─────────────────────────────      ─────────────────────────────
① os.environ VLLM_USE_V1=0    →    ① 原样保留（位置不变！）
② import uvicorn/Namespace/LLM →   ② 换成 4 个新导入（uvloop / parser / 校验 / run_server）
③ LLM(...) 造引擎        [删]  →
④ 手搓 Namespace(...)    [删]  →   ③ make_arg_parser + parse_args([...]) 造完整 args
⑤ build_app + uvicorn.run  [换] →   ④ validate_parsed_serve_args + uvloop.run(run_server(args))
```

### 3.2 逐步对照（原行号 → 怎么改）

**Step 0：环境变量原样保留（test_vllm.py:1-2）**

```python
import os
os.environ["VLLM_USE_V1"] = "0"   # 必须在 import vllm 之前（envs.py:95 默认 V1，import 时固化）
```

一行都不能动、位置也不能挪。0.10.2 默认走 V1 引擎，脚本锁 V0 是实测过的配置。

**Step 1：删掉外挂引擎（原 L6、L10-15）**

```python
# 【删】from vllm import LLM
# 【删】llm = LLM(
#     model="./model/Qwen3-0.6B",
#     max_model_len=4096,
#     gpu_memory_utilization=0.7,
#     served_model_name="qwen3-0.6b",
# )
```

为什么删：① 2.3 已证 `args.llm_engine` 是死代码；② 它白占 ~4.5GB，服务自建第二个引擎必 OOM。**但里面的参数意图不要丢**，Step 3 要逐项平移。

**Step 2：换导入（原 L4-5、L7）**

```python
# 【删】import uvicorn                      # run_server 内部自己起 HTTP 服务，不再需要
# 【删】from argparse import Namespace      # args 由 parser 产出，不再手搓
# 【改】from vllm.entrypoints.openai import api_server  →  按需具体导入：

import uvloop                                                    # serve.py:50 官方同款事件循环
from vllm.utils import FlexibleArgumentParser                    # cli_args.py:28 同款导入
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import run_server
```

**Step 3：手搓 Namespace → 官方 parser（原 L17-27 整块替换）**

```python
    # 官方 serve 子命令同款组装（serve.py:64）：make_arg_parser 接收 parser 并注册
    # 全部 前端参数(FrontendArgs) + 引擎参数(AsyncEngineArgs)，一个不漏
    parser = make_arg_parser(FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."))

    # 原 LLM(...) 的每个参数逐项平移为 CLI 风格列表；数值仍是字符串，
    # argparse 会按注册类型自动转换（与命令行行为一致）
    args = parser.parse_args([
        "--model", "./model/Qwen3-0.6B",
        "--served-model-name", "qwen3-0.6b",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.7",
        "--host", "0.0.0.0",          # ← 原 uvicorn.run(host=...) 平移过来
        "--port", "8000",             # ← 原 uvicorn.run(port=...) 平移过来
        "--disable-log-stats",        # ← 原 Namespace 漏传了；布尔开关不带值，单独一个元素
    ])
```

原脚本 → 新参数对照表：

| 原 LLM(...) / uvicorn 参数 | 新 parse_args 列表元素 | 说明 |
|---|---|---|
| `model="./model/Qwen3-0.6B"` | `"--model", "./model/Qwen3-0.6B"` | |
| `served_model_name="qwen3-0.6b"` | `"--served-model-name", "qwen3-0.6b"` | 请求体里 `"model"` 字段用的名字 |
| `max_model_len=4096` | `"--max-model-len", "4096"` | **6GB 必守**，见 3.4-① |
| `gpu_memory_utilization=0.7` | `"--gpu-memory-utilization", "0.7"` | |
| `uvicorn.run(host=..., port=...)` | `"--host", "0.0.0.0"`, `"--port", "8000"` | 由 `setup_server` 接管绑定 |
| （无） | `"--disable-log-stats"` | 关吞吐统计日志，测试时少刷屏 |

> 原 Namespace 里的 `response_role`、`return_tokens_as_token_ids`、`chat_template` 等字段**不用再操心**——parser 全都注册过且有默认值；哪天真需要，用官方名字（`--response-role`、`--return-tokens-as-token-ids`、`--chat-template`）往列表里加即可。这正是"完整 parser"相对"手搓 Namespace"的意义。

**Step 4：启动调用替换（原 L28-29 整块替换）**

```python
    # 【删】app = api_server.build_app(args)
    # 【删】uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    validate_parsed_serve_args(args)      # serve.py:53 官方启动前校验（chat template 等）
    uvloop.run(run_server(args))          # 内部完成：建引擎 → build_app → init_app_state → serve_http
```

不自己调 `build_app` 的原因：第二节链路图里 `run_server` 会按正确顺序把引擎塞进 `app.state`——自己拼这四步容易漏 `setup_server`/`serve_http` 的 socket 细节，官方函数一行全包。

### 3.3 微调模型适配（同一脚本的两种改法）

**变体一：LoRA 适配器热挂载（推荐——一个进程同时测 base 和微调后）**

vLLM 原生支持 LoRA 服务：适配器作为**独立可寻址的模型名**挂在同一个服务里。适配器实测 `r=8, alpha=16, target=q/k/v/o/gate/up/down`（`adapter_config.json`），全部是 vLLM 支持的注入模块。只需在 3.2 Step 3 的 `parse_args` 列表**末尾追加 4 个元素**，其余步骤完全不变：

```python
    args = parser.parse_args([
        "--model", "./model/Qwen3-0.6B",           # 基座不变
        "--served-model-name", "qwen3-0.6b",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.7",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--disable-log-stats",
        # ─────── 以下为 LoRA 追加项 ───────
        "--enable-lora",                            # 布尔开关，不带值（arg_utils.py:800）
        "--max-loras", "1",                         # 同时驻留的适配器数，1 够用，省显存
        "--max-lora-rank", "8",                     # 必须 ≥ 适配器的 r（这里正好 8）
        "--lora-modules",
        "qwen3-0.6b-lora=./training/qwen3-0.6b_Lora/lora_adapter",
        # ── 格式：name=path（LoRAParserAction 老格式，cli_args.py:46-48；
        #    也可传 JSON：'{"name": "...", "path": "..."}'）──
    ])
```

之后同一个服务里：

```bash
# 基线回答
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "介绍一下LoRA"}],
       "temperature": 0, "max_tokens": 256,
       "chat_template_kwargs": {"enable_thinking": false}}'

# 微调后回答 —— 只换 model 字段
#   "model": "qwen3-0.6b-lora"
```

6GB 友好性：适配器只有几 MB，vLLM 为 LoRA 额外预留的元张量显存与 `max_loras × max_lora_rank` 成正比——**同进程对比是最干净的方案**（同一 KV cache 预算、同一服务配置）。

**变体二：SFT 全量权重单独起服务**

`training/qwen3-0.6b_SFT/lora_adapter/` 名不副实（坑 13）：里面是**完整权重** `model.safetensors` + config + tokenizer + `chat_template.jinja`，vLLM 按普通模型目录加载。只改 Step 3 列表的前两个元素：

```python
    args = parser.parse_args([
        "--model", "./training/qwen3-0.6b_SFT/lora_adapter",   # 指向全量权重目录
        "--served-model-name", "qwen3-0.6b-sft",
        # ...其余元素与 3.2 完全相同，不要 --enable-lora 那四项
    ])
```

6GB 一次只能驻留一个引擎：先测 base（或变体一），Ctrl+C 释放，`nvidia-smi` 确认显存归零，再改参数起这个。

**不可用产物清单（别浪费时间）**

| 产物 | 为什么 vLLM 服务不了 |
|---|---|
| `qwen3-0.6b_QLora` 适配器 | 基座是 bnb 4bit 量化，vLLM 对"bnb 量化基座 + LoRA"组合支持受限；效果对比用 LoRA 线代替 |
| `qwen3-0.6b_AdaLora` 适配器 | 变秩 `rank_pattern` 随训练演化，vLLM LoRA 只支持定秩 |
| Prefix / Prompt / PTuningV2 | prompt 系是虚拟 token / 每层 KV 注入，vLLM 不支持 prompt learning，只支持 LoRA 类适配器 |

> 想看这几条线的生成效果，走既有 transformers 线脚本（`test_Prefix_qwen3-0.6b.py` 等的加载分支）。

### 3.4 6GB 显存约束下的参数选择

**必守（不动就崩）**

1. **`--max-model-len 4096` 必须显式给**。Qwen3 原生上下文上限是 40960，不传该参数 vLLM 会按模型上限做 KV cache 预算，6GB 直接 OOM 在引擎初始化。
2. **一次只驻留一个引擎**。变体一里 base + LoRA 同进程是安全的；绝不允许同时开两个服务进程（第二个连权重 1.12GB 都放不下，参考 2.3 的算账）。
3. **保留 `VLLM_USE_V1=0`**。V1 引擎默认开启（envs.py:95），走 torch.compile 与另一套显存画像，在 6GB WSL2 上未验证；V0 是本仓库实测过的形态。
4. **换服务前先确认释放**。Ctrl+C 后 `nvidia-smi` 查看显存归零再起下一个；WSL2 偶发进程残留占卡。

**推荐（测试体验）**

5. `--disable-log-stats` 关掉每轮吞吐统计刷屏（日志里 `disable_log_stats: True` 就是它）。
6. `--disable-frontend-multiprocessing`：V0 下 vLLM 可能用 MQLLMEngine 子进程跑引擎（api_server.py:238 起的决策逻辑），WSL2 上若遇子进程/gloo 问题，加这个强制进程内运行。
7. 请求里 `max_tokens` 别开太大：当前 KV cache 仅 1.66GB，4096 上下文下最大并发约 3.8 路，单条长生成没问题，多条并发会排队。

**应急三档（真 OOM 时逐档加）**

| 档位 | 追加/修改参数 | 代价 |
|---|---|---|
| 第一档 | `--gpu-memory-utilization` 0.7 → 0.6 | KV cache 变小，吞吐降 |
| 第二档 | `--enforce-eager` | 省 0.23GB CUDA graph，解码变慢 |
| 第三档 | `--max-model-len 2048` + `--max-num-seqs 4` | 上下文与并发上限都降 |

最后一档之外的隐性手段：关掉 WSL2 里其他占显存的进程（Windows 侧浏览器/桌面共享显存），`nvidia-smi` 看底噪。

### 3.5 方向 B：CLI 直接起服务（快速跑通实验用）

与方向 A 完全等价（`vllm serve` 就是它的官方封装）：

```bash
VLLM_USE_V1=0 vllm serve ./model/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --max-model-len 4096 --gpu-memory-utilization 0.7 \
  --host 0.0.0.0 --port 8000 --disable-log-stats
```

LoRA 变体追加 `--enable-lora --max-loras 1 --max-lora-rank 8 --lora-modules qwen3-0.6b-lora=./training/qwen3-0.6b_Lora/lora_adapter`。

### 3.6 方向 C：手动补全 Namespace（不推荐，但值得想明白）

给手搓 Namespace `setdefaults` 补字段。机制上已被 2.1 + 2.2 判死刑：前端参数 + 两个 dataclass 的**全部字段**都要补，且 `from_cli_args` 是硬 getattr 无兜底。如果做这个练习，目的应是"数一数服务端到底要读多少字段"，而不是当成修法。

---

## 四、对比问答的请求要点

1. **用 `/v1/chat/completions` 而不是 `/v1/completions`**：vLLM 会自动套用模型目录里的 chat template。base 目录和 SFT 目录都带 Qwen3 的 `chat_template.jinja`，与 SFT 训练时"手工拼 Qwen IM 模板"同源，**推理口径与训练口径一致**——这正是训练时"prompt 部分 mask、只对回答算 loss"所要求的格式。
2. **关掉 Qwen3 的思维链再对比**：Qwen3 模板默认 `enable_thinking=True`，回答前会先吐一段 `<think>...</think>`；SFT 数据（zhihu-kol）里并没有思维链，混进来会污染对比。请求体加 `"chat_template_kwargs": {"enable_thinking": false}`（vLLM 0.10.2 已支持，protocol.py:511）。
3. **统一解码参数保证可比**：`"temperature": 0`（等价于训练评估线的 `do_sample=False`）、固定 `max_tokens`（如 256）。

curl 模板见 3.3 变体一。

**建议的观察点**（生成侧主观对照，配合已有的 eval_loss）：

- **收尾能力**：SFT 线专门处理过"pad_token=eos=<|im_end|> 被抹成 -100 学不会收尾"的问题（坑 11），看微调后是否稳定以 `<|im_end|>` 收尾、不再无限续写；
- **复读与跑题**：base 模型对指令的服从度、循环复读次数；
- **风格贴合**：与 zhihu-kol 训练数据（INSTRUCTION/RESPONSE）风格的接近程度。

---

## 五、参考实现（先自己改，再对照）

**基线版**（对应 3.2 各步，与 `vllm serve` 官方组装方式一致）：

```python
import os
os.environ["VLLM_USE_V1"] = "0"   # 必须在 import vllm 之前（envs.py 默认 V1，import 时固化）

import uvloop
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import run_server


def main():
    # 官方 serve 子命令同款组装：注册全部前端参数 + 引擎参数
    parser = make_arg_parser(FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."))

    # 原 LLM(...) + uvicorn.run 的全部意图平移为 CLI 风格参数；引擎由 run_server 内部创建
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
```

**LoRA 对比版**：在基线版的 `parse_args` 列表末尾追加（3.3 变体一）：

```python
        "--enable-lora",
        "--max-loras", "1",
        "--max-lora-rank", "8",
        "--lora-modules",
        "qwen3-0.6b-lora=./training/qwen3-0.6b_Lora/lora_adapter",
```

**SFT 版**：把前两个元素换成（3.3 变体二，且不要 LoRA 四项）：

```python
        "--model", "./training/qwen3-0.6b_SFT/lora_adapter",
        "--served-model-name", "qwen3-0.6b-sft",
```

自测清单：

- [ ] 启动后日志出现 `Initializing a V0 LLM engine`（确认 VLLM_USE_V1=0 生效）；
- [ ] 只出现**一次**引擎初始化（确认外挂 LLM 已删、无双引擎）；
- [ ] `curl http://localhost:8000/v1/models`：基线版返回 `qwen3-0.6b`；LoRA 版返回 `qwen3-0.6b` + `qwen3-0.6b-lora` 两个；
- [ ] 用第四节的 curl 分别拿到 base 与微调侧回答，对比收尾/复读/风格；
- [ ] Ctrl+C 后 `nvidia-smi` 显存归零，能立刻起下一个变体。

---

## 六、踩坑归档（建议并入总坑集）

**坑 20：vLLM ≥0.10 的服务端必须由"完整 args"驱动，引擎注入通道已从 args 迁移到 app.state**。手搓 Namespace + 外挂 `LLM().llm_engine` 的老 hack 已死：① `build_app`/`init_app_state` 的前端字段读取面打地鼠打不完；② `AsyncEngineArgs.from_cli_args` 是无默认值硬 getattr（arg_utils.py:924），缺一字段即崩；③ `args.llm_engine` 无人读取，外挂引擎只白占显存、挤爆 6GB 卡。正解：`make_arg_parser().parse_args([...])` → `run_server(args)`（等价于 `vllm serve`），或直接用 CLI。附带信号：**报 `disable_fastapi_docs` 的 AttributeError 发生在引擎初始化成功之后，说明模型与引擎没问题，是服务组装方式错了**。

**坑 21：6GB 卡上 vLLM 服务的硬约束**——`--max-model-len` 必须显式压小（Qwen3 原生 40960 的 KV cache 预算在 6GB 上直接 OOM）；一次只驻留一个引擎，换服务先 `nvidia-smi` 确认释放；LoRA 微调成果用 `--enable-lora --lora-modules name=path` 同进程热挂载对比最省显存；QLoRA/AdaLora/prompt 系产物不可服务，生成侧对比走 transformers 线脚本。
