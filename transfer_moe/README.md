# Transfer MoE：方法与剪枝比例统计

当前 `qwen3_5_moe` 适配 Qwen3.5 / Qwen3.6 MoE，已实现 [Fixed-K](skip_fixedk/README.md)、[DynRoute](skip_dynroute/README.md)、[TransferMoE](skip_transfer/README.md) 的 HF / vLLM 接入。DIEP、BAN 暂时只有原理说明。

所有 patch 通过环境变量统一控制是否使用 [utils.py](utils.py) 中的 `ActivationStats`。开启后，每个模型拥有一个 `model.activation_stats`，各层共享这个对象，每层分别计数。patch 应在加载权重、确定设备之后，推理和图捕获之前安装。

## 统计开关

默认关闭。需要测量剪枝比例时，在 HF 脚本或 vLLM 服务启动前设置：

```bash
export TRANSFER_MOE_RECORD_STATS=1
```

关闭使用 `export TRANSFER_MOE_RECORD_STATS=0`，或 `unset TRANSFER_MOE_RECORD_STATS`。只有值为 `1` 时记录。

关闭时不分配计数器、不执行统计用的 `sum` / 累加、不更新 token 阶段标记，也不安装 HF 统计 hook 或 vLLM 输入准备 hook。Fixed-K 的 vLLM 直接调用原生专家函数。此时 `model.activation_stats` 为 `None`，`get_activation_stats` RPC 返回 `null`，清零 RPC 不做操作。

环境变量在安装 patch 时读取，不在每层 forward 中反复读取。HF eager 模式可以修改当前 Python 进程的环境变量后重新 patch，旧统计 hook 会移除；重新开启从零计数。vLLM 必须在服务启动前设置，切换时重启服务，使 CUDA Graph 按新的开关重新捕获。关闭统计时，图中不包含统计操作。测时延时使用关闭模式，统计剪枝比例时使用开启模式。

## 返回内容与统计口径

```json
{
  "layer_pruning_ratios": {"0": 0.5, "1": 0.25},
  "overall_pruning_ratio": 0.375,
  "prefill": {
    "layer_pruning_ratios": {"0": 0.6, "1": 0.3},
    "overall_pruning_ratio": 0.45
  },
  "decode": {
    "layer_pruning_ratios": {"0": 0.4, "1": 0.2},
    "overall_pruning_ratio": 0.3
  }
}
```

上述数值为示例，`0.5` 表示剪掉 50%。层编号从 0 开始。

- **每层平均剪枝比例**：该层累计跳过的专家调用数 / 该层原生 Top-K 专家调用数。
- **总体平均剪枝比例**：所有层累计跳过的专家调用数 / 所有层原生 Top-K 专家调用数。先累加分子、分母，再相除，不对不同大小 batch 的比例直接取平均。
- 顶层两个字段保留 prefill + decode 的合计；`prefill`、`decode` 分别返回各阶段的每层和总体比例。合计仍按专家调用数加权，不能直接平均两个阶段的比例。
- 原生 K 指剪枝前每个 token 的 routed expert 数量。shared expert 不计入；Transfer 的补偿输出不增加实际激活数量。
- HF 面向常规带 KV cache 的 `generate`：进入 forward 时 cache 为空记作 prefill，已有历史 token 记作 decode。根据传入的二维 `attention_mask` 排除 padding，decode 只取本次输入对应的 mask；未提供 mask 时统计本次所有输入位置。
- vLLM 按每个 token 的实际序列位置区分：位置小于该请求的 prompt 长度属于 prefill，否则属于 decode。同一 batch 可同时累计两个阶段，支持 chunked prefill、请求重排和 prefix cache；只剩一个 token 的 prefill 仍记为 prefill。
- vLLM 排除 CUDA Graph 补齐的行，启动预热完成后自动清零。命中 KV cache 而未执行的 token 不计数；统计的是实际 MoE 前向计算次数，而非最终生成的 token 数。
- 尚无样本的层、阶段和总体返回 `null`；完整保留且已经执行过推理时返回 `0.0`。

计数在各层所在设备上完成，不在每层调用 `.item()` / `.tolist()`；只有 `report()` 读取设备计数。`reset()` 原地清零，CUDA Graph 捕获的计数器地址保持不变。

## HF：读取与清零

使用 `python -m transfer_moe.infer` 时，设置 `TRANSFER_MOE_RECORD_STATS=1` 即可在生成文本之后自动打印统计 JSON，包含本次推理的整体、prefill 和 decode 剪枝比例。

开启统计并安装 patch 后，在自己的推理代码中直接使用：

```python
model.activation_stats.reset()
# 运行本轮 model(...) 或 model.generate(...)
report = model.activation_stats.report()
print(report["prefill"]["overall_pruning_ratio"])
print(report["decode"]["overall_pruning_ratio"])
```

以下脚本在仓库根目录执行，完整展示加载、推理和保存统计：

```bash
export TRANSFER_MOE_RECORD_STATS=1
export MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"
export METHOD="skip_dynroute"
export PATCH_KWARGS='{"topp_threshold": 0.8}'

python - <<'PY'
from importlib import import_module
import json
import os
from pathlib import Path

import torch
from transfer_moe.utils import load_model_and_processor

model, processor = load_model_and_processor(os.environ["MODEL"], "cuda:0", "bfloat16")
patch = import_module(
    f"transfer_moe.{os.environ['METHOD']}.qwen3_5_moe.patch_hf"
).patch
patch(model, **json.loads(os.environ["PATCH_KWARGS"]))
inputs = processor(text="请解释混合专家模型。", return_tensors="pt")
inputs = inputs.to(model.get_input_embeddings().weight.device)

model.activation_stats.reset()
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=128, do_sample=False)

text = json.dumps(model.activation_stats.report(), ensure_ascii=False, indent=2)
print(text)
Path("activation_stats.json").write_text(text + "\n", encoding="utf-8")
PY
```

Fixed-K 使用 `METHOD=skip_fixedk`、`PATCH_KWARGS='{"topk":4}'`。Transfer 使用 `METHOD=skip_transfer`，并在 `PATCH_KWARGS` 中同时指定 `calibration_path` 与 `topk` 或 `topp_threshold`。HF 多卡可将加载时的 `"cuda:0"` 改为 `"auto"`。

## vLLM 服务：EvalScope 前清零、完成后读取

三个自定义 worker 都继承公共 `ActivationStatsWorkerMixin`，提供 `reset_activation_stats()` 和 `get_activation_stats()`。后者返回与 HF 完全相同的 JSON，包含整体、prefill、decode。当前统计适配 **vLLM 0.24.0 的 V1 runner、单卡 / TP**；使用各方法文档中的未量化 Triton 路径，不启用 DBO、PP、EP、DP、SP 或 PCP。

worker 在原生 `_prepare_inputs` 完成后，使用最终的请求索引、GPU 序列位置和 prompt 长度更新阶段标记。标记存放在地址固定的设备缓冲区中，CUDA Graph 每次重放读取当前值；图补齐的行标记为 `-1`，不参与统计。统计 hook 不改动原生 MoE 的 dispatch / GEMM，也不关闭 CUDA Graph。[原生输入准备与补齐实现](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/worker/gpu_model_runner.py)

在受控评测实例上，启动时设置 `VLLM_SERVER_DEV_MODE=1`，通过现成的 `/collective_rpc` 接口访问 worker。[vLLM 开发接口说明](https://docs.vllm.ai/en/v0.24.0/serving/online_serving/#server-in-development-mode)

例如，启动 DynRoute 服务：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export VLLM_SERVER_DEV_MODE=1
export TRANSFER_MOE_RECORD_STATS=1
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"

vllm serve "$MODEL" \
  --served-model-name dynroute \
  --dtype bfloat16 \
  --moe-backend triton \
  --worker-cls transfer_moe.skip_dynroute.qwen3_5_moe.patch_vllm.DynRouteWorker \
  --additional-config '{"skip_dynroute":{"topp_threshold":0.8}}' \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000
```

Transfer 使用其原有 `TransferMoEWorker` 和校准参数即可。Fixed-K 仍用 `--hf-overrides` 设置 k；需要读取统计时，再添加只负责计数的 `FixedKWorker`，并传入**覆盖前的原生 K**。例如原生 K=8、保留 k=4：

```bash
vllm serve "$MODEL" \
  --served-model-name fixedk \
  --hf-overrides '{"text_config":{"num_experts_per_tok":4}}' \
  --dtype bfloat16 \
  --moe-backend triton \
  --worker-cls transfer_moe.skip_fixedk.qwen3_5_moe.patch_vllm.FixedKWorker \
  --additional-config '{"skip_fixedk":{"native_topk":8}}' \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000
```

`native_topk` 必须来自原始 checkpoint 配置，不能填覆盖后的 k。DynRoute 和 Transfer 自动从未覆盖的配置取得原生 K。

服务就绪后，在运行 EvalScope 的终端中先清零：

```bash
export VLLM_URL="http://127.0.0.1:8000"
curl -fsS "$VLLM_URL/collective_rpc" \
  -H 'Content-Type: application/json' \
  -d '{"method":"reset_activation_stats"}'
```

接着运行原有 EvalScope 命令。等本轮所有请求结束后，执行以下读取脚本；结果打印到当前终端，并保存到 `activation_stats.json`：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

request = Request(
    os.environ["VLLM_URL"].rstrip("/") + "/collective_rpc",
    data=json.dumps({"method": "get_activation_stats"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    workers = json.load(response)["results"]
# 单卡只有一个结果；当前 TP 各 rank 的统计重复，取其中一个。
report = workers[0]
text = json.dumps(report, ensure_ascii=False, indent=2)
print(text)
Path("activation_stats.json").write_text(text + "\n", encoding="utf-8")
PY
```

整个统计窗口内，该实例只运行这一轮评测。下一轮再清零，服务无需重启。EvalScope 的请求与原有报告生成逻辑不用修改，剪枝统计单独保存在上述 JSON 中。

## 验证

```bash
python -m unittest transfer_moe.tests.test_activation_stats \
  transfer_moe.tests.test_dynroute transfer_moe.tests.test_transfer \
  transfer_moe.tests.test_vllm -v
```

CPU 测试覆盖开关切换与输出一致性、关闭时不创建计数器及移除 hook、各阶段计数和总量加权、padding、带 KV cache 的生成、重复 patch，以及 worker 输入元数据的混合阶段、短 prefill、分块输入、缓存命中和请求重排。CUDA 测试覆盖图重放中阶段标记与真实 token 数变化；vLLM 测试使用实际专家 kernel / runner。本地无 CUDA 时，GPU 项明确跳过，仍需在目标服务器执行。
