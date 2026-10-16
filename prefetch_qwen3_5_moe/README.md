# Qwen MoE 的张量路由缓存

RouterCache 按层保存整块张量。训练和推理使用同一种存储方式：每层每次 forward 写入一个 token 区间，查询时按绝对 token 位置取切片。

## 存储结构

```python
cache.cache_gt[layer_id]   # 这一层的真实路由
cache.cache_pre[layer_id]  # 对这一层的预测路由
```

两个列表的长度都是模型层数。某层第一次写入时，在对应设备上分配一个 RouterResult：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `router_logits` | `[batch, max_tokens, num_experts]` | 完整 logits，训练用 |
| `router_scores` | `[batch, max_tokens, top_k]` | 选中专家的归一化权重 |
| `router_indices` | `[batch, max_tokens, top_k]` | 专家编号 |
| `valid_mask` | `[batch, max_tokens]` | 已保存结果的源 token 是否有效 |
| `token_start`、`token_end` | 两个整数 | 已写入的绝对位置区间 `[start, end)` |

层记录为 `None` 表示该层尚未保存结果。张量中未写入的位置，valid_mask 为 False、专家编号为 -1。padding 位置也由 valid_mask 标记。

例如直接查看第 5 层、token 7 的预测：

```python
layer = cache.cache_pre[5]
logits = layer.router_logits[:, 7, :]  # [batch, experts]
```

判断该位置是否已写入，可以使用查询方法：

```python
prediction = cache.get_prefetch(token_idx=7, layer_id=5)
if prediction is not None:
    expert_ids = prediction.router_indices  # [batch, top_k]
```

## 每层一次保存整段

假设当前是第 3 层，一次处理 8 个 token，提前两层预测。gate_prefetch 的输出整理为 `[batch, 8, experts]` 后，整段传入：

```python
cache.save_prefetch(
    token_start=0,
    layer_id=5,
    router_logits=prefetch_logits,
    router_scores=prefetch_scores,
    router_indices=prefetch_indices,
    valid_mask=attention_mask,
)
```

保存方法中的核心操作是：

```python
layer = self.cache_pre[layer_id]
token_end = token_start + router_logits.shape[1]
layer.router_logits[:, token_start:token_end] = router_logits
```

scores、indices 和 valid_mask 写入相同区间。`save_gt()` 则将整段真实路由写入 `cache_gt[layer_id]`，真实 logits 在保存时 detach。预测 logits 的切片赋值由 PyTorch 记录到计算图，损失可以反传到 gate_prefetch。

每层按连续位置追加写入；下一次写入的起点等于该层记录的 token_end。

## mode_prefetch 决定目标层和写入区间

`mode_prefetch = (layers_ahead, tokens_ahead)`：

| mode_prefetch | 预测目标 |
| --- | --- |
| `(1, 0)` | 同一个 token，下一层 |
| `(2, 0)` | 同一个 token，后两层 |
| `(0, 1)` | 下一个 token，同一层 |
| `(0, 2)` | 后两个 token，同一层 |
| `(1, 1)` | 下一个 token，下一层 |

在 [patch.py](patch.py) 的 `moe_forward()` 中，目标位置直接计算为：

```python
target_token_start = cache.token_start + tokens_ahead
target_layer_id = self.layer_id + layers_ahead
```

例如 `(0, 1)` 下，处理源位置 `[0, 8)`，预测写入目标位置 `[1, 9)`。位置 8 的预测会保留到第一次 decode。

当前层通过 `get_prefetch_range()` 一次读取本次输入对应的整段预测。结果保存在：

```python
block.prefetch_indices     # [batch, 本次输入长度, top_k]
block.prefetch_valid_mask  # [batch, 本次输入长度]
```

没有该层预测记录时，两者为 None；已有记录时，使用 valid_mask 判断具体位置。单 token 查询仍使用 `get_prefetch(token_idx, layer_id)`。

## 容量和本次输入位置

训练直接调用模型：

```python
model(**inputs, use_cache=False, logits_to_keep=1, output_router_logits=False)
```

`use_cache=False` 时，forward wrapper 为本轮创建新缓存，训练容量为 `sequence_length + tokens_ahead`，并从输入中准备位置和 mask。每层在第一次保存时按这个容量分配张量。一次 forward 后计算 loss、完成 backward，再处理下一批。

使用 KV cache 生成时，请求开始执行一次 `cache.clear()`，使用 [router_cache.py](router_cache.py) 中的默认容量 1024。这个容量需要覆盖：

```python
max_tokens >= prompt_length + max_new_tokens + tokens_ahead
```

假设 prompt 长度为 8：

| 本次输入 | cache.token_start | 写入真实路由的区间 |
| --- | --- | --- |
| prompt | 0 | `[:, 0:8]` |
| 第一次 decode | 8 | `[:, 8:9]` |
| 第二次 decode | 9 | `[:, 9:10]` |

长序列生成需调大 `RouterCache.clear(max_tokens=1024)` 的默认容量；复用已有 KV 前缀时，容量也需要覆盖该前缀占用的绝对位置。超出容量时缓存会报错。decode 在已分配张量中写入后续位置，整层缓存的存储地址保持稳定。

`token_start` 是输入序列中的列位置，包含 padding 占用的列。forward wrapper 负责准备好 `cache.attention_mask`，形状为 `[batch, 本次输入长度]`、dtype 为 bool。各层 MoE 直接使用它，写入缓存时由张量赋值复制到该层设备。

`model.generate()` 的 attention_mask 按 Transformers 原生规则准备，包括省略时根据输入推断 padding。forward wrapper 从中取出本轮输入对应的列；如果 forward 收到的 mask 为 None，则在模型入口准备全 True mask。

## 使用原生 generate

`patch()` 和 `load_prefetch()` 会自动安装 wrapper。推理直接调用模型方法：

```python
cache = load_prefetch(model, "outputs/qwen_prefetch.safetensors")
output_ids = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,
    num_beams=1,
    output_router_logits=False,
)
prediction = cache.get_prefetch(token_idx=7, layer_id=5)
```

generate wrapper 的主体：

```python
cache.clear()
return original_generate(*args, **kwargs)
```

训练和生成的调用顺序：

1. 生成请求通过 generate wrapper 清空旧路由，然后执行原生 generate。
2. 每次调用模型时，forward wrapper 从 KV cache 长度读取本轮 token_start，准备本轮 mask，再执行原生 forward。`use_cache=False` 时先为本轮准备新缓存。
3. 每层 moe_forward 根据这些位置批量保存真实路由和预测路由。

采样、EOS、stopping_criteria、logits_processor、streamer 及返回值格式由原生 generate 处理。设置 `return_dict_in_generate=True` 时，返回值仍是 Transformers 的生成结果对象。

缓存按固定 batch 行顺序保存，适用于原生 greedy 和 sampling（num_beams=1），包括采样时的 num_return_sequences。请使用这些生成模式；beam 和 assisted 所需的缓存重排或回退尚未实现。使用默认 DynamicCache 和二维 padding mask；静态/编译缓存尚未验证。

`use_cache=False` 时，模型每轮重新计算整段输入，RouterCache 也重算整段路由，结束后保留最后一轮结果。训练和生成都使用这个分支。

生成返回后保留路由供查询。异常会原样抛出，已写入的部分结果也会保留；下一次 generate 开始时统一清空。

## 模型怎样调用保存方法

`patch(model, mode_prefetch)` 创建共享缓存和预测头，并将每层 MoE 的 forward 绑定到 [patch.py](patch.py) 的 `moe_forward()`。

`gate_prefetch` 使用模型原有的 `Qwen3_5MoeTopKRouter` 类。每个有效源层新建一个独立 Router 实例，以目标层真实 gate 的权重初始化，并使用 FP32 参数。调用直接返回三个结果：

```python
prefetch_logits, prefetch_scores, prefetch_indices = self.gate_prefetch(hidden.detach().float())
```

`patch.py` 从目标 gate 获取 `router_class`，再用该层配置创建实例。真实 gate 的参数保持冻结，预测 Router 的参数参与训练。

因此调用 `model(...)` 时，每层依次执行：

1. gate_prefetch 计算整段预测，调用一次 `cache.save_prefetch()`。
2. 当前层调用一次 `cache.get_prefetch_range()` 读取预测。
3. 真实 gate 计算整段路由，调用一次 `cache.save_gt()`。
4. 使用真实路由执行专家。

完整 forward 后，训练通过 `cache.loss(router_mask=batch["router_mask"])` 计算 assistant 文本位置的损失。

训练和推理调用模型时设置 `output_router_logits=False`，两类路由分别从 RouterCache 读取。模型原有的 OutputRecorder 按 Router 类型收集输出；启用它会同时采集真实 gate 和 gate_prefetch。

## 损失按层计算

`cache.matched()` 每次返回一层的重合区间：`(layer_id, prediction, teacher)`。

例如预测覆盖 `[1, 9)`，真实结果覆盖 `[0, 8)`，两者在 `[1, 8)` 配对：

```python
prediction_logits = cache.cache_pre[layer_id].router_logits[:, 1:8]
teacher_logits = cache.cache_gt[layer_id].router_logits[:, 1:8]
```

`cache.loss()` 对该层的整个有效区间计算 KL(真实分布 || 预测分布)，最后对所有有效的 batch/token/layer 位置取平均。源和目标都有效时才计入损失。传入 router_mask 时，还会根据绝对目标位置筛选 assistant 文本 token；提前 token 预测的源位置可以来自 prompt。

训练顺序为：

```python
optimizer.zero_grad(set_to_none=True)
# inputs 包含完整文本和图像输入；labels、router_mask 由训练入口单独处理。
model(**inputs, use_cache=False, logits_to_keep=1, output_router_logits=False)
loss = cache.loss(router_mask=batch["router_mask"])
loss.backward()
optimizer.step()
cache.clear()
```

主模型已经冻结，训练期间保持 autograd 开启。所有写入完成后计算 loss，再 backward；下一批开始时使用新的缓存张量。

## 平均命中专家数量

`cache.prefetch_hit_stats()` 根据预测的完整 logits 取前 k′ 个专家，与真实 Router 选出的 top-k 专家求交集，再按有效的 token-layer 位置平均：

```text
avg_hits[k′] = 所有有效位置的命中专家数量之和 / 有效位置数量
```

例如真实 top-k 为 8，prefetch_k=12 时输出 avg_hits=6.3，表示预测的 12 个专家中，平均覆盖了 6.3 个真实专家。数值范围为 0 到 min(k′, k)，单位是专家个数。

```python
stats = cache.prefetch_hit_stats([8, 12, 16, 32])
print(stats["avg_hits"])       # {8: 平均数量, 12: 平均数量, ...}
print(stats["total_hits"])     # 各个 k′ 对应的命中总数，可跨样本累加
print(stats["num_positions"])  # 参与统计的有效 batch/token/layer 位置数
```

统计使用 `matched()` 对齐预测和真值，仅纳入源位置和目标位置都有效的记录。padding、缺少预测的位置和只有预测但尚无真值的未来位置不会计入分母。没有有效位置时，num_positions 为 0，avg_hits 中的值为 None。

验证集调用 `cache.prefetch_hit_stats(ks, router_mask=batch["router_mask"])`，将统计范围限定在 assistant 文本的目标位置，与训练损失使用相同的范围。

每层对最大的 k′ 执行一次 top-k，再累计各个排名前缀的命中数量。k′ 必须为正整数，且不超过该层专家总数。这个参数控制评估时选取多少个预测专家；模型仍按真实 Router 执行专家计算。

## 运行入口

[train.sh](train.sh) 使用 YAML 配置启动训练。从仓库根目录运行：

```bash
bash prefetch_qwen3_5_moe/train.sh
```

脚本固定读取 `prefetch_qwen3_5_moe/configs/train.yaml`。`CUDA_VISIBLE_DEVICES` 选择可见 GPU，`NPROC_PER_NODE` 指定本机训练进程数（默认 1）。四卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash prefetch_qwen3_5_moe/train.sh
```

请在仓库根目录执行，相对路径都从仓库根目录解析。脚本使用 torchrun 启动，Accelerate 根据进程编号自动选择对应 GPU。

[train.py](train.py) 从 [configs/train.yaml](configs/train.yaml) 读取训练配置。从仓库根目录运行：

```bash
python -m prefetch_qwen3_5_moe.train --config prefetch_qwen3_5_moe/configs/train.yaml
```

主要配置沿用原训练程序的字段：

| 配置 | 含义 |
| --- | --- |
| model.path / dtype | Qwen 模型路径和加载精度；设备由 Accelerate 分配 |
| output_dir | 本次训练结果目录 |
| prefetch.mode_prefetch / ks | 提前层数与 token 数、评估的专家预算 |
| data.max_length / max_image_side | 完整序列长度上限、图像最长边上限 |
| data.train | JSONL 训练数据源及 weight，可设置每个源的 limit |
| data.train_limit | 数据源混合后的总样本数上限 |
| data.validation | 按名称配置验证集 path 和 limit，例如 vl、text |
| training.max_steps | optimizer 更新总次数，必须为正整数；数据读完后重新打乱并继续读取 |
| training.gradient_accumulation | 每卡累积样本数，每卡每次 forward 一个样本 |
| training.learning_rate / weight_decay | AdamW 参数 |
| training.warmup_steps / max_grad_norm | warmup 后使用 cosine 学习率衰减，以及梯度裁剪 |
| training.log_every / eval_every / save_every | 按 optimizer step 计数的日志、验证、保存间隔 |

数据直接复用 [prefetch/datasets/dataset.py](../prefetch/datasets/dataset.py) 中的 training_records、load_records 和 SFTCollator。支持原训练程序的 CogVLM 图文记录、Tulu 文本记录及其加权混合。JSONL 每行的基本结构是：

```json
{"id": "sample-1", "messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}], "images": []}
```

图文记录的 images 填写本地图像路径列表。AutoProcessor 使用当前 Qwen checkpoint 的聊天模板和图像处理，将 pixel_values、image_grid_thw 等字段随完整上下文传入模型。data.router_tokens 使用 assistant；图像、prompt 和特殊 token 参与模型计算，损失和验证指标取 assistant 文本位置。

data.max_length 检查的是经过当前 processor 处理、包含视觉 token 的完整序列长度。超长样本会报错，应按当前模型 processor、max_length 和 max_image_side 预先过滤。

训练使用 Accelerate 管理 DDP、数据分片和梯度累积。每张 GPU 保存完整模型和独立的 RouterCache；各卡处理不同样本，在累积结束时同步 gate_prefetch 梯度。主模型保持 eval 并冻结参数，gate_prefetch 使用 FP32 参数通过 autograd 更新。DDP 要求每张卡能容纳完整模型及当前样本的训练缓存。

每卡每次 forward 一个样本，全局有效 batch 为 `进程数 × gradient_accumulation`。梯度累积跨过数据末尾继续进行，每次更新都累积相同数量的样本；训练在完成 max_steps 次更新后结束。Accelerate 在训练数据末尾按需重复少量样本，让各卡拥有相同工作量。

`PrefetchTrainingModel.forward()` 返回 RouterCache 计算的路由 loss，DDP 通过这个返回值跟踪预测头梯度。训练日志汇总各进程的样本平均 loss；学习率按 optimizer 更新次数推进。

验证由主进程使用完整验证集执行，其他进程等待。验证使用 torch.no_grad()，按 data.validation 的名称分别输出 KL 损失及各个 k′ 的平均命中数量，按有效 token-layer 位置数加权。配置、日志和权重均由主进程写入。

output_dir 内保存：

- `run_config.yaml`：本次配置。
- `train.jsonl`：训练日志。
- `metrics/step-N.json`：定期验证和最后一步的各验证集结果。
- `checkpoint-N.safetensors`：按 save_every 保存的 gate_prefetch 权重。
- `gate_prefetch.safetensors`：训练结束时的 gate_prefetch 权重。

[infer.py](infer.py) 调用原生 model.generate，并通过同一个 RouterCache 方法统计 prefill 和 decode 的平均命中数量：

```bash
python -m prefetch_qwen3_5_moe.infer --model /path/to/model --checkpoint prefetch_qwen3_5_moe/outputs/gate_prefetch.safetensors --prompt "解释一下混合专家模型。" --max-new-tokens 32 --prefetch-k 8 12 16 32
```

训练和推理使用相同的基础模型。新请求重置 RouterCache，KV cache 由原生 generate 管理。同一请求的 decode 持续追加路由。生成结束时，最后一个采样 token 尚未进入下一轮 forward，因此尚无对应的真实路由；查询时以记录的 token_end 为准。

当前程序验证预测路由及其读取。异步专家搬运可在源层 `save_prefetch()` 后发起，在目标层执行专家前检查完成状态。

## 环境与验证

CLI 加载 Transformers 的 `Qwen3_5MoeForConditionalGeneration`。训练使用 AutoProcessor 处理文本和图文记录，推理示例处理文本 prompt。`modeling.py` 是模型接口参考；用于 Qwen3.6 时，checkpoint 需要使用对应结构。已有文本 CausalLM 实例也可传给 `patch()`。

运行需要支持该 checkpoint 的 Transformers、PyTorch、Accelerate（`>=1.2,<2`）和 safetensors；训练还需要 PyYAML、datasets 和 Pillow。训练设备由 Accelerate 分配，router 权重需已物化在对应计算设备上。

```bash
python -m unittest discover -s prefetch_qwen3_5_moe/tests -t . -v
```

测试文件位于 `tests/`，以上命令从仓库根目录运行。CPU 测试使用参考文件中的实际 MoE 类。[test_prefetch.py](tests/test_prefetch.py) 覆盖整段写入、梯度、decode 衔接、容量和 checkpoint 恢复。[test_generation.py](tests/test_generation.py) 在随机初始化的小型 Llama 中使用这些 MoE block，通过 Transformers 原生 generate 验证生成结果、路由位置、padding、请求重置、停止回调和异常处理；测试本身需要安装 Transformers。完整 Qwen checkpoint 的 CUDA 运行需在对应环境中验证。

[test_metrics.py](tests/test_metrics.py) 用可手算的排名验证各个 k′ 的命中数量、有效位置对齐和测试集加权，并检查统计后仍能正常反向传播。

[test_training.py](tests/test_training.py) 验证 YAML 入口、实际 JSONL 混合、图文 collation、assistant 监督位置、梯度累积、分验证集报告及权重保存。测试使用小型模型和测试 processor；完整 Qwen 图文训练需在目标模型环境验证。

[test_distributed_training.py](tests/test_distributed_training.py) 使用两个真实 CPU/Gloo 进程，检查数据分片、梯度同步、跨数据末尾累积和主进程评估保存，并与相同有效 batch 的单进程训练比较权重、学习率和指标。完整 Qwen 模型的多 GPU/NCCL 训练需在目标硬件上验证。

一个 RouterCache 对应一个固定 batch 的连续请求。每层每次 forward 的保存和查询次数固定；张量读写量随序列长度和专家数增长。缓存显存随 `batch × max_tokens × 层数 × 专家数` 增长。长时间生成服务需要进一步设计历史淘汰；beam 重排、并发请求和 activation checkpointing 也需要对应的生命周期管理。
