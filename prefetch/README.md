# MoE Prefetch Router 训练工程

在较早的 MoE 输入处预测目标层路由，训练一次即可评估多个候选数 k′。原 router 决定实际执行专家；真值来自当前基模（包含所选量化和 attention LoRA）的实际前向。

支持同 token 提前 M≥1 个 MoE、上一 token 同层或提前 M≥1 个 MoE、可选 attention LoRA SFT、routed-expert NF4、多卡训练、恢复、覆盖率曲线和单 prompt 推理。语义见 [DESIGN.md](DESIGN.md)。本阶段测量预测能力；Flash→DRAM 调度和端侧时延模拟可在后续实验中接入。

## 目录结构

~~~text
prefetch/
├── prerouter/    # Prerouter、全局 state、forward patch、配置与权重
├── backbone/     # 基模加载、attention LoRA、专家量化
├── datasets/     # 指令数据准备、加载与 collator
├── training/     # 训练入口、损失函数、DDP 与周期验证
├── evaluation/   # 评测入口、覆盖率统计、报告与绘图
├── examples/     # 单 prompt demo、真实模型 smoke test
├── configs/      # 实验 YAML
├── scripts/      # torchrun 启动脚本
├── tests/        # 单元测试与 DDP 测试
└── assets/       # 原模型和历史 demo 参考代码
~~~

运行时的数据仍保存到 `prefetch/data/`，训练及评测产物保存到 `prefetch/outputs/`。每个子包的 `__init__.py` 只声明包，不自动加载模型或训练依赖。

## 1. 环境与入口

服务器建议 Python 3.11，目标版本为 torch 2.9.0 / transformers 5.0.0。先安装适配服务器 CUDA 驱动的 PyTorch wheel，再安装依赖：

~~~bash
python -m pip install -r prefetch/requirements.txt
python -m pytest prefetch/tests -q
# 可选：CPU/Gloo 两进程梯度同步及无重复评测计数测试。
RUN_DDP_TESTS=1 python -m pytest prefetch/tests/test_patch.py -q
~~~

所有命令从仓库根目录运行，以文中的 python -m 完整模块路径启动。例如训练入口为 `python -m prefetch.training.train`，推理示例为 `python -m prefetch.examples.infer_prefetch_demo`。完整 checkpoint 必须包含原模型 remote code、processor 和配套依赖；[assets/modeling.py](assets/modeling.py) 用于结构参考。加载沿用原 demo 的 key_mapping。

先编辑 [same_token.yaml](configs/same_token.yaml) 中 model.path 和数据路径。另有 [previous_token.yaml](configs/previous_token.yaml) 与 [lora_sft.yaml](configs/lora_sft.yaml)。

`distance` 表示 source 到 target 的 MoE 层距离：same_token 要求 ≥1，previous_token 允许 ≥0。在 previous_token.yaml 中设为 0，即用 token t−1 的 MoE i 输入预测 token t 的同一 MoE i；默认覆盖全部 MoE，包括第 0 层。配置示例保留 distance=1，同层实验可修改为：

~~~yaml
output_dir: prefetch/outputs/previous_token_m0
prefetch:
  mode: previous_token
  distance: 0
  targets: null
~~~

将这些字段合入完整配置，其余配置保留。每种距离使用独立 output_dir 和对应训练的 head；跨距离比较覆盖率时固定共同 targets。

本地 torch 1.12 CPU 环境已通过小模型 patch、数据 mask、保存恢复、两进程 Gloo/DDP 测试。目标服务器的 torch 2.9 / transformers 5.0、完整 VL 模型、H20/NF4 CUDA 仍需按下文检查。

### Patch 阅读顺序

1. [prerouter/block.py](prerouter/block.py)：Prerouter 是独立的 nn.Module，forward 输入 hidden states，输出预测 logits。
2. [prerouter/patch.py](prerouter/patch.py)：moe_forward 明确写出原 router、信息收集、prerouter 和专家计算；patch 安装模块并替换 forward，unpatch 恢复。
3. [prerouter/state.py](prerouter/state.py)：全模型共用一个 PrerouterState，按目标层保存 predictions、router_logits、router_indices，处理层间和 token 间传递。
4. [training/train.py](training/train.py)：compute_prerouter_loss 在模型前向结束后读取 state 计算 KL；TrainingTask 负责训练调用和 DDP 可训练参数。
5. [evaluation/routing.py](evaluation/routing.py)：RoutingMetrics 读取 state 统计覆盖率和 trace；capture_generation 为 generate 的每次 forward 安装临时评测观察器。

训练和推理均要求 batch size=1。state 中的路由张量直接使用 [S,E] / [S,K]，省去 batch 维。最小训练调用：

~~~python
import torch
from prefetch.prerouter.patch import patch
from prefetch.training.train import compute_prerouter_loss
from prefetch.evaluation.routing import RoutingMetrics

model.eval().requires_grad_(False)
state = patch(model, config)           # 同时可通过 model.prerouter_state 访问
meter = RoutingMetrics(state)
state.reset(train_prerouter=True)      # 每条训练样本开始前调用
with torch.no_grad():
    model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
loss = compute_prerouter_loss(state, router_mask)
loss.backward()
meter.update(state, router_mask)       # 按需统计，与 loss 独立

target = state.pairs[0][1]
state.predictions[target]             # 已对齐目标 token 的预测 logits
state.router_logits[target]           # 原 router logits，已 detach
state.router_indices[target]          # 本次实际执行的 top-k 专家
~~~

head 注册为 source_moe.prerouter，输入是归一化后的 MoE 输入；调用时机为 source 原路由选择完成后、专家执行前。报告的 producer_timing 字段记录此时机。后续实验修改专家选择时，入口就是 moe_forward 中的 topk_indices/topk_weights 赋值。

## 2. 数据准备

统一 JSONL 格式：

~~~json
{"id":"sample-1","source":"example","group_id":"dialogue-id","images":[],"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}
~~~

CogVLM 先下载并解压官方数据，每个子数据集保留 images/、labels_en/、labels_zh/ 结构。脚本同时读取中英文标注，按同名文件匹配图片；两种语言分别生成独立样本，共用图片路径和图像内容 hash（group_id），因此同图的中英文、多条标注和跨目录副本属于同一 train/validation 分区。样本 ID 包含标注目录，区分两种语言。若根目录下没有中英文标注，则兼容读取 labels/。

脚本支持 captions/conversations。caption 样本默认使用英文提问 “Describe this image in detail.” 或中文提问“请详细描述这张图片。”，分别通过 --caption-prompt / --caption-prompt-zh 修改；conversation 保留标注中的提问和回答。

~~~bash
python -m prefetch.datasets.prepare cog  --root data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/ --output prefetch/data/cog --limit 1000

python -m prefetch.datasets.prepare tulu --dataset /data1/jiangyikun/datasets/tulu-3-sft-mixture --output prefetch/data/tulu --limit 1000
~~~

Tulu 默认使用 allenai/tulu-3-sft-mixture，按稳定 ID 留出 1% 验证集。Cog 和 Tulu 均支持 --limit N，限制划分 train/validation 之前的样本总数；省略时处理全量，设为 0 时输出空数据集。Cog 的每条 caption 或完整 conversation 各算一条样本，中英文分别计数。按子数据集、图片文件名排序，同图先处理英文再处理中文；达到 limit 即停止，因此最后一张图可能只保留部分标注。新的数据实验建议使用新输出目录。


接着用真实 processor 校验 assistant span，并过滤超长样本；长度/图像设置须与训练 YAML 相同：

~~~bash
for source in cog tulu; do
  for split in train validation; do
    python -m prefetch.datasets.prepare filter \
      --input "prefetch/data/$source/$split.jsonl" \
      --output "prefetch/data/$source/$split.filtered.jsonl" \
      --model-path /path/to/full/BF16/checkpoint \
      --max-length 2048 --max-image-side 672
  done
done
~~~

过滤保留完整对话和图像 span，报告 kept/overlength 数与 processor 配置。全量过滤需要 CPU 和图像读取时间，先用小样本检查模板。若 chat prefix 与完整序列 token 不一致，错误包含样本 ID，应按该 checkpoint 的 processor 调整 datasets/dataset.py 的区间提取。

每卡微批固定 1 条，不做 packing。data.train[].weight 控制 VL/text 采样；all_exhausted 策略可能重采样较小集合。data.validation 中的集合独立出报告，也可增加子集。默认只评价 assistant 文本输入位置，可切换 all_text；LM labels 包含 assistant 结束 token，router 文本指标排除 tokenizer 的 special tokens。

[CogVLM-SFT-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K) 为 CC-BY-NC-4.0；[Tulu](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) 部分子集也有非商业约束。商业用途需按实际子集授权筛选。

## 3. 单卡检查

先将 model.quantization 设为 none、model.nf4_checkpoint 设为 null，分别检查文本和图像：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.smoke_test \
  --config prefetch/configs/same_token.yaml \
  --sample-file prefetch/data/tulu/train.filtered.jsonl

CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.smoke_test \
  --config prefetch/configs/same_token.yaml \
  --sample-file prefetch/data/cog/train.filtered.jsonl
~~~

脚本打印监督文本、token 数、loss、梯度和显存峰值；断言 patch 前后 native logits 一致、冻结参数没有梯度，并执行一个 optimizer step。

model.router_forward_kwargs 默认传 logits_to_keep: 1，减少 router 阶段词表投影。检查输出 shape 是否仅含最后 1 个位置；完整 VL wrapper 若不支持，可设为 {}，或在其实现中透传该参数。patch 校验 processor input_ids 与真实 MoE 输入长度一致。

随后切换 experts_nf4 重复测试，额外检查 packed 权重和 expert 输入梯度。再用 lora_sft.yaml 检查 LoRA SFT。建议先用 max_length=512、max_image_side=336 的小样本，同时用对应设置重新生成过滤数据。

## 4. 训练与恢复

~~~bash
# 两卡短训前，将 max_steps 设为 2，log/eval/save_every 设为 1。
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash prefetch/scripts/train_ddp.sh prefetch/configs/same_token.yaml

# 完整配置与独立 output_dir 就绪后运行八卡。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash prefetch/scripts/train_ddp.sh prefetch/configs/same_token.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash prefetch/scripts/train_ddp.sh prefetch/configs/previous_token.yaml

# 同配置、同 world size 续训，只加载可信本地 trainer_state.pt。
NPROC_PER_NODE=8 bash prefetch/scripts/train_ddp.sh prefetch/configs/same_token.yaml \
  --resume prefetch/outputs/same_token_m1/checkpoint-100
~~~

每卡一份基模，DDP 仅管理可训练参数。8 卡×微批 1×累积 8 的有效训练 batch 为 64，与端侧 batch=1 的推理设定分别定义。从 BF16 源加载时，serial_load 默认按 rank 串行执行，CPU 需容纳一份 BF16 基模及缓冲。配置 nf4_checkpoint 后，各 rank 并行读取已量化权重，serial_load 不再参与；用法见下节。

router 阶段基模与 LoRA 冻结，基模处于 eval/no_grad。patch 在独立梯度作用域内执行 prerouter 和预测对齐，source 特征、teacher 标签均 detach；模型前向结束后，training/train.py 计算各目标层平均 KL，再反向。验证损失与覆盖率按有效 token 汇总。LoRA 阶段可启用 non-reentrant gradient checkpointing。

可选 LoRA 流程：

1. 运行 lora_sft.yaml，得到 attention_lora/checkpoint-N。
2. 在 router 配置设置 lora.enabled: true、lora.checkpoint: 对应目录。
3. 训练 predictor，此时 adapter 冻结，原 router 继续执行。

adapter 仅适配语言 decoder 的 self_attn.qkv_proj/o_proj。采用本工程轻量 LoRA 实现，保存 lora.safetensors 与 lora_config.json。

## 5. NF4 量化

model.quantization 可选 none / experts_nf4。后者将三维 routed-expert Parameter 显式适配为每专家两组 bitsandbytes Linear4bit；gate、shared experts、attention 和视觉模块沿用原精度。通用 HF load_in_4bit 不会自动覆盖这些三维专家参数。

### 分组大小

`model.nf4_blocksize` 默认 64，即每个专家矩阵展平后，每 64 个连续权重共享一组缩放信息。这不是按输出通道单独分组。可设置 64、128、256、512、1024、2048、4096；范围依据 [bitsandbytes 0.48.2 quantize_4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.2/bitsandbytes/functional.py)。双重量化 compress_statistics 保持开启，其中缩放系数的二级分组为 256，与权重的 blocksize 独立。

较小分组通常降低量化误差，同时增加缩放元数据；任务精度仍需实测。建议先比较 64 与 128。加载报告和 smoke_test 会输出并检查实际 blocksize。

当前配置量化覆盖 26,424,115,200 个源参数，其 BF16 载荷约 49.2 GiB、4-bit 原始载荷约 12.3 GiB，另加量化状态、其余权重、激活与训练状态。实际转换数和 CUDA allocator 统计记录在 loading_report.json。

NF4 路径以显存和语义对齐为目标，逐专家实现的吞吐需实测。bitsandbytes 0.48.2 / H20 / Torch 2.9 的组合及 LoRA 所需的输入反向能力必须通过 smoke test。每种量化配置以自身实际路由为真值，评测会核对 checkpoint 的基模和 adapter。

### 一次量化，重复加载

先编辑 YAML 中的 model.path，在单卡上导出纯基模。这个入口会关闭 LoRA，并从 BF16 源创建新的 NF4 权重：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.backbone.export_nf4 \
  --config prefetch/configs/same_token.yaml \
  --blocksize 64 --output prefetch/outputs/base_nf4_b64
~~~

输出目录必须是新目录。产物包括每个 MoE 的 packed uint8 权重、完整量化状态、其余原精度权重分片，以及 nf4_config.json。它是本工程的基模 checkpoint，不包含 prerouter 或 LoRA；LoRA 仍由 lora.checkpoint 单独加载。manifest 最后写入，导出中断时请使用新目录重试。

随后在训练、评测或 demo 使用的 YAML 中设置：

~~~yaml
model:
  path: /path/to/full/BF16/checkpoint
  quantization: experts_nf4
  nf4_blocksize: 64
  nf4_checkpoint: prefetch/outputs/base_nf4_b64
  serial_load: true
  router_forward_kwargs: {logits_to_keep: 1}
~~~

将字段合入原配置，其余设置保留。加载时从 model.path 读取配置、remote code 和 processor，创建空参数骨架，然后直接恢复已量化权重；不读取 BF16 模型权重，也不重新量化。请保留原目录及不可变版本，model.path 须与导出时一致。nf4_blocksize 必须与 checkpoint 一致；要比较 128，请另行导出并使用独立目录。

NF4 checkpoint 由本工程加载，不直接传给通用 AutoModel.from_pretrained。加载时按 MoE/原精度分片读取，CPU 不再需要完整 BF16 权重；每卡仍保存完整 NF4 基模，多卡并发读盘吞吐需实测。首次导出仍需 BF16 源权重所需的 CPU 内存。

服务器上先验证真实 bitsandbytes 保存/恢复和输入反向，再运行完整模型 smoke_test：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  prefetch/tests/test_quantization.py prefetch/tests/test_nf4_checkpoint.py -q

CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.smoke_test \
  --config prefetch/configs/same_token.yaml \
  --sample-file prefetch/data/cog/train.filtered.jsonl
~~~

CUDA roundtrip 测试覆盖 blocksize=64/128，检查 packed 权重、量化状态和输出一致，并检查恢复后的输入梯度。本地 CPU 测试只验证序列化流程与加载分支，不能替代真实 NF4 CUDA 验证。

### 验证基模输出与量化精度

单 prompt demo 直接运行基模，不安装 prerouter 或 LoRA。默认按 YAML 选择量化及 checkpoint；可用 --quantization 覆盖。纯文本省略 --image：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.infer_base_demo \
  --config prefetch/configs/same_token.yaml --quantization none \
  --prompt '请描述这张图片，并说明判断依据。' --image /datasets/example.jpg \
  --output prefetch/outputs/base_check/bf16.json

CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.infer_base_demo \
  --config prefetch/configs/same_token.yaml --quantization experts_nf4 \
  --prompt '请描述这张图片，并说明判断依据。' --image /datasets/example.jpg \
  --output prefetch/outputs/base_check/nf4.json
~~~

两个进程顺序执行，避免同时驻留两份模型。none 会忽略 YAML 中的 nf4_checkpoint；experts_nf4 在配置了 checkpoint 时直接加载，否则现场量化。两次使用同一预处理与 greedy decoding，但合理回答可能不同；单条回答用于检查可用性，不代表任务精度保持。

定量比较使用固定验证集，按 assistant 的有效 next-token 数加权计算 NLL 和 perplexity；越低越好。以下命令分别运行 BF16/NF4，默认评估前 128 条，也可以换成 Cog 的验证集：

~~~bash
for quant in none experts_nf4; do
  CUDA_VISIBLE_DEVICES=0 python -m prefetch.evaluation.evaluate_base \
    --config prefetch/configs/same_token.yaml --quantization "$quant" \
    --sample-file prefetch/data/tulu/validation.filtered.jsonl --limit 128 \
    --output "prefetch/outputs/base_check/$quant-nll.json"
done
~~~

比较 assistant_nll 的 NF4−BF16 差值和 assistant_perplexity 的相对变化，并确认 examples/tokens 一致。这里的 PPL 是给定指令/图像、只统计 assistant 标签（含结束 token）的条件 PPL，不等同于通用语料 PPL 或任务准确率。正式精度结论仍需固定任务集上的正确率/生成质量指标；prefetch Recall/FullCoverage 衡量预测路由的覆盖率，不能替代基模精度评测。

## 6. 评测与单 prompt demo

周期验证自动生成 JSON、CSV、Recall PNG、FullCoverage PNG，包含 global/per-layer、有效 token-layer pair 数和完整 rank histogram。LoRA 阶段报告 SFT 验证损失；router 阶段报告覆盖率。

~~~bash
NPROC_PER_NODE=8 bash prefetch/scripts/evaluate.sh \
  prefetch/outputs/same_token_m1/checkpoint-2000 \
  prefetch/outputs/eval_teacher

NPROC_PER_NODE=8 bash prefetch/scripts/evaluate.sh \
  prefetch/outputs/previous_token_m1/checkpoint-2000 \
  prefetch/outputs/eval_generation --mode generation --max-new-tokens 128

CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.infer_prefetch_demo \
  --checkpoint prefetch/outputs/previous_token_m1/checkpoint-2000 \
  --prompt '请描述一下这张图。' --image /datasets/example.jpg \
  --output prefetch/outputs/demo/metrics.json
~~~

纯文本 demo 省略 --image。连线检查可省略 --checkpoint 并传 --config，此时报告标记 predictor=initialized。

generation 评测只使用验证对话的第一个 assistant 回答前的上下文，prefill/decode 分开报告。支持 batch=1、num_beams=1、use_cache=true 的普通生成。返回的最后一个生成 token 通常尚未再输入模型，因此 decode 路由调用数可能比生成 token 数少 1。

demo 使用 with capture_generation(model, state, meter) 包住每次 generate：进入时重置请求状态，每个 forward 完成后统计，退出时释放观察器和请求张量。上一 token 模式用 prefill 尾部预测首个 decode。手动逐 token 推理可先调用 state.reset(generation=True)，再在每次 forward 后调用 meter.update(state)。trace 包含 source/target MoE 序号、输入 token 位置、预测集和真值；trace_limit 分阶段限额。demo 时间包含预测及统计开销，不代表 Flash 预取收益。

比较层距离时，在 prefetch.targets 固定共同目标 MoE 序号，或从直方图重新汇总共同层：

~~~bash
python -m prefetch.evaluation.plot prefetch/outputs/eval_teacher/text-stepfinal.json \
  --layers 2 3 4 --output-prefix prefetch/outputs/common_layers/text
~~~

Recall@E 与 FullCoverage@E 对有效样本必须为 1，空计数输出 null。最大真实专家 rank 的分位数可估计覆盖 95%/99% 调用所需的经验 k′；它不构成未见数据的覆盖保证。

## 7. 代码导航与产物

| 文件 | 职责 |
|---|---|
| prerouter/block.py | 独立 Prerouter 神经网络 |
| prerouter/patch.py | MoE forward、模型入口与安装/卸载 |
| prerouter/state.py | 全局路由数据、层映射与跨 token 状态 |
| evaluation/routing.py / evaluation/metrics.py | 覆盖率计数、trace 与汇总报告 |
| prerouter/configuration.py / prerouter/checkpoint.py | 预测配置、head 保存加载 |
| backbone/loading.py / backbone/quantization.py / backbone/structure.py | 原模型、attention adapter、NF4、MoE 结构与路由评分 |
| backbone/export_nf4.py / backbone/nf4_checkpoint.py | NF4 基模导出、分片权重与量化状态恢复 |
| datasets/prepare.py / datasets/dataset.py | 规范化、过滤、processor、mask |
| training/train.py / training/runtime.py | KL 损失、两阶段训练、DDP、保存恢复、周期验证 |
| evaluation/evaluate.py / evaluation/plot.py | 独立评测与曲线 |
| examples/infer_base_demo.py / evaluation/evaluate_base.py | 纯基模 prompt 推理、固定验证集 NLL/PPL |
| examples/infer_prefetch_demo.py / examples/smoke_test.py | 单 prompt 演示、真实模型检查 |

训练 checkpoint 保存 head 或 LoRA、运行配置、optimizer/scheduler、各 rank RNG、epoch 和下个 batch 位置。head 模块归 source block 所有，predictor.safetensors 仍采用目标 MoE 序号作为键；已有 target-keyed head checkpoint 可加载，optimizer 参数顺序保持配置中的 pair 顺序。基模权重由 model.path 或独立的 model.nf4_checkpoint 提供，请保留不可变的基模、NF4 checkpoint 与 adapter 版本。当前支持 n_group=1、常规 attention；换模型结构前检查层映射和执行语义。
