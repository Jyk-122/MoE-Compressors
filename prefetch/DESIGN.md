# OpenPXX VL Prefetch Router 工程设计

目标环境：Linux、H20、PyTorch 2.9.0、Transformers 5.0.0、Python 3.11。实现位于本目录，完整模型从原 checkpoint 的 remote code 加载。

## 1. 实验定义

固定基模版本、量化方式、attention adapter、数据 split、head 结构与温度，比较预测位置。一个 predictor 输出完整排序，一次训练评估所有候选数 k′。

原 router 决定实际专家及权重；head 是旁路。真值使用当前基模的 gate 输出及原生 route_tokens_to_experts。可选 attention LoRA 先用原路由 SFT，再冻结 adapter 训练 predictor。router 阶段只对 head 求梯度，teacher 标签 detach。

此阶段实现预测训练与正确性评测。Flash→DRAM 调度、缓存容量和端侧时延模型可后续读取排序与覆盖率结果，独立控制传输变量。

## 2. 模型接口

参考 [assets/modeling.py](assets/modeling.py)、[assets/config.json](assets/config.json)：37 个 decoder，前 2 个 dense，35 个 MoE；hidden=2560、experts=384、K=8、expert intermediate=256。当前 use_mhc=true、use_mla=false、n_group=1。

Prerouter 是独立的 nn.Module，注册为 source_moe.prerouter。输入来自 mHC 合并、pre_mlp_layernorm 之后的 [1,S,2560]，head 实际接收 [S,2560]。prerouter/patch.py 的 moe_forward 沿用参考模型的 gate、路由选择、routed/shared experts 计算，并在专家执行前收集信息、调用 prerouter。decoder residual 可能含 4 条 stream，含义不同。

gate 显式使用 FP32 linear。原路由按 sigmoid(logits)+e_score_correction_bias 选择专家，再用未加 bias 的 sigmoid 分数得到组合权重。PrerouterState 收集本次原生路由选择的真实集合 R，评测直接读取。shared experts 保留原执行路径。

LoRA 对应语言 decoder 的 self_attn.qkv_proj/o_proj；通过实际 MoE 模块路径定位语言 decoder 容器，包括前面的 dense decoder。

## 3. 位置和对齐

MoE 序号从 0 开始，checkpoint 同时保存真实模块路径：

| mode | source | target | 有效目标 |
|---|---|---|---|
| same_token | token t，MoE i−M 输入 | token t，MoE i router | i≥M |
| previous_token | token t−1，MoE i−M 输入 | token t，MoE i router | i≥M |

M=distance 按 MoE 数计算：same_token 要求 M≥1；previous_token 允许 M≥0。M=0 时，source 和 target 属于同一 MoE，token 相差 1，所有层 i≥0 均可作为目标；序列首 token 仍无跨 token 标签。prefetch.targets 可固定共同目标层，默认使用全部有效目标。

训练和推理的 batch size 均为 1，MoE 输入为 [1,S,H]，state 中的路由信息为 [S,E] 或 [S,K]。每卡训练一条样本，可用 DDP 和梯度累积增加有效训练 batch；样本间独立，不做 packing。

全模型共用一个 PrerouterState，以下字典都以目标 MoE 序号为键：

| 字段 | 内容 | 生命周期 |
|---|---|---|
| predictions | 对齐到目标位置的预测 logits [S,E] | 当前 forward |
| router_logits | detached 原 router logits [S,E] | 当前 forward |
| router_indices | 实际选出的专家 [S,K] | 当前 forward |
| next_predictions | 下一 decode 的预测 [1,E] | 跨一个 forward |

传递规则集中在 prerouter/state.py：

1. same_token：source i 的输出直接放入 predictions[i+M]。
2. previous_token 完整序列：source i 的 source[:-1] 对齐 target i+M 的 target[1:]，首位置无效；检查 source/target attention mask。M=0 时在同一层完成对齐。
3. previous_token decode：forward 开始时把 next_predictions 交给 predictions，再建立新的 next_predictions。当前 forward 的 head 输出保存到下一轮。

state.valid_mask 标记对齐后有效的位置，并排除配置中的特殊 token。训练函数再与数据的 router_mask 取交集；router mask 对应当前输入 token，LM loss 才做下一 token shift。

每条样本开始调用 state.reset(train_prerouter=True)。手动生成开始调用 state.reset(generation=True)，支持 num_beams=1、use_cache=true 的单 token decode；prefill 的尾部预测用于首个 decode。next_predictions 只保存最后一个位置的 detached clone。完整 VL processor 的 input_ids 长度须与 MoE 输入一致。

## 4. 模块职责和调用顺序

组织方式参考 Edge0 的 [PrerouterState](https://github.com/Edge0-AI/Edge0/blob/fb4cd2c49ebe22bb230e1451ecb8fb4957ca62e6/src/edge0/prerouter/state.py) 和 [安装/forward patch](https://github.com/Edge0-AI/Edge0/blob/fb4cd2c49ebe22bb230e1451ecb8fb4957ca62e6/src/edge0/prerouter/install.py)。

- prerouter/block.py：Prerouter 的网络结构和 forward。
- prerouter/patch.py：patch 返回 model.prerouter_state；安装 head，替换 MoE 和 model forward；unpatch(model) 恢复。
- prerouter/state.py：PrerouterState 收集路由张量，管理层映射、token 对齐与请求生命周期。
- training/train.py：router_loss / compute_prerouter_loss 定义损失；TrainingTask 运行基模，再读取 state 算 loss。
- evaluation/routing.py：RoutingMetrics 独立读取 state，累计覆盖率和 trace。
- prerouter/checkpoint.py：save_predictor / load_predictor 保存加载 head。

一次 router 训练：

~~~python
state.reset(train_prerouter=True)
with torch.no_grad():
    model(**inputs, use_cache=False)
loss = compute_prerouter_loss(state, router_mask)
loss.backward()
~~~

基模与 attention LoRA 冻结并处于 eval；source 输入和 teacher 标签 detach。只有 prerouter 及其预测对齐在单独的梯度作用域内执行。loss 在模型前向结束后计算，处于 DDP wrapper 的 forward 内。验证/推理设置 train_prerouter=False。

DDP wrapper 只注册可训练 ParameterList，参数按配置中的 pair 顺序收集，保持 optimizer 恢复顺序；冻结基模通过普通引用持有。head/LoRA 使用 FP32 master weights，新增支路内部 CUDA BF16 autocast。

生成评测使用 capture_generation(model, state, meter) 包住一次 generate。评测侧临时挂 model forward hook，每步完成后 meter.update(state)，请求结束时卸载观察器并清理 state；报告计数单独保留。直接调用 model forward 只收集信息，由调用方决定后续处理。

prerouter 读取 source MoE 输入，实际调用在 source 原路由选择之后、专家计算之前；报告 producer_timing=after_source_routing_before_experts。未来改变执行路由时，修改 moe_forward 中的 topk_indices/topk_weights 选择即可，监督和指标仍可读取原 router 标签。

## 5. Head、损失与指标

每目标默认 Linear(2560,512)→GELU→Linear(512,384)。linear 基线使用单层 Linear，初始化为目标 gate 权重。

默认 score_kl：

    s_true = sigmoid(z_true) + bias
    s_pred = sigmoid(z_pred) + bias
    p = softmax(s_true / T)
    q = softmax(s_pred / T)
    L = mean_valid_tokens KL(p || q)

teacher/bias detach。T 显式配置，初始 0.1 需小样本验证；损失不额外乘 T²。logit_kl 作为原始 logits 蒸馏对照。训练对样本、目标层取均值，验证汇总有效 token；空有效样本给出连接全部 head 的零损失。

P_k′ 按 s_pred 的稳定降序排名；R 来自目标层本次实际执行的 topk_indices。每个有效 token/目标层：

    Recall@k′ = |R ∩ P_k′| / K
    FullCoverage@k′ = 1[R ⊆ P_k′]

真实专家 rank 直方图计算 Recall，最大真实专家 rank 直方图计算 FullCoverage。GPU 间整数计数相加后计算比例；per-layer/global、VL/text、teacher_forcing/prefill/decode 分别报告。空分母为 null，k′=E 必须完整覆盖。

完整直方图支持任意 k′、共同层汇总和经验覆盖分位数。经验 k′ 需在未见样本和真实生成分布再次检验。

## 6. Attention LoRA

stage=lora 仅训练语言 attention 的低秩 A/B，使用 assistant causal LM loss，初始 B=0。支持 non-reentrant gradient checkpointing。

stage=router 加载并冻结 adapter，用固定适配基模的实际路由训练 head。lora.enabled/checkpoint 控制加载。保存内容为 lora_A/lora_B 与 rank/alpha/dropout，完整基模仍使用原 checkpoint。

## 7. NF4 专家量化

HF 5.0.0 通用 bitsandbytes 替换针对 nn.Linear/Conv1D。本模型专家权重为 gate_up_proj:[E,2I,H]、down_proj:[E,H,I] 三维 Parameter，因此显式适配执行器。

experts_nf4 将每专家两个二维矩阵包装为 bitsandbytes Linear4bit，保留激活、top-k 权重和 index_add 聚合。其他模块保留原精度；none 提供 BF16 基线。

各 rank 并发加载自己的基模副本。BF16 通过 Transformers/Accelerate 的 device_map 直接加载到本 rank GPU；已有 NF4 checkpoint 则恢复 packed 权重和量化状态。现场 NF4 的流程为：CPU 原模型→CPU 逐 MoE/专家量化→释放原专家 BF16 权重→整模迁移到本 rank GPU→安装 adapter/head。现场量化期间主机需容纳并发副本及临时内存；固定实验可通过预先导出的 NF4 checkpoint 复用权重。

routed-expert 参数共 35×384×3×2560×256=26,424,115,200。BF16 载荷约 49.2 GiB，4-bit 载荷约 12.3 GiB，另加量化状态和其他开销。报告记录实际转换数、CUDA allocator 峰值。

bitsandbytes 0.48.2 源码提供输入梯度能力，可用于上游 attention LoRA；CUDA/H20/Torch 2.9 的实际组合由 smoke test 验证。逐专家实现以显存和语义对齐为目标，吞吐需实测。

固定版本依据：

- [HF v5.0.0 bitsandbytes 替换](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/integrations/bitsandbytes.py)
- [bitsandbytes 0.48.2 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.2/bitsandbytes/nn/modules.py)
- [bitsandbytes 0.48.2 autograd](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.2/bitsandbytes/autograd/_functions.py)

## 8. 数据、并行和产物

CogVLM captions/conversations 与 Tulu messages 统一为 JSONL。Cog 按图像内容 hash、Tulu 按 ID 做稳定 split。过滤阶段通过真实 processor 的 assistant 前后 chat prefix 校验 token 区间，过滤超长样本，保留完整视觉 span。

训练 microbatch=1，通过梯度累积控制有效 batch。图像最大边、序列上限、数据源权重记录在配置中。VL/text 验证独立，验证 sampler 不重复补齐样本；各 rank 脱离 DDP wrapper 运行不同长度验证 shard，最后统一归约计数。

checkpoint 保存 predictor 或 LoRA safetensors、配置、optimizer/scheduler、各 rank RNG、epoch、下个 batch 位置。head 归 source block，权重键保持 target_moe 前缀，兼容已有该格式的 predictor checkpoint。恢复要求同 run config 和 world size，数据/基模路径须指向不可变内容。

报告为 JSON、CSV、Recall/FullCoverage PNG；demo trace 按阶段限额，包含 source/target token 与层位置。数据许可与运行命令见 README.md。

## 9. 验收

1. 小模型检查 patch 保留 native forward、层映射、mask、请求隔离。
2. previous_token 完整序列与 prefill+decode 计数一致。
3. head 梯度、teacher detach、零有效 token、checkpoint roundtrip。
4. LoRA 零初始化与保存恢复；NF4 packed 权重、expert 前向和输入梯度。
5. 真实文本/图像 processor 的 mask/长度、BF16/NF4 native logits 对齐。
6. 两卡短训、恢复、验证归约，再扩大到八卡。

本地 torch 1.12 CPU 环境已验证全局路由收集、两种传递模式、prefill/decode 等价性、独立 loss/梯度对照、batch=1 约束、原输出和 forward 签名、checkpoint 保存恢复，以及两进程 Gloo/DDP 梯度同步与计数归约。目标 torch 2.9 / transformers 5.0、完整 VL wrapper 与 H20/NF4 CUDA 兼容性需按 README 在服务器执行验证。
