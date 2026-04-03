# 启动虚拟环境
conda activate moe

# 观察显卡使用情况
nvitop

# skipping样例

## topP不需要calib，可以直接进行eval 
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 METHOD=topp_skip PATCH_KWARGS='{"threshold":0.8}' TASKS="gsm8k" EVAL_BATCH_SIZE=1 MODEL=/data1/jiangyikun/models/Qwen3-30B-A3B-Instruct-2507 bash run_skipping.sh eval

## 如果进行性能评测，建议使用多卡并行推理以加速
## 如果进行专家id的记录，建议使用单卡，可以调整eval_limit来减少测试样本数量
CUDA_VISIBLE_DEVICES=1 METHOD=topp_skip PATCH_KWARGS='{"threshold":0.8}' TASKS="gsm8k" EVAL_LIMIT=100 EVAL_BATCH_SIZE=1 MODEL=/data1/jiangyikun/models/Qwen3-30B-A3B-Instruct-2507 bash run_skipping.sh eval

# pruning样例

## REAP需要首先进行calib
## calib得到的结果保存到./outputs/Qwen3-30B-A3B-Instruct-2507/reap_pruning/目录下
## REAP在calib时默认剪枝率为0.5，可以在calib_kwargs中指定其他剪枝率，其会保存对应的映射
CUDA_VISIBLE_DEVICES=1 METHOD=reap_pruning CALIBRATION_DATASET="/data1/jiangyikun/datasets/wikitext:wikitext-2-raw-v1" MODEL=/data1/jiangyikun/models/Qwen3-30B-A3B-Instruct-2507 bash run_pruning.sh calib

## 评测时需要指定adapter目录
## 评测时可以在patch_kwargs直接指定其他剪枝率，patch会自动根据adapter中的映射进行重算
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 METHOD=reap_pruning PATCH_KWARGS='{"prune_ratio":0.5}' ADAPTER_DIR="./outputs/Qwen3-30B-A3B-Instruct-2507/reap_pruning/" TASKS="gsm8k" EVAL_BATCH_SIZE=1 MODEL=/data1/jiangyikun/models/Qwen3-30B-A3B-Instruct-2507 bash run_pruning.sh eval

## 同样的，在进行专家id记录时，还是建议使用单卡评测
CUDA_VISIBLE_DEVICES=1 METHOD=reap_pruning PATCH_KWARGS='{"prune_ratio":0.5}' ADAPTER_DIR="./outputs/Qwen3-30B-A3B-Instruct-2507/reap_pruning/" TASKS="gsm8k" EVAL_BATCH_SIZE=1 MODEL=/data1/jiangyikun/models/Qwen3-30B-A3B-Instruct-2507 bash run_pruning.sh eval

# Notes
## 所有的eval会保存评测结果
## 目前我们只考虑生成式任务，所以需要设置EVAL_BATCH_SIZE为1
## 可以修改moe_stats_collector来使其可以记录所有专家的id，也可以使用其他方式记录
## 保存id后构建cache，来绘制 内存-专家IO 的曲线