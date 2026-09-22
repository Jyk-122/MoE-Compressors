import time
import types
import pickle
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, BitsAndBytesConfig
from transformers.utils import is_torch_npu_available
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
if is_torch_npu_available() and "910" in torch.npu.get_device_name():
    # import torch_npu
    from torch_npu.contrib import transfer_to_npu
from loguru import logger



parser = argparse.ArgumentParser(description="Inference demo with Layer-wise TopP threshold pruning for PXX_VL MoE.")
parser.add_argument("--prune_state_path", type=str, default="/data5/jiangyikun/MoE/PXX_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/state_scale_topp_layerwise_learn_50_40_mixed_v5_fixed.pth")

args = parser.parse_args()

################### Origin model #########################
model_path="/data5/jiangyikun/MoE/PXX_VL-30A2B-dev_5.2.1.1_SP1_0616/HFModel/BF16"


image_path = "/data5/jiangyikun/MoE/PXX_VL-30A2B-dev_5.2.1.1_SP1_0616/demo.jpg"
prompt = "请描述一下这张图。"
# prompt = "你是谁"

print(f"LOAD MODEL FROM: {model_path}")
key_mapping = {
        r"^visual": "model.visual",
        r"^model(?!\.(language_model|visual|lm_head))": "model.language_model",}
model =  AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype="bfloat16",
        key_mapping=key_mapping
        )
model.eval()

####################### patch pruning ###########################
collector = { "moe_mask": [] }

from patch import patch
model = patch(model, args.prune_state_path, collector)

# from patch_mask import patch
# model = patch(
#     model, 
#     threshold_pth_path="/data5/jiangyikun/MoE/PXX_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/thresholds_learned_fc_50_40_checkpoint_3.pth",
#     prune_state_path="/data5/jiangyikun/MoE/PXX_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/state_scale_topp_fc_learn_50_40_v1.pth",
#     collector=collector
# )

####################### infer example ###########################

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }
] 

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,)
print(f"text is : {text}")
if image_path is not None:
    images = [Image.open(image_path).convert("RGB")]
else:
    images = None
if not isinstance(text, list):
    text = [text]
inputs = processor(text=text, images=images, padding=False, return_tensors="pt",)
# inputs = processor(text=text, padding=False, return_tensors="pt",)

inputs = inputs.to(model.device)

time_start = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=200,
                                    do_sample=False)
time_cost = time.time() - time_start
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
res = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f"OUTPUT: {res}")

active_prefill = torch.cat([x.float().unsqueeze(1).cuda(0) for x in collector["moe_mask"][:35]], dim=1)
active_decode = torch.cat([x.float().unsqueeze(1).cuda(0) for x in collector["moe_mask"][35:]], dim=1).reshape(-1, 35, 8)

N_prefill = active_prefill.shape[0]
N_decode = active_decode.shape[0]
prune_ratio_prefill = 1 - active_prefill.mean()
prune_ratio_decode = 1 - active_decode.mean()

print(f"{'='*65}")
print(f"  Decode tokens: {N_decode}")
print(f"  Decode prune ratio: {prune_ratio_decode}")
print(f"  Prefill tokens: {N_prefill}")
print(f"  Prefill prune ratio: {prune_ratio_prefill}")
print(f"  Total time cost: {time_cost:.3f}s")
print(f"{'='*65}")
