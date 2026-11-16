"""Load an HF model, apply the selected patch, and use native generate()."""

import argparse
from importlib import import_module
import json

import torch

from .data import multimodal_messages, open_images, processor_call, to_device
from .utils import activation_stats_enabled, load_model_and_processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="skip_transfer")
    parser.add_argument("--model-type", default="qwen3_5_moe", choices=("qwen3_5_moe",))
    parser.add_argument("--patch-kwargs", type=json.loads, default={},
                        help="JSON object passed directly to the method's patch(model, **kwargs)")
    parser.add_argument("--prompt", default="请解释混合专家模型的工作原理。")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--max-image-side", type=int, default=672)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model, args.device, args.dtype)
    module = import_module(f"transfer_moe.{args.method}.{args.model_type}.patch_hf")
    module.patch(model, **args.patch_kwargs)
    messages = multimodal_messages([{"role": "user", "content": args.prompt}], args.images)
    images = open_images(args.images, args.max_image_side)
    inputs = processor_call(processor, messages, images, generation=True)
    inputs = to_device(inputs, model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output[0, inputs["input_ids"].shape[1]:]
    print(processor.tokenizer.decode(generated, skip_special_tokens=True))
    if activation_stats_enabled():
        print(json.dumps(model.activation_stats.report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
