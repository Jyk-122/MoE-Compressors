"""加载 gate_prefetch，通过原生 generate 评估 decode 阶段的预取命中数量。"""
import argparse
import json
import math
from pathlib import Path

from accelerate.utils import set_seed
import torch
import yaml

from prefetch.datasets.dataset import (
    OverlengthSample, load_records, multimodal_messages, open_images, processor_call, to_device,
)
from .patch import load_prefetch


def generation_inputs(processor, example, max_image_side=None):
    """取第一轮回答之前的消息，加上 assistant 生成前缀。"""
    messages = example["messages"]
    prompt_messages = []
    for message in messages:
        if message["role"] == "assistant":
            break
        prompt_messages.append(message)
    if not prompt_messages:
        raise ValueError(f"{example['id']}: generation needs a prompt before the first assistant answer")
    paths = example.get("images") or []
    return processor_call(
        processor, multimodal_messages(prompt_messages, paths),
        open_images(paths, max_image_side), generation=True,
    )


@torch.no_grad()
def evaluate_generation(model, processor, records, prefetch_k=(8, 12, 16, 32),
                        max_length=2048, max_new_tokens=128, max_image_side=None,
                        temperature=1.0):
    """逐条生成，只累计实际进入 decode forward 的目标 token 路由。"""
    model.eval()
    device = model.get_input_embeddings().weight.device
    cache = model.router_cache
    total_hits = {k: 0 for k in prefetch_k}
    total_loss = 0.0
    num_positions = 0
    num_examples = 0
    generated_tokens = 0
    decode_tokens = 0

    for example in records:
        inputs = generation_inputs(processor, example, max_image_side)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length > max_length:
            raise OverlengthSample(f"{example['id']}: prompt has {prompt_length} tokens > {max_length}")
        inputs = to_device(inputs, device)
        # 缓存包含 prompt、生成 token，以及提前预测的未来位置。
        capacity = prompt_length + max_new_tokens + model.mode_prefetch[1]
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, num_return_sequences=1, use_cache=True,
            return_dict_in_generate=False, output_router_logits=False,
            router_cache_max_tokens=capacity,
        )

        # prompt 属于 prefill；其后已写入真实路由的位置属于 decode。
        router_mask = torch.zeros((1, capacity), device=device, dtype=torch.bool)
        router_mask[:, prompt_length:] = True
        stats = cache.prefetch_hit_stats(prefetch_k, router_mask=router_mask)
        positions = stats["num_positions"]
        if positions > 0:
            loss = cache.loss(router_mask=router_mask, temperature=temperature)
            total_loss += loss.item() * positions
        num_positions += positions
        for k in total_hits:
            total_hits[k] += stats["total_hits"][k]
        num_examples += 1
        generated_tokens += output_ids.shape[1] - prompt_length
        # 最后采样出的 token 尚未执行 forward，以 cache 的真实终点为准。
        decode_tokens += cache.cache_gt[0].token_end - prompt_length
        cache.clear(max_tokens=capacity)

    if num_examples == 0:
        raise ValueError("Evaluation dataset is empty")
    avg_hits = {k: None for k in total_hits}
    loss = None
    if num_positions > 0:
        loss = total_loss / num_positions
        for k in total_hits:
            avg_hits[k] = total_hits[k] / num_positions
    return {
        "num_examples": num_examples,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "loss": loss,
        "num_positions": num_positions,
        "avg_hits": avg_hits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Trained gate_prefetch .safetensors file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", help="JSON report path; defaults to output_dir/metrics/test.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = config["data"]
    temperature = config["prefetch"]["temperature"]
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("prefetch.temperature must be finite and positive")
    if data["max_length"] <= 0:
        raise ValueError("data.max_length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    split = "test"
    if split not in data:
        split = "validation"
    if not data.get(split):
        raise ValueError(f"data.{split} must contain at least one evaluation dataset")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    set_seed(config.get("seed", 42))

    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_config["path"])
    dtype = model_config.get("dtype", "auto")
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    model_kwargs = {"dtype": dtype, "device_map": {"": torch.device(args.device)}}
    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_config["path"], **model_kwargs)
    load_prefetch(model, args.checkpoint)

    print(f"Checkpoint: {args.checkpoint}; mode_prefetch: {model.mode_prefetch}; data: {split}", flush=True)
    reports = {}
    for name, source in data[split].items():
        records = load_records(source["path"], source.get("limit"))
        print(f"Generating: {name}, {len(records)} examples", flush=True)
        reports[name] = evaluate_generation(
            model, processor, records, config["prefetch"]["ks"],
            max_length=data["max_length"], max_new_tokens=args.max_new_tokens,
            max_image_side=data.get("max_image_side"), temperature=temperature,
        )
        print(json.dumps({name: reports[name]}, ensure_ascii=False), flush=True)
    report = {
        "evaluation": "first_answer_generation",
        "scope": "decode",
        "checkpoint": args.checkpoint,
        "mode_prefetch": model.mode_prefetch,
        "max_new_tokens": args.max_new_tokens,
        "temperature": temperature,
        "datasets": reports,
    }
    output = Path(config["output_dir"]) / "metrics" / "test.json"
    if args.output:
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"Saved test report: {output}", flush=True)


if __name__ == "__main__":
    main()
