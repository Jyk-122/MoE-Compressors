"""调用 Transformers 原生 generate，并查询 RouterCache 中的路由。"""
import argparse

from .patch import load_prefetch


def main():
    parser = argparse.ArgumentParser(description="Generate text and inspect cached router predictions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="请简要解释什么是混合专家模型。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefetch-k", type=int, nargs="+", default=[8, 12, 16, 32],
                        help="Numbers of predicted experts to evaluate")
    args = parser.parse_args()
    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map=args.device)
    cache = load_prefetch(model, args.checkpoint)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        model.get_input_embeddings().weight.device)
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos_token_id,
        output_router_logits=False,
    )
    print(tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))

    stats = cache.prefetch_hit_stats(args.prefetch_k)
    print(f"Matched token-layer positions (prefill + decode): {stats['num_positions']}")
    for k, avg_hits in stats["avg_hits"].items():
        if avg_hits is not None:
            print(f"prefetch_k={k} avg_hits={avg_hits:.4f}")


if __name__ == "__main__":
    main()
