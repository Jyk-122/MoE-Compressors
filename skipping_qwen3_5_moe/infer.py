"""Generate using a caller-selected top-p threshold and calibrated compensation."""
import argparse
import json

from .model import load_compensation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--prompt", default="请解释混合专家模型的工作原理。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", device_map={"": args.device}).eval()
    runtime = load_compensation(model, args.checkpoint)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.get_input_embeddings().weight.device)
    output = model.generate(**inputs, threshold=args.threshold, max_new_tokens=args.max_new_tokens,
                            do_sample=False, num_beams=1, output_router_logits=False)
    print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
    print(json.dumps(runtime.summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
