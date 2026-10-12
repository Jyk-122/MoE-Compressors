
import os
import json
import pickle
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.utils import is_torch_npu_available
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types
from datasets import load_dataset
from safetensors.torch import load_file, save_file

from evolution import evolve_topk_allocation

logger = logging.getLogger("OptimalScale")


def load_calibration_text_data(
    processor,
    calibration_dataset: str = "/data1/jiangyikun/datasets/slimpajama-200k",
    max_calib_samples: int = 128,
    max_context_len: int = 2048,
):
    logger.info("Loading calibration dataset: %s", calibration_dataset)
    print("### ", calibration_dataset)
    parts = calibration_dataset.split(":", 1)
    dataset_name = parts[0]
    config_name = parts[1] if len(parts) > 1 else None
    
    ds = load_dataset(dataset_name, config_name, split="train")

    # --- 1. 数据格式标准化 ---
    formatted_texts = []
    
    if "instruction_zh" in ds.column_names:
        logger.info("Detected alpaca-data-gpt4-chinese-style dataset. Formatting to chat messages...")
        for item in ds:
            # 合并 instruction 和 input
            user_content = item["instruction_zh"]
            if "input_zh" in item and item["input_zh"]:
                user_content += "\n" + item["input_zh"]
            
            messages = [
                {"role": "user", "content": {"type":"text", "text": user_content}},
                {"role": "assistant", "content": item.get("output_zh", "")}
            ]
            
            # 使用 processor 的 chat_template 转换为最终文本
            # tokenize=False 返回字符串，add_generation_prompt=False 因为我们已经有输出了
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
    # 判断是否为 Alpaca 格式 (包含 instruction 字段)
    elif "instruction" in ds.column_names:
        logger.info("Detected Alpaca-style dataset. Formatting to chat messages...")
        for item in ds:
            # 合并 instruction 和 input
            user_content = item["instruction"]
            if "input" in item and item["input"]:
                user_content += "\n" + item["input"]
            
            messages = [
                {"role": "user", "content": {"type":"text", "text": user_content}},
                {"role": "assistant", "content": item.get("output", "")}
            ]
            
            # 使用 processor 的 chat_template 转换为最终文本
            # tokenize=False 返回字符串，add_generation_prompt=False 因为我们已经有输出了
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
    else:
        # 兼容原有逻辑：读取 "text" 或第一个字段
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        formatted_texts = [str(t) for t in ds[col] if t and str(t).strip()]

    # --- 2. 文本分块逻辑 (复用并优化) ---
    logger.info("Valid samples after formatting: %d, building chunks of ~%d tokens", 
                len(formatted_texts), max_context_len)

    chunks = []
    current_chunk_text = ""
    pbar = tqdm(total=max_calib_samples, desc="Building calibration chunks", unit="chunk")
    
    for text in formatted_texts:
        current_chunk_text += text
        
        # 只有当 token 数达到阈值时才切分
        inputs = processor(text=current_chunk_text, add_special_tokens=False)
        n_tokens = len(inputs["input_ids"][0])
        
        if n_tokens >= max_context_len:
            print("text: ", current_chunk_text)
            inputs = processor(text=current_chunk_text, padding=False, return_tensors="pt",)
            chunks.append(inputs)
            current_chunk_text = "" # 重置
            pbar.update(1)
            
            if len(chunks) >= max_calib_samples:
                break

    pbar.close()
    logger.info("Calibration data loaded, %d chunks", len(chunks))
    return chunks


def load_calibration_gqa_data(
    processor,
    calibration_dataset: str = "/data1/jiangyikun/datasets/GQA/",
    max_calib_samples: int = 128,
    max_context_len: int = 2048,
):
    logger.info("Loading calibration dataset: %s", calibration_dataset)
    
    ds_instructions = load_dataset(calibration_dataset, "train_balanced_instructions", split="train")
    ds_images = load_dataset(calibration_dataset, "train_balanced_images", split="train")

    # 1. 关键：构建 imageId -> PIL.Image 映射字典，解决你没取图的核心问题
    img_id2pil = {}
    logger.info("Building image id to PIL image mapping...")
    for img_item in tqdm(ds_images, desc="Load all GQA images"):
        img_id = img_item["id"]
        img = img_item["image"]
        img_id2pil[img_id] = img
    logger.info(f"Total unique images loaded: {len(img_id2pil)}")

    data_len = len(ds_instructions)
    ds_instructions_sample = ds_instructions.shuffle(seed=42).select(range(max_calib_samples))

    # 校准chunk改为存储 (image: PIL.Image, prompt_text: str)
    multimodal_chunks = []
    pbar = tqdm(total=max_calib_samples, desc="Building GQA calibration chunks", unit="sample")
    
    for i in range(len(ds_instructions_sample)):
        item = ds_instructions_sample[i]
        # 提取GQA图文问答
        imageId = item["imageId"]
        fullAnswer = item["fullAnswer"]
        question = item["question"]

        # 修正变量名判断
        if not question or not fullAnswer:
            pbar.update(1)
            continue
        # 图片不存在直接跳过
        if imageId not in img_id2pil:
            logger.warning(f"imageId {imageId} not found in image dataset, skip")
            pbar.update(1)
            continue
        # 真正取出对应图片
        img = img_id2pil[imageId]
        images = [img]
        
        # 构造多模态对话消息（标准格式，适配Qwen/LLaVA等多模态processor）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": fullAnswer}]
            }
        ]
        # 渲染带视觉标记的完整prompt文本
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        trim_suffix = "<|message_start|>助手：</think>"
        if text.endswith(trim_suffix):
            text = text[:-len(trim_suffix)]
        if not isinstance(text, list):
            text = [text]
        print("text: ", text)

        # 统计文本token长度（图片token固定由vision encoder处理，不计入text上下文限制）
        inputs = processor(text=text, images=images, padding=False, return_tensors="pt",)
        # print("inputs: ", inputs)

        # GQA规则：1图=1独立chunk，不拼接多张图，只要文本不超长直接存入
        multimodal_chunks.append(inputs)
        pbar.update(1)

    pbar.close()
    logger.info("GQA multimodal calibration loaded, total %d image-text samples", len(multimodal_chunks))
    # 返回元组列表：[(PIL_Image, prompt_str), ...] 供量化校准器读取
    return multimodal_chunks


def load_calibration_chartqa_data(
    processor,
    calibration_dataset: str = "/data1/jiangyikun/datasets/ChartQA/",
    max_calib_samples: int = 128,
    max_context_len: int = 2048,
):
    logger.info("Loading calibration dataset: %s", calibration_dataset)
    
    ds = load_dataset(calibration_dataset, split="test")

    data_len = len(ds)
    ds_sample = ds.shuffle(seed=42).select(range(max_calib_samples))

    # 校准chunk改为存储 (image: PIL.Image, prompt_text: str)
    multimodal_chunks = []
    pbar = tqdm(total=max_calib_samples, desc="Building ChartQA calibration chunks", unit="sample")
    
    for i in range(len(ds_sample)):
        item = ds_sample[i]
        # 提取GQA图文问答
        answer = item["answer"]
        question = item["question"]
        images = [item['image']]
        
        # 构造多模态对话消息（标准格式，适配Qwen/LLaVA等多模态processor）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        # 渲染带视觉标记的完整prompt文本
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        trim_suffix = "<|message_start|>助手：</think>"
        if text.endswith(trim_suffix):
            text = text[:-len(trim_suffix)]
        if not isinstance(text, list):
            text = [text]
        print("text: ", text)

        # 统计文本token长度（图片token固定由vision encoder处理，不计入text上下文限制）
        inputs = processor(text=text, images=images, padding=False, return_tensors="pt",)
        # print("inputs: ", inputs)

        # GQA规则：1图=1独立chunk，不拼接多张图，只要文本不超长直接存入
        multimodal_chunks.append(inputs)
        pbar.update(1)

    pbar.close()
    logger.info("ChartQA multimodal calibration loaded, total %d image-text samples", len(multimodal_chunks))
    # 返回元组列表：[(PIL_Image, prompt_str), ...] 供量化校准器读取
    return multimodal_chunks


def load_calibration_cogvlm_data(
    processor,
    calibration_dataset: str = "/data1/jiangyikun/datasets/CogVLM-SFT-311K/",
    max_calib_samples: int = 128,
    max_context_len: int = 2048,
):
    logger.info("Loading calibration dataset: %s", calibration_dataset)
    
    image_path = "/data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/images"
    labels_en_path = "/data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/labels_en"
    labels_zh_path = "/data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/labels_zh"
    
    image_list = sorted(os.listdir("/data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/images"))

    image_sample = image_list[:max_calib_samples]

    # 校准chunk改为存储 (image: PIL.Image, prompt_text: str)
    multimodal_chunks = []
    pbar = tqdm(total=max_calib_samples, desc="Building CogVLM calibration chunks", unit="sample")
    
    for i in range(len(image_sample)):
        image = Image.open(os.path.join(image_path, image_list[i]))
        with open(os.path.join(labels_en_path, image_list[i][:-4] + ".json"), "r") as f:
            label = json.load(f)
        # 提取GQA图文问答
        answer = label["conversations"][1]["content"]
        question = label["conversations"][0]["content"]
        images = [image]
        
        # 构造多模态对话消息（标准格式，适配Qwen/LLaVA等多模态processor）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        # 渲染带视觉标记的完整prompt文本
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        trim_suffix = "<|message_start|>助手：</think>"
        if text.endswith(trim_suffix):
            text = text[:-len(trim_suffix)]
        if not isinstance(text, list):
            text = [text]
        print("text: ", text)

        # 统计文本token长度（图片token固定由vision encoder处理，不计入text上下文限制）
        inputs = processor(text=text, images=images, padding=False, return_tensors="pt",)
        print("inputs: ", inputs)

        # GQA规则：1图=1独立chunk，不拼接多张图，只要文本不超长直接存入
        multimodal_chunks.append(inputs)
        pbar.update(1)

    pbar.close()
    logger.info("ChartQA multimodal calibration loaded, total %d image-text samples", len(multimodal_chunks))
    # 返回元组列表：[(PIL_Image, prompt_str), ...] 供量化校准器读取
    return multimodal_chunks


def _get_moe_layer_indices(model) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.language_model.layers)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
    ]


def _get_moe_layers(model) -> list[tuple]:
    moe_layers = []
    for i, layer in enumerate(model.language_model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_layers.append((i, layer.mlp))
    return moe_layers



def patched_pangu_vl_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    **kwargs,
):
    """Patched OpenPanguVLModel.forward — creates modality masks and sets on MoE blocks."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # ---- Visual embedding injection (same as original) ----
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if self.use_mhc:
            inputs_embeds = inputs_embeds.repeat(1, 1, self.mhc_num_stream)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            n_image_tokens = (input_ids == self.config.image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"tokens: {n_image_tokens}, features {n_image_features}"
                )
            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: "
                    f"tokens: {n_video_tokens}, features {n_video_features}"
                )
            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # ---- Position IDs (same as original) ----
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_noncompiled = (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if prefill_noncompiled or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ====================================================================
    #  MoDES: Create modality masks and set on MoE blocks
    # ====================================================================

    if input_ids is not None:
        moe_visual_mask = (
            (input_ids == self.config.image_token_id)
            | (input_ids == self.config.video_token_id)
        ).view(-1)  # [B*S]

        # "text" = genuine content tokens that are NOT special/structural
        # and NOT visual (image_pad / video_pad).
        # KEY FIX: we exclude visual tokens AND special tokens from text,
        # so structural tokens like <|message_start|>, <|vision_start|>
        # etc. are classified as "other" (neither text nor visual) and
        # are ALWAYS kept (never skipped by either modality).
        moe_text_mask = ~moe_visual_mask # [B*S]

        # ---- DEBUG: Log mask statistics ----
        n_total = input_ids.numel()
        n_visual = moe_visual_mask.sum().item()
        n_text = moe_text_mask.sum().item()
        logger.info(f"[MoDES mask] total={n_total}, visual={n_visual}, text={n_text}")

        # Padding mask (exclude padding from expert computation)
        if input_ids.shape[-1] != 1 and attention_mask is not None:
            moe_padding_mask = (~attention_mask.to(torch.bool)).view(-1)  # [B*S]
        else:
            moe_padding_mask = None

        # Set on all MoE layers
        for layer in self.language_model.layers:
            if hasattr(layer.mlp, 'experts'):  # MoE layer
                mlp = layer.mlp
                mlp._moe_text_mask = moe_text_mask[:, None]    # [B*S, 1]
                mlp._moe_visual_mask = moe_visual_mask[:, None] # [B*S, 1]
                if moe_padding_mask is not None:
                    mlp._moe_text_mask = moe_text_mask[:, None] & (~moe_padding_mask[:, None])
                    mlp._moe_visual_mask = moe_visual_mask[:, None] & (~moe_padding_mask[:, None])
                    mlp._moe_padding_mask = moe_padding_mask[:, None]

    # ====================================================================
    #  MoDES: Set layer-skip / tau-skip flags on MoE blocks
    #  NOTE: We only overwrite these from kwargs during calibration.
    #  During inference (model.generate), kwargs are not passed, so
    #  the flags already set by load_modes_state() are preserved.
    # ====================================================================
    for layer_idx, layer in enumerate(self.language_model.layers):
        if hasattr(layer.mlp, 'experts'):  # MoE layer
            mlp = layer.mlp
            mlp._threshold = {
                'visual': threshold_state['visual'][layer_idx],
                'text': threshold_state['text'][layer_idx]
            }

    # ---- Call language model (unchanged interface) ----
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    try:
        from modeling_openpangu_vl import OpenPanguVLModelOutputWithPast
    except ImportError:
        import importlib
        _mod = importlib.import_module(self.__class__.__module__)
        OpenPanguVLModelOutputWithPast = getattr(_mod, 'OpenPanguVLModelOutputWithPast')
    output = OpenPanguVLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def calib_scale(model, processor, layer_topk=[], save_path="state_scale.pth"):
    model.model.forward = types.MethodType(patched_pangu_vl_model_forward, model.model)

    model.eval()
    moe_layers = _get_moe_layers(model)
    num_experts = model.config.n_routed_experts
    num_moe_layers = len(moe_layers)
    
    print(f"Using per-layer topK: {layer_topk}")

    A_stats: dict[int, torch.Tensor] = {}
    B_stats: dict[int, torch.Tensor] = {}

    active = {
        "moe_mask": [],
        "text_mask": [],
        "visual_mask": []
    }

    for (decoder_layer_idx, block), k_eff in zip(moe_layers, layer_topk):
        def patched_sparse_moe_forward(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx, _k=k_eff):
            residuals = hidden_states
            bsz, seq_len, hidden_dim = hidden_states.shape
            hidden_reshaped = hidden_states.view(-1, hidden_dim)
            num_tokens = hidden_reshaped.shape[0]

            router_logits = F.linear(hidden_reshaped.type(torch.float32), self_block.gate.weight.type(torch.float32))
            
            router_logits = router_logits.sigmoid()
            # router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

            router_indices = torch.topk(router_logits + self_block.e_score_correction_bias, self_block.top_k, dim=-1, sorted=True)[1]
            orig_routing_weights = router_logits.gather(1, router_indices)
            if self_block.norm_topk_prob:
                orig_routing_weights = orig_routing_weights / (orig_routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
            # MUST keep rsf here: alpha_orig (line ~780) feeds Y (least-squares target) and
            # final_output (propagated to next layers) DIRECTLY from orig_routing_weights,
            # BEFORE the norm_w re-normalization below cancels rsf. Removing rsf makes the
            # OptimalScale target = true_routed_output / rsf (0.4x), so S is solved 1/rsf too
            # small and inference attenuates the routed experts by 1/rsf. rsf is a scalar so
            # it still cancels in norm_w/cumsum/kept_routing_weights below — no other change.
            orig_routing_weights = orig_routing_weights * self_block.routed_scaling_factor

            # router_indices_kept = torch.topk(router_logits + self_block.e_score_correction_bias, _k, dim=-1, sorted=False)[1]
            # kept_routing_weights = router_logits.gather(1, router_indices_kept)
            # if self_block.norm_topk_prob:
            #     kept_routing_weights = kept_routing_weights / (kept_routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
            # kept_routing_weights = kept_routing_weights * self_block.routed_scaling_factor

            # 模拟 TopP 逻辑 (用于构建输入特征 V)
            # 与 calib_threshold_learn.py / patch_mask.py 对齐：TopP cumsum 前先按
            # raw weight 降序排序，避免 e_score_correction_bias 非常数时 choice-score
            # 顺序与 raw-weight 顺序不一致导致剪枝 mask 偏离训练语义。
            norm_w = orig_routing_weights / orig_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            norm_w_sorted, sort_idx = norm_w.sort(dim=-1, descending=True)
            cumsum_probs = norm_w_sorted.cumsum(dim=-1)

            # TopP under fixed threshold
            threshold_visual = self_block._threshold['visual']
            threshold_text = self_block._threshold['text']
            num_keep_visual = (cumsum_probs < threshold_visual).sum(dim=-1) + 1
            num_keep_text = (cumsum_probs < threshold_text).sum(dim=-1) + 1
            num_keep = num_keep_visual * self_block._moe_visual_mask.squeeze(1).type_as(num_keep_visual) + num_keep_text * self_block._moe_text_mask.squeeze(1).type_as(num_keep_text)
            num_keep = num_keep.clamp(min=1, max=self_block.top_k)

            # # TopP under threshold from FC
            # per_token_threshold = (
            #     self_block.threshold_fc(hidden_states.float())
            #     .sigmoid()
            #     .type_as(orig_routing_weights)
            #     .reshape(num_tokens,)
            # )
            # num_keep = (cumsum_probs < per_token_threshold.unsqueeze(-1)).sum(dim=-1) + 1
            # num_keep = num_keep.clamp(min=1, max=self_block.top_k)

            ########
            # active_mask 在 sorted 顺序下构造，再 un-sort 回 router_indices 顺序，
            # 保证下面 `torch.where(router_indices == expert_idx)` 与 kept_routing_weights 对齐。
            pos = torch.arange(self_block.top_k, device=router_indices.device).unsqueeze(0)
            active_mask_sorted = pos < num_keep.unsqueeze(1)
            active_mask = torch.zeros_like(active_mask_sorted)
            active_mask.scatter_(1, sort_idx, active_mask_sorted)

            active["moe_mask"].append(active_mask)
            active["visual_mask"].append(self_block._moe_visual_mask)
            active["text_mask"].append(self_block._moe_text_mask)
            
            kept_routing_weights = orig_routing_weights * active_mask.to(orig_routing_weights.dtype)
            if self_block.norm_topk_prob:
                kept_routing_weights = kept_routing_weights / (kept_routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
            kept_routing_weights = kept_routing_weights * self_block.routed_scaling_factor

            V = torch.zeros((num_tokens, num_experts, hidden_dim), dtype=torch.float32, device=hidden_states.device)
            Y = torch.zeros((num_tokens, hidden_dim), dtype=torch.float32, device=hidden_states.device)
            
            final_output = torch.zeros_like(hidden_reshaped)

            for expert_idx in range(num_experts):
                token_idx_orig, top_k_pos_orig = torch.where(router_indices == expert_idx)
                if token_idx_orig.numel() > 0:
                    current_state = hidden_reshaped[token_idx_orig]
                    gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    expert_output = F.linear(self_block.experts.act_fn(gate) * up, self_block.experts.down_proj[expert_idx])
                    alpha_orig = orig_routing_weights[token_idx_orig, top_k_pos_orig, None]
                    Y.index_add_(0, token_idx_orig, expert_output * alpha_orig)
                    final_output.index_add_(0, token_idx_orig, (expert_output * alpha_orig).to(final_output.dtype))

                    alpha_kept = kept_routing_weights[token_idx_orig, top_k_pos_orig, None]
                    V[token_idx_orig, expert_idx, :] = expert_output * alpha_kept
                
                # token_idx_kept, top_k_pos_kept = torch.where(router_indices_kept == expert_idx)
                # if token_idx_kept.numel() > 0:
                #     current_state = hidden_reshaped[token_idx_kept]
                #     gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                #     expert_output = F.linear(self_block.experts.act_fn(gate) * up, self_block.experts.down_proj[expert_idx])
                #     alpha_kept = kept_routing_weights[token_idx_kept, top_k_pos_kept, None]
                #     V[token_idx_kept, expert_idx, :] = expert_output * alpha_kept

            if _layer not in A_stats:
                A_stats[_layer] = torch.zeros((hidden_dim, num_experts, num_experts), dtype=torch.float64, device='cpu')
                B_stats[_layer] = torch.zeros((hidden_dim, num_experts), dtype=torch.float64, device='cpu')

            chunk_size = 640
            for start_d in range(0, hidden_dim, chunk_size):
                end_d = min(hidden_dim, start_d + chunk_size)
                V_chunk = V[:, :, start_d:end_d]
                Y_chunk = Y[:, start_d:end_d]
                
                A_stats[_layer][start_d:end_d] += torch.einsum('ted,tfd->def', V_chunk, V_chunk).cpu()
                B_stats[_layer][start_d:end_d] += torch.einsum('ted,td->de', V_chunk, Y_chunk).cpu()

            hidden_states = final_output.reshape(bsz, seq_len, hidden_dim)
            hidden_states = hidden_states + self_block.shared_experts(residuals)

            return hidden_states

        block.forward = types.MethodType(patched_sparse_moe_forward, block)

    # texts = load_calibration_text_data(processor, calibration_dataset="/data1/jiangyikun/datasets/slimpajama-200k")

    # for start in tqdm(range(0, len(texts)), total=len(texts), desc="Calibration Forward (os_lexi_skip)"):
    #     batch_texts = texts[start : start + 1]
    #     inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #     inputs["labels"] = inputs["input_ids"].clone()
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         print(outputs)

    # inputs = load_calibration_gqa_data(processor)
    # inputs = load_calibration_chartqa_data(processor)
    # inputs = load_calibration_cogvlm_data(processor)
    inputs = [
        *load_calibration_text_data(processor, max_calib_samples = 32, calibration_dataset="/data1/jiangyikun/datasets/slimpajama-200k"),
        *load_calibration_text_data(processor, max_calib_samples = 32, calibration_dataset="/data1/jiangyikun/datasets/CodeAlpaca-20k/"),
        *load_calibration_cogvlm_data(processor, max_calib_samples = 64),
        *load_calibration_chartqa_data(processor, max_calib_samples = 32),
        *load_calibration_text_data(processor, max_calib_samples = 32, calibration_dataset="/data1/jiangyikun/datasets/alpaca-data-gpt4-chinese"),
    ]
    for start in tqdm(range(0, len(inputs)), total=len(inputs), desc="Calibration Forward"):
        with torch.no_grad():
            outputs = model(**{ k:v.cuda() for k,v in inputs[start].items()})
            print(outputs)

    logger.info("Solving closed-form Normal Equations for S matrices...")
    state: dict[str, torch.Tensor] = {"meta.adapter_version": torch.tensor(1, dtype=torch.int32)}
    state["meta.layer_topp_visual"] = torch.tensor(threshold_state["visual"])
    state["meta.layer_topp_text"] = torch.tensor(threshold_state["text"])
    
    lambda_reg = 1e-4
    
    for layer_idx in A_stats.keys():
        A = A_stats[layer_idx]
        B = B_stats[layer_idx].unsqueeze(-1)
        
        eye = torch.eye(num_experts, dtype=torch.float64).unsqueeze(0)
        A_reg = A + lambda_reg * eye
        
        S_raw = torch.linalg.solve(A_reg, B).squeeze(-1)
        S_matrix = S_raw.transpose(0, 1)
        
        active_expert_mask = A.sum(dim=0).diagonal() > 1e-8
        S_matrix[~active_expert_mask, :] = 1.0
        
        state[f"layer_{layer_idx}.expert_S_matrix"] = S_matrix.float().contiguous()

    save_file(state, save_path)
    logger.info(f"Calibration completed. Adapters saved to {save_path}")

if __name__ == "__main__":

    # dataset = load_dataset("/data1/jiangyikun/datasets/CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/")
    # print(dataset)
    # for data in tqdm(dataset):
    #     print(data)
    #     break

    # processor = AutoProcessor.from_pretrained("/data5/jiangyikun/MoE/Pangu_VL-30A2B-dev_5.2.1.1_SP1_0616/HFModel/BF16", trust_remote_code=True)
    # data = load_calibration_cogvlm_data(processor, max_calib_samples=4)
    # exit(0)



    model_path = "/data5/jiangyikun/MoE/Pangu_VL-30A2B-dev_5.2.1.1_SP1_0616/HFModel/BF16"

    print(f"LOAD MODEL FROM: {model_path}")
    key_mapping = {
            r"^visual": "model.visual",
            r"^model(?!\.(language_model|visual|lm_head))": "model.language_model",}
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype="bfloat16",
            key_mapping=key_mapping
            )
    model.eval()

    print(model)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # state = calib_topk(model, processor)
    # torch.save(state, "state_topk_gqa.pth")

    state = torch.load("state_topk_gqa.pth")
    layer_topk = state[f"topk_{0.5}"]

    # threshold_state = {
    #     "visual": [1.0, 1.0, *[0.5] * 35],
    #     "text": [1.0, 1.0, *[0.6] * 35]
    # }

    # from calib_threshold_uniform import find_threshold
    # with open("/data5/jiangyikun/MoE/Pangu_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/threshold_state_uniform.pkl", "rb") as f:
    #     calib_result = pickle.load(f)
    #     calib_result = calib_result["sweep_data"]
    # th_v = find_threshold(calib_result, 0.5, modality="visual")
    # th_t = find_threshold(calib_result, 0.4, modality="text")
    # layer_topp_visual = torch.tensor([1.0, 1.0, *[th_v]*35])
    # layer_topp_text = torch.tensor([1.0, 1.0, *[th_t]*35])

    with open("/data5/jiangyikun/MoE/Pangu_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/thresholds_learned_60_v4.pkl", "rb") as f:
        calib_result = pickle.load(f)
    # result = find_thresholds(calib_result, target_skip_visual=args.target_skip_visual, target_skip_text=args.target_skip_text)
    layer_topp_visual = torch.tensor([1.0, 1.0, *calib_result["thresholds_visual"].tolist()])
    layer_topp_text = torch.tensor([1.0, 1.0, *calib_result["thresholds_text"].tolist()])
    
    ### v1 ###
    # layer_topp_visual = torch.tensor([1.0, 1.0, 0.601, 0.598, 0.6, 0.6, 0.599, 0.602, 0.606, 0.61, 0.609, 0.608, 0.61, 0.61, 0.617, 0.608, 0.612, 0.62, 0.625, 0.633, 0.607, 0.611, 0.615, 0.606, 0.602, 0.605, 0.602, 0.603, 0.612, 0.604, 0.606, 0.604, 0.605, 0.606, 0.603, 0.603, 0.588])
    # layer_topp_text = torch.tensor([1.0, 1.0, 0.631, 0.622, 0.623, 0.618, 0.637, 0.618, 0.617, 0.634, 0.635, 0.631, 0.648, 0.634, 0.638, 0.646, 0.616, 0.611, 0.689, 0.67, 0.654, 0.664, 0.66, 0.711, 0.655, 0.683, 0.694, 0.73, 0.698, 0.779, 0.738, 0.734, 0.776, 0.771, 0.82, 0.822, 0.742])
    ### v2 ###
    # layer_topp_visual = torch.tensor([1.0, 1.0, 0.591, 0.583, 0.589, 0.585, 0.576, 0.59, 0.6, 0.597, 0.598, 0.598, 0.608, 0.613, 0.625, 0.598, 0.611, 0.623, 0.638, 0.636, 0.603, 0.61, 0.625, 0.612, 0.611, 0.608, 0.598, 0.604, 0.633, 0.619, 0.616, 0.623, 0.633, 0.63, 0.616, 0.596, 0.578])
    # layer_topp_text = torch.tensor([1.0, 1.0, 0.434, 0.421, 0.431, 0.425, 0.476, 0.519, 0.456, 0.624, 0.515, 0.476, 0.583, 0.521, 0.542, 0.627, 0.549, 0.606, 0.782, 0.753, 0.627, 0.726, 0.748, 0.797, 0.768, 0.764, 0.796, 0.847, 0.811, 0.928, 0.827, 0.839, 0.878, 0.865, 0.921, 0.855, 0.799])
    ### v3 ###
    # layer_topp_visual = torch.tensor([1.0, 1.0, 0.599, 0.593, 0.601, 0.594, 0.599, 0.606, 0.604, 0.613, 0.62, 0.615, 0.627, 0.615, 0.624, 0.614, 0.614, 0.62, 0.624, 0.636, 0.606, 0.609, 0.615, 0.604, 0.601, 0.605, 0.6, 0.599, 0.607, 0.599, 0.603, 0.601, 0.601, 0.6, 0.598, 0.595, 0.583])
    # layer_topp_text = torch.tensor([1.0, 1.0, 0.626, 0.579, 0.627, 0.677, 0.651, 0.683, 0.672, 0.697, 0.78, 0.716, 0.737, 0.74, 0.741, 0.712, 0.652, 0.657, 0.675, 0.742, 0.726, 0.684, 0.622, 0.675, 0.627, 0.641, 0.637, 0.68, 0.641, 0.707, 0.636, 0.66, 0.664, 0.662, 0.734, 0.786, 0.703])
    ### v4 ###
    # layer_topp_visual = torch.tensor([1.0, 1.0, 0.597, 0.583, 0.613, 0.595, 0.602, 0.605, 0.607, 0.62, 0.627, 0.626, 0.666, 0.635, 0.65, 0.622, 0.622, 0.59, 0.596, 0.559, 0.601, 0.619, 0.605, 0.605, 0.599, 0.601, 0.598, 0.601, 0.614, 0.594, 0.595, 0.602, 0.606, 0.603, 0.597, 0.592, 0.581])
    # layer_topp_text = torch.tensor([1.0, 1.0, 0.664, 0.636, 0.688, 0.697, 0.742, 0.744, 0.684, 0.734, 0.831, 0.812, 0.84, 0.818, 0.804, 0.737, 0.743, 0.746, 0.716, 0.772, 0.811, 0.744, 0.718, 0.69, 0.641, 0.618, 0.616, 0.657, 0.582, 0.645, 0.597, 0.59, 0.548, 0.568, 0.644, 0.726, 0.546])
    
    threshold_state = {
        "visual": layer_topp_visual,
        "text": layer_topp_text,
    }

    # threshold_fc_state = torch.load("/data5/jiangyikun/MoE/Pangu_VL-30A2B-dev_5.2.1.1_SP1_0616/Prune/OptimalScale/thresholds_learned_fc_50_40_checkpoint_3.pth", map_location="cpu")
    # fc_indices = sorted({int(k.split(".")[1]) for k in threshold_fc_state if k.startswith("fc.")})
    # moe_layer_indices = [
    #     i for i, layer in enumerate(model.language_model.layers)
    #     if hasattr(layer.mlp, 'experts')
    # ]
    # n_moe = len(moe_layer_indices)
    # for i, layer_idx in enumerate(moe_layer_indices):
    #     mlp = model.language_model.layers[layer_idx].mlp

    #     # Build the fc layer in float32 (matches training) and load learned weights
    #     key_w = f"fc.{i}.weight"
    #     key_b = f"fc.{i}.bias"
    #     w = threshold_fc_state[key_w]
    #     device = next(mlp.parameters()).device
    #     fc = nn.Linear(w.shape[1], w.shape[0], bias=True).to(device=device)
    #     fc.weight.data.copy_(w.to(device=device))
    #     fc.bias.data.copy_(threshold_fc_state[key_b].to(device=device))
    #     mlp.threshold_fc = fc


    state = calib_scale(model, processor, layer_topk, save_path="state_scale_topp_layerwise_learn_60_mixed_v5_fixed.pth")
    # state = calib_scale(model, processor, layer_topk, save_path="state_scale_topp_fc_learn_50_40_v2.pth")

