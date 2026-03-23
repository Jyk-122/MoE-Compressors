"""
MoE Compressor - MoE 剪枝/合并方法的抽象基类

【框架设计思路】
本框架用于统一各种 training-free 的 MoE 压缩方法（如剪枝、合并）。设计目标：
1. 轻量存储：每种方法仅保存 adapter（剪枝/合并所需的修改量），不保存整份 MoE 权重
2. 非侵入式 patch：通过读取 adapter 动态修改模型，返回标准 ModelForCausalLM，可与 transformers 直接推理
3. 统一评测：eval 接口在基类中实现，所有方法复用同一套 lm_eval 评测流程

【运行流程】
- calib：单卡校准，在校准集上计算统计量，保存 adapter.safetensors
- eval：多卡评测。若 adapter_dir 非空则先 patch 再评测剪枝模型；否则评测原模型
- patch 不作为独立步骤暴露，由 eval 在需要时内部调用
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class MoECompressor(ABC):
    """
    MoE 压缩方法抽象基类。

    各方法子类需实现 calib、patch。eval 在基类实现。
    """

    ADAPTER_FILENAME = "adapter.safetensors"

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: str | Path | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: 基座模型路径或 HuggingFace 模型名
            adapter_dir: adapter 目录。None 表示仅评估原模型，不涉及剪枝
            device: 推理设备
            torch_dtype: 模型 dtype，默认 bfloat16
            trust_remote_code: 是否信任远程代码
        """
        self.model_name_or_path = model_name_or_path
        self.adapter_dir = Path(adapter_dir) if adapter_dir is not None else None
        self.device = device
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs

        # adapter 路径，patch 时按需加载
        self.adapter_path = self._get_adapter_path() if self.adapter_dir is not None else None

    def _get_adapter_path(self) -> Path:
        """返回 adapter.safetensors 的完整路径。"""
        return self.adapter_dir / self.ADAPTER_FILENAME

    # -------------------------------------------------------------------------
    # 抽象接口：子类必须实现
    # -------------------------------------------------------------------------

    @abstractmethod
    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上计算统计量，并保存 adapter，需要指定 adapter_dir。

        Args:
            calibration_dataset: HuggingFace 数据集路径，格式如 "wikitext:wikitext-2-raw-v1"
            max_calib_samples: 最大校准样本数
            max_context_len: 每个校准样本的目标 token 数
            batch_size: 批量大小
            **kwargs: 其它方法相关的参数

        Returns:
            None
        """
        pass

    @abstractmethod
    def patch(
        self,
        model: PreTrainedModel,
        **kwargs,
    ) -> PreTrainedModel:
        """
        打补丁：读取 adapter（若有），对给定 model 原地修改 MoE 层结构、权重和 forward 逻辑。

        由 eval() 在 HFLM 初始化后、simple_evaluate 前调用，对 lm._model 做 patch。

        Args:
            model: 待 patch 的 ModelForCausalLM（通常为 HFLM._model）
            **kwargs: 由 eval 将 **patch_kwargs（解析后的 dict）传入 patch（如 pruning 的 prune_ratio、skipping 的 k）。

        Returns:
            model: 打过补丁的 model（原地修改，返回同一对象）
        """
        pass

    # -------------------------------------------------------------------------
    # 评测接口：基类实现，所有方法共用
    # -------------------------------------------------------------------------

    def eval(
        self,
        tasks: list[str] | None = None,
        num_fewshot: int | None = None,
        batch_size: int | str = 1,
        limit: float | None = None,
        gen_kwargs: str | dict | None = None,
        patch_kwargs: dict | None = None,
        **lm_eval_kwargs,
    ) -> dict[str, Any]:
        """
        评测：用 lm_eval 评估模型。

        使用 HFLM(pretrained=model_path) 传入路径，以支持 accelerate 分布式数据并行。
        若 `adapter_dir` 非空或 `patch_kwargs` 非空，则对 lm._model 调用 patch 后再评测。

        Args:
            tasks: 评测任务名列表，如 ["wikitext", "hellaswag"]
            num_fewshot: few-shot 数量
            batch_size: 评测 batch size，可为 "auto"
            limit: 每任务样本上限，如 0.1 表示 10%
            gen_kwargs: 生成参数，如 "max_gen_toks=1024" 或 dict，对 generate_until 任务生效
            patch_kwargs: 传给 patch(**patch_kwargs) 的参数字典（如 skipping 的 k）；剪枝默认可为空
            **lm_eval_kwargs: 传给 simple_evaluate 的额外参数

        Returns:
            lm_eval 的 results 字典
        """
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError("eval 需要 lm_eval: pip install lm_eval[hf]") from e

        logger.info("Starting evaluation, tasks: %s", tasks or ["wikitext"])
        logger.info("Loading model via HFLM (pretrained=%s) for distributed eval", self.model_name_or_path)

        # 忽略 datasets 和 httpx 的警告
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        lm = HFLM(
            pretrained=self.model_name_or_path,
            batch_size=batch_size,
            device=self.device,
            dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # 避免跨次 eval 复用上一次 patch 写入的统计对象
        self._acceleration_stats_collector = None
        should_patch = self.adapter_dir is not None or (
            patch_kwargs is not None and len(patch_kwargs) > 0
        )
        if should_patch:
            logger.info("[eval] Applying patch to lm._model")
            self.patch(lm._model, **(patch_kwargs or {}))

        tasks = tasks or ["wikitext"]
        results = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            confirm_run_unsafe_code=True,
            gen_kwargs=gen_kwargs,
            **lm_eval_kwargs,
        )
        
        collector = getattr(self, "_acceleration_stats_collector", None)
        if collector is not None and isinstance(results, dict):
            summary = collector.distributed_summary()
            results["runtime_routing"] = {
                **summary,
                "patch_kwargs": patch_kwargs or {},
            }
        return results

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------

    def load_calibration_data(
        self,
        tokenizer,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
    ) -> list[str]:
        """
        从 HuggingFace 数据集加载校准文本。
        支持 WikiText (纯文本) 和 Alpaca (指令微调) 格式的校准数据加载。
        针对 wikitext 等按换行粗暴切分的数据集（大量空字符串）：逐行拼接。
        当块内 token 数达到 max_context_len 时形成一个校准样本，直到得到 max_calib_samples 个样本。

        Args:
            tokenizer: 用于 encode 的 tokenizer，由 calib 加载后传入
            calibration_dataset: 数据集路径，格式 "dataset" 或 "dataset:config"，
                如 "wikitext:wikitext-2-raw-v1"
            max_calib_samples: 最大校准样本数
            max_context_len: 每个样本的目标 token 数上限

        Returns:
            校准文本块列表，每块约 max_context_len tokens
        """
        logger.info("Loading calibration dataset: %s", calibration_dataset)
        parts = calibration_dataset.split(":", 1)
        dataset_name = parts[0]
        config_name = parts[1] if len(parts) > 1 else None
        
        ds = load_dataset(dataset_name, config_name, split="train")

        # --- 1. 数据格式标准化 ---
        formatted_texts = []
        
        # 判断是否为 Alpaca 格式 (包含 instruction 字段)
        if "instruction" in ds.column_names:
            logger.info("Detected Alpaca-style dataset. Formatting to chat messages...")
            for item in ds:
                # 合并 instruction 和 input
                user_content = item["instruction"]
                if "input" in item and item["input"]:
                    user_content += "\n" + item["input"]
                
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item.get("output", "")}
                ]
                
                # 使用 tokenizer 的 chat_template 转换为最终文本
                # tokenize=False 返回字符串，add_generation_prompt=False 因为我们已经有输出了
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
            n_tokens = len(tokenizer.encode(current_chunk_text, add_special_tokens=False))
            
            if n_tokens >= max_context_len:
                chunks.append(current_chunk_text)
                current_chunk_text = "" # 重置
                pbar.update(1)
                
                if len(chunks) >= max_calib_samples:
                    break

        pbar.close()
        logger.info("Calibration data loaded, %d chunks", len(chunks))
        return chunks
