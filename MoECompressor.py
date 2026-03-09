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

import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
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

        logger.info("Loading model: %s", model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loading tokenizer: %s", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        self.adapter_path = None
        self.adapter = None
        if self.adapter_dir is not None:
            self.adapter_path = self._get_adapter_path()
            if self.adapter_path.exists():
                logger.info("Loading adapter: %s", self.adapter_path)
                self.adapter = load_file(str(self.adapter_path))

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
        **kwargs,
    ) -> PreTrainedModel:
        """
        打补丁：读取 adapter，对 self.model 非侵入式地修改模型结构、权重和 forward 逻辑。

        通常在 eval 前由 run.py 根据 adapter_dir 是否传入自动调用，不作为独立用户动作暴露。
        返回的模型为标准 ModelForCausalLM，可直接用于 transformers 推理或传给 eval。

        Args:
            kwargs: 与压缩方法相关的参数。

        Returns:
            self.model: 打过补丁的 ModelForCausalLM
        """
        pass

    # -------------------------------------------------------------------------
    # 评测接口：基类实现，所有方法共用
    # -------------------------------------------------------------------------

    def eval(
        self,
        model: PreTrainedModel | None = None,
        tasks: list[str] | None = None,
        num_fewshot: int | dict[str, int] = 0,
        batch_size: int | str = 1,
        limit: float | None = None,
        **lm_eval_kwargs,
    ) -> dict[str, Any]:
        """
        评测：用 lm_eval 评估模型。

        内部将模型用 HFLM 封装，调用 simple_evaluate。所有压缩方法共用此流程。
        调用前若已通过 patch() 修改 self.model，则评测剪枝模型；否则评测原模型。

        Args:
            model: 待评测模型。若 None，使用 self.model（可能已被 patch 修改）
            tasks: 评测任务名列表，如 ["wikitext", "hellaswag"]
            num_fewshot: few-shot 数量
            batch_size: 评测 batch size，可为 "auto"
            limit: 每任务样本上限，如 0.1 表示 10%
            **lm_eval_kwargs: 传给 simple_evaluate 的额外参数

        Returns:
            lm_eval 的 results 字典
        """
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError("eval 需要 lm_eval: pip install lm_eval[hf]") from e

        model = model if model is not None else self.model
        tokenizer = self.tokenizer

        logger.info("Starting evaluation, tasks: %s", tasks or ["wikitext"])

        try:
            from lm_eval.models.utils import configure_pad_token
            tokenizer = configure_pad_token(tokenizer, model_config=model.config)
        except Exception:
            pass

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=self.device,
            batch_size=batch_size,
            dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        tasks = tasks or ["wikitext"]
        results = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            **lm_eval_kwargs,
        )
        return getattr(results, "results", results) if results is not None else {}

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------

    def load_calibration_data(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
    ) -> list[str]:
        """
        从 HuggingFace 数据集加载校准文本。

        针对 wikitext 等按换行粗暴切分的数据集（大量空字符串）：逐行拼接。
        当块内 token 数达到 max_context_len 时形成一个校准样本，直到得到 max_calib_samples 个样本。

        Args:
            calibration_dataset: 数据集路径，格式 "dataset" 或 "dataset:config"，
                如 "wikitext:wikitext-2-raw-v1"
            max_calib_samples: 最大校准样本数
            max_context_len: 每个样本的目标 token 数上限

        Returns:
            校准文本块列表，每块约 max_context_len tokens
        """
        logger.info("Loading calibration dataset: %s", calibration_dataset)
        parts = calibration_dataset.split(":", 1)
        name = parts[1] if len(parts) > 1 else None
        ds = load_dataset(parts[0], name, split="train")

        col = "text" if "text" in ds.column_names else ds.column_names[0]
        raw = ds[col]
        lines = [t for t in raw if t and str(t).strip()]
        logger.info("Valid lines after filtering: %d, building chunks of ~%d tokens", len(lines), max_context_len)

        chunks = []
        current_lines = []
        pbar = tqdm(total=max_calib_samples, desc="Building calibration chunks", unit="chunk")
        for line in lines:
            line = (line if isinstance(line, str) else str(line)).strip()
            if not line:
                continue
            current_lines.append(line)
            combined = "\n".join(current_lines)
            n_tokens = len(self.tokenizer.encode(combined, add_special_tokens=False))
            if n_tokens >= max_context_len:
                chunks.append(combined)
                pbar.update(1)
                if len(chunks) >= max_calib_samples:
                    break
                current_lines = []

        pbar.close()
        logger.info("Calibration data loaded, %d chunks", len(chunks))
        return chunks
