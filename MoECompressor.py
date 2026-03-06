"""
MoE Compressor - MoE 剪枝/合并方法的抽象基类

【框架设计思路】
本框架用于统一各种 training-free 的 MoE 压缩方法（如剪枝、合并）。设计目标：
1. 轻量存储：每种方法仅保存 adapter（剪枝/合并所需的修改量），不保存整份 MoE 权重
2. 非侵入式 patch：通过读取 adapter 动态修改模型，返回标准 ModelForCausalLM，可与 transformers 直接推理
3. 统一评测：eval 接口在基类中实现，所有方法复用同一套 lm_eval 评测流程

【接口约定】
- calib：子类实现。在校准集上计算统计量（如专家重要性），保存到 adapter.safetensors
- patch：子类实现。读取 adapter，修改模型结构/权重，返回可推理模型
- eval：基类实现。将 patch 后的模型用 HFLM 封装，调用 lm_eval.simple_evaluate 评测
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        self.adapter_path = None
        self.adapter = None
        if self.adapter_dir is not None:
            self.adapter_path = self._get_adapter_path()
            if self.adapter_path.exists():
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
        calibration_data: list[str] | None = None,
        calibration_dataset: str | None = None,
        max_calib_samples: int = 512,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上计算统计量，并保存adapter，需要指定adapter_dir。

        Args:
            calibration_data (list[str] | None): 可选，直接传入的校准文本数据。
            calibration_dataset (str | None): 可选，HuggingFace数据集名，格式如"wikitext:wikitext-2-raw-v1"。
            max_calib_samples (int): 最大校准样本数，默认512。
            batch_size (int): 批量大小，默认1。
            **kwargs: 其它方法相关的参数。

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
        打补丁：读取 adapter，对self.model非侵入式地修改模型结构、权重和 forward 逻辑。

        返回的模型为标准 ModelForCausalLM，可直接用于 transformers 推理或传给 eval。

        Args:
            kwargs: 与compression方法相关的参数。

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
        评测：用 lm_eval 评估 patch 后的模型。

        内部将模型用 HFLM 封装，调用 simple_evaluate。所有压缩方法共用此流程，
        保证评测方式一致。

        Args:
            model: patch 后的模型。若 None，会先调用 self.patch()
            adapter_path: adapter 路径。若 None，则表示直接评估原模型，否则评估剪枝模型。
            tasks: 评测任务名列表，如 ["wikitext", "hellaswag"]
            num_fewshot: few-shot 数量
            batch_size: 评测 batch size，可为 "auto"
            limit: 每任务样本上限，如 0.1 表示 10%
            device: 覆盖 self.device
            **lm_eval_kwargs: 传给 simple_evaluate 的额外参数

        Returns:
            lm_eval 的 results 字典
        """
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError("eval 需要 lm_eval: pip install lm_eval[hf]") from e

        if model is None:
            model = self.model
            tokenizer = self.tokenizer
        else:
            tokenizer = model.tokenizer
            
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
        calibration_data: list[str] | None = None,
        calibration_dataset: str | None = None,
        max_calib_samples: int = 512,
    ) -> list[str]:
        """
        加载校准文本。支持直接传入字符串列表，或从 HuggingFace 数据集读取。

        calibration_dataset 格式：若数据集需指定 config，用 "dataset:config"，
        如 "wikitext:wikitext-2-raw-v1"。
        """
        if calibration_data:
            return calibration_data[:max_calib_samples]

        if calibration_dataset:
            parts = calibration_dataset.split(":", 1)
            name = parts[1] if len(parts) > 1 else None
            ds = load_dataset(parts[0], name, split="train")
            if "text" in ds.column_names:
                texts = [t for t in ds["text"][:max_calib_samples] if t and t.strip()]
            else:
                col = ds.column_names[0]
                texts = [str(t) for t in ds[col][:max_calib_samples]]
            return texts
        raise ValueError("需提供 calibration_data 或 calibration_dataset")

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--adapter_dir", type=str, default=None)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--dtype", type=str, default="float16")
        return parser
