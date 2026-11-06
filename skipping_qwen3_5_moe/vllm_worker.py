"""vLLM worker entry point; imported independently in every GPU process."""
import torch
import vllm
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

from .vllm_patch import install_compensation, validate_vllm_config


logger = init_logger(__name__)


class CompensationWorker(Worker):
    def load_model(self, *args, **kwargs):
        if vllm.__version__.split("+")[0] != "0.24.0":
            raise RuntimeError(f"CompensationWorker targets vLLM 0.24.0; found {vllm.__version__}")
        text_config = validate_vllm_config(self.vllm_config)
        options = self.vllm_config.additional_config["skipping_compensation"]
        super().load_model(*args, **kwargs)
        before = torch.cuda.memory_allocated(self.device)
        layers = install_compensation(
            self.model_runner.get_model(), options["checkpoint"], options["threshold"],
            text_config, self.parallel_config.tensor_parallel_size, tensor_model_parallel_all_reduce)
        # Account for the added FP32 parameters when vLLM sizes the KV cache.
        self.model_runner.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
        logger.info("Compensation installed: rank=%d layers=%d threshold=%.6f checkpoint=%s",
                    self.rank, len(layers), options["threshold"], options["checkpoint"])
