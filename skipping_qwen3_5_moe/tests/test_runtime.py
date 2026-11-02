import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ..config import configure_attention
from ..train import make_accelerator


class RuntimeTests(unittest.TestCase):
    def test_each_gpu_rank_uses_its_local_device(self):
        accelerator = SimpleNamespace(num_processes=2, device=torch.device("cuda:1"))
        with patch("accelerate.Accelerator", return_value=accelerator):
            with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1",
                                         "ACCELERATE_TORCH_DEVICE": "cuda:0"}):
                with patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "device_count", return_value=2), \
                        patch.object(torch.cuda, "set_device") as set_device:
                    actual = make_accelerator({"gradient_accumulation": 2}, expected_world_size=2)
                    self.assertIs(actual, accelerator)
                    self.assertEqual(os.environ["ACCELERATE_TORCH_DEVICE"], "cuda:1")
                    set_device.assert_any_call(1)
                    set_device.assert_any_call(torch.device("cuda:1"))

    def test_launch_process_count_is_verified(self):
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"))
        with patch("accelerate.Accelerator", return_value=accelerator):
            with self.assertRaisesRegex(RuntimeError, "Requested 2 training processes"):
                make_accelerator({"gradient_accumulation": 2}, cpu=True, expected_world_size=2)

    def test_sdpa_backend_configuration(self):
        convolution_enabled = torch.backends.cudnn.enabled
        with patch.object(torch.backends.cuda, "enable_cudnn_sdp", create=True) as select:
            configure_attention()
            select.assert_called_once_with(False)
        self.assertEqual(torch.backends.cudnn.enabled, convolution_enabled)


if __name__ == "__main__":
    unittest.main()
