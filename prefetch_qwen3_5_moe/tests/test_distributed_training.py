"""用两个真实 CPU/Gloo 进程验证 DDP 与单进程的更新结果。"""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch as mock_patch

from safetensors.torch import load
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from ..patch import patch, save_prefetch
from ..train import evaluate, make_accelerator, read_config, train
from .test_training import training_model


def batches(count):
    result = []
    for index in range(count):
        ids = torch.tensor([[1, 6 + index] + [7, 8, 9][:1 + index % 3]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, :2] = False
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), router_mask=mask)
        if index % 2:
            batch.update(pixel_values=torch.ones(1, 3, 4, 4), image_grid_thw=torch.tensor([[1, 1, 3]]))
        result.append(batch)
    return result


def single_batch(examples):
    return examples[0]


def loader(records):
    return DataLoader(records, batch_size=1, collate_fn=single_batch)


def ddp_worker(rank, directory, config, sample_count, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      LOCAL_WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      OMP_NUM_THREADS="1", ACCELERATE_TORCH_DEVICE="cpu")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    accelerator = make_accelerator(config["training"], cpu=True)
    try:
        assert accelerator.num_processes == 2
        torch.manual_seed(123)
        model = training_model()
        patch(model, config["prefetch"]["mode_prefetch"])
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        with mock_patch("prefetch_qwen3_5_moe.train.save_prefetch", wraps=save_prefetch) as save:
            with mock_patch("prefetch_qwen3_5_moe.train.evaluate", wraps=evaluate) as evaluation:
                step = train(model, loader(batches(sample_count)), {"text": loader(batches(3))},
                             config, accelerator)
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        heads = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        torch.save(dict(heads=heads, calls=model.forward_calls, step=step,
                        saves=save.call_count, evaluations=evaluation.call_count),
                   Path(directory) / f"rank-{rank}.pt")
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


class DistributedTrainingTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU/Gloo")
    def test_ddp_matches_single_process(self):
        # 五个样本的情况验证 Accelerate 在数据末尾补齐各卡工作量。
        for mode, count in (([1, 0], 6), ([0, 1], 5)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = read_config(Path(__file__).resolve().parents[1] / "configs" / "train.yaml")
                config["output_dir"] = str(directory / "distributed")
                config["prefetch"].update(mode_prefetch=mode, ks=[1, 2, 4])
                config["training"].update(max_steps=3, gradient_accumulation=2,
                                          learning_rate=0.01, warmup_steps=1,
                                          log_every=1, eval_every=2, save_every=2)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                context = mp.spawn(ddp_worker, args=(str(directory), config, count, port),
                                   nprocs=2, join=False)
                deadline = time.monotonic() + 120
                try:
                    while not context.join(timeout=1):
                        if time.monotonic() > deadline:
                            self.fail("DDP workers did not finish within 120 seconds")
                finally:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)
                ranks = [torch.load(directory / f"rank-{rank}.pt", map_location="cpu") for rank in range(2)]
                self.assertEqual([rank["step"] for rank in ranks], [3, 3])
                self.assertEqual([rank["saves"] for rank in ranks], [2, 0])
                self.assertEqual([rank["evaluations"] for rank in ranks], [2, 0])
                samples = []
                for rank in ranks:
                    samples.append([call["tokens"][0][1] - 6 for call in rank["calls"]
                                    if call["grad_enabled"]])
                self.assertEqual(samples[0], [0, 2, 4, 0, 2, 4])
                expected_second = [1, 3, 5, 1, 3, 5]
                if count == 5:
                    expected_second = [1, 3, 0, 1, 3, 0]
                self.assertEqual(samples[1], expected_second)
                self.assertFalse(any(not call["grad_enabled"] for call in ranks[1]["calls"]))
                for name in ranks[0]["heads"]:
                    torch.testing.assert_close(ranks[0]["heads"][name], ranks[1]["heads"][name],
                                               rtol=0, atol=0)

                # 单进程累积 4 个样本，与 2 进程各累积 2 个样本的有效 batch 相同。
                reference = copy.deepcopy(config)
                reference["output_dir"] = str(directory / "single")
                reference["training"]["gradient_accumulation"] = 4
                records = batches(count)
                if count == 5:
                    records.append(records[0])
                torch.manual_seed(123)
                model = training_model()
                patch(model, mode)
                initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                accelerator = make_accelerator(reference["training"], cpu=True)
                train(model, loader(records), {"text": loader(batches(3))}, reference, accelerator)
                actual = load((directory / "distributed" / "gate_prefetch.safetensors").read_bytes())
                expected = load((directory / "single" / "gate_prefetch.safetensors").read_bytes())
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
                self.assertTrue(any(not torch.equal(p, initial[name]) for name, p in model.named_parameters()
                                    if p.requires_grad))
                actual_logs = [json.loads(line) for line in (directory / "distributed" / "train.jsonl").read_text().splitlines()]
                expected_logs = [json.loads(line) for line in (directory / "single" / "train.jsonl").read_text().splitlines()]
                self.assertEqual(len(actual_logs), 3)
                for actual_log, expected_log in zip(actual_logs, expected_logs):
                    self.assertEqual(actual_log["step"], expected_log["step"])
                    self.assertEqual(actual_log["learning_rate"], expected_log["learning_rate"])
                    self.assertAlmostEqual(actual_log["router_kl"], expected_log["router_kl"], places=6)
                for step in (2, 3):
                    actual_metrics = json.loads((directory / "distributed" / "metrics" / f"step-{step}.json").read_text())
                    expected_metrics = json.loads((directory / "single" / "metrics" / f"step-{step}.json").read_text())
                    self.assertEqual(actual_metrics["text"]["num_positions"], expected_metrics["text"]["num_positions"])
                    self.assertEqual(actual_metrics["text"]["avg_hits"], expected_metrics["text"]["avg_hits"])
                    self.assertAlmostEqual(actual_metrics["text"]["loss"], expected_metrics["text"]["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
