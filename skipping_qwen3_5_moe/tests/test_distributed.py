"""Two CPU/Gloo ranks compared with the same effective single-process batch."""
import copy
from datetime import timedelta
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from ..config import read_config
from ..model import patch
from ..train import make_accelerator, train
from .helpers import TinyModel, calibration_for, batch
from .test_workflow import collate_one


def examples():
    samples = []
    for index in range(4):
        sample = batch()
        sample["input_ids"] = (sample["input_ids"] + index) % 19
        samples.append(sample)
    return samples


def make_model():
    torch.manual_seed(123)
    model = TinyModel()
    patch(model, calibration_for(model))
    return model


def ddp_worker(rank, directory, config, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1",
                      ACCELERATE_TORCH_DEVICE="cpu")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    accelerator = make_accelerator(config["training"], cpu=True)
    try:
        model = make_model()
        loader = DataLoader(examples(), batch_size=1, collate_fn=collate_one)
        train(model, loader, {}, config, accelerator)
        parameters = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
        torch.save(parameters, Path(directory) / f"rank-{rank}.pt")
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


class DistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU/Gloo")
    def test_ddp_matches_single_process_accumulation(self):
        torch.set_num_threads(1)
        config = read_config(Path(__file__).parents[1] / "configs" / "train.yaml")
        config["training"].update(max_steps=2, gradient_accumulation=2, warmup_steps=0,
                                  log_every=2, eval_every=2, save_every=2)
        with tempfile.TemporaryDirectory() as directory:
            config["output_dir"] = str(Path(directory) / "distributed")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            context = mp.spawn(ddp_worker, args=(directory, config, port), nprocs=2, join=False)
            deadline = time.monotonic() + 100
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("DDP test exceeded 100 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            ranks = [torch.load(Path(directory) / f"rank-{rank}.pt", map_location="cpu") for rank in range(2)]
            for name in ranks[0]:
                torch.testing.assert_close(ranks[0][name], ranks[1][name], rtol=0, atol=0)
            single = copy.deepcopy(config)
            single["output_dir"] = str(Path(directory) / "single")
            single["training"]["gradient_accumulation"] = 4
            model = make_model()
            accelerator = make_accelerator(single["training"], cpu=True)
            train(model, DataLoader(examples(), batch_size=1, collate_fn=collate_one), {}, single, accelerator)
            for name, value in model.named_parameters():
                if value.requires_grad:
                    torch.testing.assert_close(value, ranks[0][name], rtol=1e-5, atol=1e-6)
            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
