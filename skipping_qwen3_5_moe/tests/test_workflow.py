import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from ..config import read_config
from ..model import patch, load_compensation
from ..train import make_accelerator, train, evaluate, make_task
from .helpers import TinyModel, calibration_for, batch


def collate_one(samples):
    return samples[0]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_accumulated_training_evaluation_and_saved_compensation(self):
        config_path = Path(__file__).parents[1] / "configs" / "train.yaml"
        config = read_config(config_path)
        config["training"].update(max_steps=2, gradient_accumulation=2, warmup_steps=0,
                                  log_every=1, eval_every=2, save_every=1)
        model = TinyModel()
        baseline = copy.deepcopy(model)
        runtime = patch(model, calibration_for(model))
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = [p.detach().clone() for p in parameters]
        loader = DataLoader([batch()], batch_size=1, collate_fn=collate_one)
        validation = {"tiny": loader}
        with tempfile.TemporaryDirectory() as directory:
            config["output_dir"] = directory
            accelerator = make_accelerator(config["training"], cpu=True)
            steps = train(model, loader, validation, config, accelerator)
            self.assertEqual(steps, 2)
            self.assertTrue(all(not torch.equal(old, parameter) for old, parameter in zip(initial, parameters)))
            saved = Path(directory) / "compensation.safetensors"
            load_compensation(baseline, saved)
            expected = evaluate(make_task(model, config), validation)
            actual = evaluate(make_task(baseline, config), validation)
            self.assertAlmostEqual(actual["tiny"]["loss"], expected["tiny"]["loss"], places=5)
            report = json.loads((Path(directory) / "metrics" / "step-2.json").read_text())
            self.assertAlmostEqual(report["tiny"]["loss"], expected["tiny"]["loss"], places=5)
            lines = (Path(directory) / "train.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["step"] for line in lines], [1, 2])
            self.assertEqual(json.loads(lines[-1])["pruning_ratio"], 0.5)
            self.assertTrue((Path(directory) / "checkpoint-1.safetensors").exists())
            accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
