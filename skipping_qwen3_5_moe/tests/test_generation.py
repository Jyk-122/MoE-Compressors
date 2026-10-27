import copy
import unittest

import torch

from ..model import patch
from .helpers import calibration_for, tiny_generation_model


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_native_generate_threshold_scope_and_cached_decode(self):
        model = tiny_generation_model()
        baseline = copy.deepcopy(model)
        runtime = patch(model, calibration_for(model))
        ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mask = ids.ne(0)
        kwargs = dict(attention_mask=mask, max_new_tokens=3, do_sample=False,
                      return_dict_in_generate=True, output_scores=True)
        expected = baseline.generate(ids, **kwargs)
        actual = model.generate(ids, threshold=1, **kwargs)
        torch.testing.assert_close(actual.sequences, expected.sequences)
        for a, b in zip(actual.scores, expected.scores):
            torch.testing.assert_close(a, b)
        self.assertEqual(runtime.threshold, 0.5)
        self.assertEqual(runtime.summary()["pruning_ratio"], 0)
        # Original top-k=2 in this tiny HF model: threshold .5 retains one.
        generated = model.generate(ids, threshold=0.5, **kwargs)
        with torch.no_grad():
            full = model(generated.sequences[:, :-1],
                         attention_mask=torch.cat((mask, torch.ones(2, 2, dtype=torch.bool)), -1),
                         threshold=0.5).logits
        for i, score in enumerate(generated.scores):
            torch.testing.assert_close(score, full[:, ids.shape[1] - 1 + i], rtol=1e-4, atol=1e-6)
        self.assertEqual(runtime.summary()["pruning_ratio"], 0.5)
        self.assertEqual(runtime.generation_depth, 0)


if __name__ == "__main__":
    unittest.main()
