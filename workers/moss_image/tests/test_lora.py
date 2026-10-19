"""Unmerged adapters preserve base weights and leave no state after failure."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))


class LoRATests(unittest.TestCase):
    def weights(self, name='0'):
        return {f'diffusion_model.{name}.lora_down.weight': torch.tensor([[2., 0.]]),
                f'diffusion_model.{name}.lora_up.weight': torch.tensor([[3.], [4.]]),
                f'diffusion_model.{name}.alpha': torch.tensor(1.)}

    def test_unmerged_math_and_cleanup_across_turbo_base_and_exception(self):
        from moss_image_infer.lora import activate_lora
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data.copy_(torch.eye(2))
        original = model[0]
        x = torch.tensor([[1., 2.]])
        for failure in (False, True):
            try:
                with activate_lora(model, self.weights(), strength=.5):
                    torch.testing.assert_close(model(x), torch.tensor([[4., 6.]]))
                    torch.testing.assert_close(original.weight, torch.eye(2))
                    if failure:
                        raise RuntimeError('decode failed')
            except RuntimeError:
                self.assertTrue(failure)
            self.assertIs(model[0], original)
            torch.testing.assert_close(model(x), x)

    def test_validates_every_target_before_patching_any_layer(self):
        from moss_image_infer.lora import activate_lora, LoRAError
        model = nn.Sequential(nn.Linear(2, 2))
        original = model[0]
        cases = [self.weights('missing'), {**self.weights(), 'junk': torch.tensor(0.)},
                 {k: v for k, v in self.weights().items() if 'up.weight' not in k},
                 {**self.weights(), 'diffusion_model.0.lora_up.weight': torch.ones(3, 1)},
                 {**self.weights(), 'diffusion_model.0.alpha': torch.tensor(float('nan'))}]
        for state in cases:
            with self.subTest(keys=list(state)), self.assertRaises(LoRAError):
                with activate_lora(model, state, strength=.7):
                    self.fail('Invalid LoRA reached inference')
            self.assertIs(model[0], original)

