"""The current frozen factory must use exactly a two-pixel local input context."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from train_pilot_baseline import make_model


class ValidationContextArchitectureTests(unittest.TestCase):
    def test_actual_factory_output_gradient_is_exactly_inside_five_by_five_context(self):
        model = make_model()
        # Positive synthetic weights/input keep every ReLU path active. This
        # avoids a vacuous zero gradient hiding a larger receptive field.
        # No saved project weights, satellite input or optimizer is involved.
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.fill_(.01)
                    layer.bias.fill_(.1)
        inputs = torch.ones((1, 27, 64, 64), dtype=torch.float32,
                            device='cpu', requires_grad=True)
        model(inputs)[0, 0, 32, 32].backward()
        gradient = inputs.grad[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        context = torch.zeros((64, 64), dtype=torch.bool, device='cpu')
        context[30:35, 30:35] = True
        self.assertTrue(bool((gradient[:, ~context] == 0).all()))
        self.assertTrue(bool((gradient[:, context] > 0).all()))


if __name__ == '__main__':
    unittest.main()
