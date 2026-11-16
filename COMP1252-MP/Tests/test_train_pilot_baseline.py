"""Training-loop gates for reviewed pixels and bounded CPU smoke updates."""

import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
from train_pilot_baseline import CountingOptimizer, make_model, train_step


class InspectedLogitsModel(nn.Module):
    """A tiny differentiable fixture with observable forward/logit gradients."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(27, 1, 1)
        with torch.no_grad():
            self.conv.weight.fill_(.01)
            self.conv.bias.zero_()
        self.forward_calls = 0
        self.forbid_forward = False
        self.nonfinite_output = False
        self.last_logits = None

    def forward(self, features):
        self.forward_calls += 1
        if self.forbid_forward:
            raise AssertionError("An ignored batch must not run the model")
        logits = self.conv(features)
        if self.nonfinite_output:
            logits = logits*float("nan")
        logits.retain_grad()
        self.last_logits = logits
        return logits


def example():
    features = torch.tensor([[[[1., 3.], [2., 4.]]]]).expand(1, 27, 2, 2).clone()
    labels = torch.tensor([[[1, 255], [0, 255]]], dtype=torch.uint8)
    return {"features": features, "labels": labels, "valid": labels != 255}


def snapshot(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


class BaselineTrainingStepTests(unittest.TestCase):
    def test_finite_reviewed_batch_changes_parameters_and_records_an_actual_step(self):
        model = InspectedLogitsModel()
        optimizer = CountingOptimizer(model.parameters(), lr=.05)
        before = snapshot(model)
        result = train_step(model, optimizer, example())
        self.assertTrue(result["updated"])
        self.assertEqual(result["reviewed_pixels"], 2)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertGreater(result["gradient_norm"], 0.)
        self.assertGreater(result["parameter_delta_l2"], 0.)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_ignored_logits_receive_exactly_zero_gradient_during_real_model_training(self):
        model = InspectedLogitsModel()
        optimizer = CountingOptimizer(model.parameters(), lr=.05)
        sample = example()
        train_step(model, optimizer, sample, pos_weight=2.)
        gradient = model.last_logits.grad[:, 0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[~sample["valid"]], torch.zeros(2)))
        self.assertTrue((gradient[sample["valid"]].abs() > 0).all())

    def test_all_ignored_batch_skips_forward_weight_decay_and_existing_momentum(self):
        model = InspectedLogitsModel()
        optimizer = CountingOptimizer(model.parameters(), lr=.05, momentum=.9, weight_decay=.2)
        train_step(model, optimizer, example())
        self.assertTrue(optimizer.optimizer.state)
        before = snapshot(model)
        old_calls, old_steps = model.forward_calls, optimizer.step_calls
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.forbid_forward = True
        ignored = example()
        ignored["labels"].fill_(255)
        ignored["valid"].fill_(False)
        ignored["features"].fill_(float("nan"))
        result = train_step(model, optimizer, ignored)
        self.assertEqual(result, {"updated": False, "loss": None, "reviewed_pixels": 0,
                                  "gradient_norm": 0., "parameter_delta_l2": 0.})
        self.assertEqual(model.forward_calls, old_calls)
        self.assertEqual(optimizer.step_calls, old_steps)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_malformed_validity_and_targets_fail_before_forward_or_optimizer_step(self):
        for malformed in ("validity", "label", "boolean_labels"):
            with self.subTest(malformed=malformed):
                model = InspectedLogitsModel()
                optimizer = CountingOptimizer(model.parameters(), lr=.05)
                sample = example()
                if malformed == "validity":
                    sample["valid"].fill_(True)
                elif malformed == "label":
                    sample["labels"][0, 0, 0] = 2
                else:
                    sample["labels"] = sample["labels"] > 0
                before = snapshot(model)
                with self.assertRaises(ValueError):
                    train_step(model, optimizer, sample)
                self.assertEqual(model.forward_calls, 0)
                self.assertEqual(optimizer.step_calls, 0)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_nonfinite_feature_even_at_ignored_location_cannot_enter_a_convolution(self):
        model = InspectedLogitsModel()
        optimizer = CountingOptimizer(model.parameters(), lr=.05)
        sample = example()
        sample["features"][0, 0, 0, 1] = float("nan")
        before = snapshot(model)
        with self.assertRaisesRegex((ValueError, RuntimeError), "finite"):
            train_step(model, optimizer, sample)
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_nonfinite_model_output_never_updates_weights(self):
        model = InspectedLogitsModel()
        model.nonfinite_output = True
        optimizer = CountingOptimizer(model.parameters(), lr=.05)
        before = snapshot(model)
        with self.assertRaisesRegex((ValueError, RuntimeError), "finite"):
            train_step(model, optimizer, example())
        self.assertEqual(optimizer.step_calls, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_nonfinite_backward_gradient_is_detected_before_optimizer_step(self):
        model = InspectedLogitsModel()
        optimizer = CountingOptimizer(model.parameters(), lr=.05)
        handle = model.conv.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        before = snapshot(model)
        try:
            with self.assertRaisesRegex((ValueError, RuntimeError), "finite"):
                train_step(model, optimizer, example())
        finally:
            handle.remove()
        self.assertEqual(optimizer.step_calls, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_factory_creates_a_small_fresh_cpu_model_without_loading_checkpoint(self):
        with patch("torch.load", side_effect=AssertionError("No existing checkpoint should be loaded")):
            first, second = make_model(), make_model()
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in first.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in first.parameters()))
        self.assertLessEqual(sum(parameter.numel() for parameter in first.parameters()), 10_000)
        self.assertTrue(set(map(id, first.parameters())).isdisjoint(set(map(id, second.parameters()))))
        output = first(torch.zeros((1, 27, 4, 4), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
