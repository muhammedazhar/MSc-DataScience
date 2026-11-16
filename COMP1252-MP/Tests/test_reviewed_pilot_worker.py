"""Protect predeclared scope and the new worker's no-forward ignore path."""

import copy
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
from pilot_overfit_worker import _CountingOptimizer
from reviewed_pilot_worker import train_step
from train_reviewed_pilot import validate_plan


class ReviewedPilotWorkerTests(unittest.TestCase):
    def test_plan_rejects_changes_to_exposure_mask_support_optimizer_and_limits(self):
        plan = json.loads((ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json').read_text())
        validate_plan(plan)
        for section, key, value in (
                ('protocol', 'attempts', 3328), ('protocol', 'early_stopping', True),
                ('protocol', 'probability_threshold', .4), ('objective', 'positive_weight', 1.),
                ('selection', 'positive_pixels', 152), ('resources', 'workers', 2),
                ('resources', 'worker_rss_mib', 1024), ('fit_gate', 'minimum', .5)):
            changed = copy.deepcopy(plan); changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_plan(changed)

    def test_all_ignored_skips_forward_rng_batchnorm_and_existing_optimizer_state(self):
        model = nn.Sequential(nn.Conv2d(27, 1, 1), nn.BatchNorm2d(1), nn.Dropout(.5))
        optimizer = _CountingOptimizer(torch.optim.SGD(model.parameters(), lr=.05,
                                                       momentum=.9, weight_decay=.1))
        # Establish momentum without using the model's forward or random state.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        calls = []
        handle = model.register_forward_hook(lambda *args: calls.append(True))
        labels = torch.full((1, 64, 64), 255, dtype=torch.uint8)
        sample = {'features': torch.full((1, 27, 64, 64), float('nan')),
                  'labels': labels, 'valid': labels != 255}
        try:
            result = train_step(model, optimizer, sample, 2640/151,
                                image_count=13, total_reviewed_pixels=2791)
        finally:
            handle.remove()
        self.assertFalse(result['updated'])
        self.assertIsNone(result['scaled_loss_before_step'])
        self.assertFalse(calls)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[key]))
        after = optimizer.optimizer.state_dict()
        self.assertEqual(after['param_groups'], optimizer_state['param_groups'])
        for key, item in optimizer_state['state'].items():
            self.assertTrue(torch.equal(item['momentum_buffer'], after['state'][key]['momentum_buffer']))

    def test_inconsistent_validity_fails_before_forward_or_optimizer(self):
        model = nn.Conv2d(27, 1, 1)
        optimizer = _CountingOptimizer(torch.optim.SGD(model.parameters(), lr=.05))
        calls = []
        handle = model.register_forward_hook(lambda *args: calls.append(True))
        labels = torch.zeros((1, 64, 64), dtype=torch.uint8)
        sample = {'features': torch.zeros((1, 27, 64, 64)), 'labels': labels,
                  'valid': torch.zeros_like(labels, dtype=torch.bool)}
        try:
            with self.assertRaisesRegex(ValueError, 'Validity'):
                train_step(model, optimizer, sample, 2640/151,
                           image_count=13, total_reviewed_pixels=2791)
        finally:
            handle.remove()
        self.assertFalse(calls)
        self.assertEqual(optimizer.step_calls, 0)


if __name__ == '__main__':
    unittest.main()
