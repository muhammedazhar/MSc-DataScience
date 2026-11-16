"""Sampling corrections must match a reviewed-pixel loss and shared gradient."""

import os
from pathlib import Path
import sys
import unittest


for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import torch
from torch.nn import functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from pilot_objectives import masked_bce_with_logits
from pilot_pixel_objective import pixel_aggregate_step_loss, pixel_step_scale


class PixelAggregateObjectiveTests(unittest.TestCase):
    def fixture(self):
        parameter = torch.tensor([.7, -.3], dtype=torch.float64, requires_grad=True)
        labels = [torch.tensor([[1, 0, 255, 255], [255, 255, 255, 255]]),
                  torch.tensor([[0, 1, 0, 1], [0, 255, 255, 255]]),
                  torch.tensor([[1, 0, 0, 1], [0, 1, 0, 255]])]
        images = []
        for index, target in enumerate(labels):
            values = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)/8
            features = values + torch.tensor([index*.8, -index*.4], dtype=torch.float64)
            raw_logits = features @ parameter
            logits = torch.where(target != 255, raw_logits,
                                 torch.full_like(raw_logits, float('nan')))
            images.append((logits, target))
        return parameter, images

    def test_unequal_support_loss_and_shared_gradient_equal_concatenated_reference(self):
        parameter, images = self.fixture()
        weight = 3.5
        total = sum(int((target != 255).sum()) for _, target in images)
        corrected = torch.stack([
            pixel_aggregate_step_loss(logits, target, target != 255, weight,
                                      image_count=len(images), total_reviewed_pixels=total)
            for logits, target in images]).mean()
        # Independent reference: concatenate only reviewed entries, then ask
        # Torch for one BCE. The shared parameter makes gradient equivalence
        # meaningful beyond comparing unrelated per-logit derivatives.
        logits = torch.cat([value[target != 255] for value, target in images])
        targets = torch.cat([target[target != 255] for _, target in images]).double()
        reference = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(weight, dtype=torch.float64))
        corrected_gradient, = torch.autograd.grad(corrected, parameter, retain_graph=True)
        reference_gradient, = torch.autograd.grad(reference, parameter)
        torch.testing.assert_close(corrected, reference, rtol=0, atol=1e-14)
        torch.testing.assert_close(corrected_gradient, reference_gradient, rtol=0, atol=1e-14)
        self.assertEqual(total, 14)

    def test_equal_image_mean_is_a_counterexample_for_unequal_support(self):
        parameter, images = self.fixture()
        old_loss = torch.stack([masked_bce_with_logits(logits, target, pos_weight=3.5)
                                for logits, target in images]).mean()
        selected = torch.cat([value[target != 255] for value, target in images])
        targets = torch.cat([target[target != 255] for _, target in images]).double()
        pixel_loss = F.binary_cross_entropy_with_logits(
            selected, targets, pos_weight=torch.tensor(3.5, dtype=torch.float64))
        old_gradient, = torch.autograd.grad(old_loss, parameter, retain_graph=True)
        pixel_gradient, = torch.autograd.grad(pixel_loss, parameter)
        self.assertGreater(float((old_loss-pixel_loss).abs()), 1e-4)
        self.assertGreater(float((old_gradient-pixel_gradient).abs().max()), 1e-4)

    def test_ignored_nonfinite_logits_have_exact_zero_gradient_after_scaling(self):
        labels = torch.tensor([[1, 255], [0, 255]])
        logits = torch.tensor([[.4, float('nan')], [-.9, float('inf')]],
                              dtype=torch.float64, requires_grad=True)
        loss = pixel_aggregate_step_loss(logits, labels, labels != 255, 2.,
                                         image_count=3, total_reviewed_pixels=10)
        expected = .6*F.binary_cross_entropy_with_logits(
            torch.tensor([.4, -.9], dtype=torch.float64),
            torch.tensor([1., 0.], dtype=torch.float64),
            pos_weight=torch.tensor(2., dtype=torch.float64))
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-14)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(torch.equal(logits.grad[labels == 255], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(bool((logits.grad[labels != 255].abs() > 0).all()))

    def test_all_ignored_loss_is_none_without_gradient_or_fake_zero(self):
        logits = torch.full((2, 2), float('nan'), requires_grad=True)
        labels = torch.full((2, 2), 255)
        result = pixel_aggregate_step_loss(logits, labels, labels != 255, 2.,
                                           image_count=3, total_reviewed_pixels=10)
        self.assertIsNone(result)
        self.assertIsNone(logits.grad)
        self.assertEqual(pixel_step_scale(0, 3, 10), 0.)

    def test_scale_is_one_for_one_image_or_equal_reviewed_support(self):
        self.assertEqual(pixel_step_scale(17, 1, 17), 1.)
        self.assertEqual(pixel_step_scale(17, 3, 51), 1.)

    def test_count_types_bounds_and_unrepresentable_scales_are_rejected(self):
        invalid = ((-1, 3, 10), (11, 3, 10), (2., 3, 10), (True, 3, 10),
                   (2, 0, 10), (2, -1, 10), (2, True, 10), (2, 3., 10),
                   (0, 3, 0), (0, 3, -1), (0, 3, True), (0, 3, 10.),
                   (1, 10**400, 1), (1, 1, 10**400))
        for counts in invalid:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                pixel_step_scale(*counts)

    def test_wrapper_enforces_global_counts_support_and_original_validity(self):
        logits = torch.zeros((1, 2))
        labels = torch.tensor([[1, 0]])
        for image_count, total in ((0, 2), (True, 2), (1, 0), (1, 1)):
            with self.subTest(image_count=image_count, total=total), self.assertRaises(ValueError):
                pixel_aggregate_step_loss(logits, labels, image_count=image_count,
                                          total_reviewed_pixels=total)
        with self.assertRaises(ValueError):
            pixel_aggregate_step_loss(logits, labels, torch.zeros_like(labels, dtype=torch.bool),
                                      image_count=1, total_reviewed_pixels=2)
        with self.assertRaises(ValueError):
            pixel_aggregate_step_loss(logits, torch.full_like(labels, 255), pos_weight=0,
                                      image_count=1, total_reviewed_pixels=2)
        with self.assertRaises(ValueError):
            pixel_aggregate_step_loss(logits, torch.full_like(labels, 255),
                                      image_count=0, total_reviewed_pixels=2)

    def test_singleton_image_batch_matches_unbatched_but_multiple_images_fail(self):
        logits = torch.tensor([[.5, -.2]], dtype=torch.float64)
        labels = torch.tensor([[1, 0]])
        expected = pixel_aggregate_step_loss(logits, labels, image_count=3,
                                             total_reviewed_pixels=10)
        actual = pixel_aggregate_step_loss(logits[None, None], labels[None],
                                           image_count=3, total_reviewed_pixels=10)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'one sampled image'):
            pixel_aggregate_step_loss(logits[None, None].expand(2, 1, 1, 2),
                                      labels[None].expand(2, 1, 2),
                                      image_count=3, total_reviewed_pixels=10)


if __name__ == '__main__':
    unittest.main()
