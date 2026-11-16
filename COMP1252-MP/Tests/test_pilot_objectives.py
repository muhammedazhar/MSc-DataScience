"""Regressions for ignored-pixel influence, gradient safety and aggregation."""

import math
import os
from pathlib import Path
import sys
import unittest


for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
from pilot_objectives import ConfusionCounts, masked_bce_with_logits


class ReviewedLossTests(unittest.TestCase):
    def test_ignored_nonfinite_predictions_never_enter_loss_or_gradients(self):
        labels = torch.tensor([[1, 255], [0, 255]], dtype=torch.uint8)
        logits = torch.tensor([[0., float("nan")], [-2., float("inf")]], requires_grad=True)
        loss = masked_bce_with_logits(logits, labels, labels != 255)
        expected = (math.log(2)+math.log1p(math.exp(-2)))/2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        self.assertEqual(float(logits.grad[1, 1]), 0.)
        self.assertAlmostEqual(float(logits.grad[0, 0]), -.25)
        self.assertGreater(float(logits.grad[1, 0]), 0.)

    def test_changing_only_ignored_predictions_cannot_change_loss_or_counts(self):
        labels = torch.tensor([[1, 255, 0, 255]])
        first = torch.tensor([[2., -100., -2., 100.]])
        second = torch.tensor([[2., float("nan"), -2., float("-inf")]])
        torch.testing.assert_close(masked_bce_with_logits(first, labels),
                                   masked_bce_with_logits(second, labels), rtol=0, atol=0)
        a, b = ConfusionCounts(), ConfusionCounts()
        a.update(first, labels); b.update(second, labels)
        self.assertEqual(a.summary(), b.summary())

    def test_all_ignored_batch_returns_none_without_creating_a_fake_zero_loss(self):
        logits = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
        labels = torch.full((1, 2, 2), 255, dtype=torch.uint8)
        self.assertIsNone(masked_bce_with_logits(logits, labels))
        self.assertIsNone(logits.grad)

    def test_positive_weight_applies_only_to_reviewed_positive_term(self):
        logits = torch.zeros((1, 1, 3), requires_grad=True)
        labels = torch.tensor([[[1, 0, 255]]])
        loss = masked_bce_with_logits(logits, labels, pos_weight=3.)
        self.assertAlmostEqual(loss.item(), 2*math.log(2), places=6)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0, 0]), -.75)
        self.assertAlmostEqual(float(logits.grad[0, 0, 1]), .25)
        self.assertEqual(float(logits.grad[0, 0, 2]), 0.)

    def test_invalid_scalar_weights_fail_even_when_every_pixel_is_ignored(self):
        logits = torch.zeros((1, 1))
        labels = torch.full((1, 1), 255)
        for weight in (0, -1., float("nan"), float("inf"), True, "2", torch.ones(2)):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                masked_bce_with_logits(logits, labels, pos_weight=weight)

    def test_python_scalar_weight_retains_float64_precision(self):
        logits = torch.zeros((1, 1), dtype=torch.float64)
        labels = torch.ones((1, 1), dtype=torch.int64)
        weight = .12345678912345678
        loss = masked_bce_with_logits(logits, labels, pos_weight=weight)
        self.assertAlmostEqual(loss.item(), weight*math.log(2), places=15)

    def test_single_channel_batch_and_unbatched_shapes_align_without_broadcasting(self):
        labels = torch.tensor([[1, 0], [255, 1]])
        logits = torch.tensor([[1., -1.], [9., 2.]])
        expected = masked_bce_with_logits(logits, labels)
        for shaped, target in ((logits[None], labels), (logits[None, None], labels),
                               (logits[None, None], labels[None]), (logits[None], labels[None])):
            torch.testing.assert_close(masked_bce_with_logits(shaped, target), expected)
        for shaped, target in ((logits[None, None].expand(1, 2, 2, 2), labels[None]),
                               (logits[None].expand(2, 2, 2), labels),
                               (logits, labels[:1]), (logits, labels[None, None])):
            with self.subTest(shape=shaped.shape), self.assertRaises(ValueError):
                masked_bce_with_logits(shaped, target)

    def test_unknown_labels_and_boolean_mask_conversion_are_rejected(self):
        logits = torch.zeros((1, 2))
        for labels in (torch.tensor([[0, 2]]), torch.tensor([[0, -1]]),
                       torch.tensor([[0., float("nan")]]), torch.tensor([[False, True]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                masked_bce_with_logits(logits, labels)

    def test_validity_must_be_boolean_exact_and_same_shape(self):
        logits, labels = torch.zeros((1, 2)), torch.tensor([[1, 255]])
        for valid in (torch.tensor([[1, 0]]), torch.tensor([[True, True]]),
                      torch.tensor([True, False]), torch.tensor([[False, False]])):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                masked_bce_with_logits(logits, labels, valid)

    def test_nonfinite_reviewed_logits_are_errors_for_both_loss_and_metrics(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            logits = torch.tensor([[value, 0.]])
            labels = torch.tensor([[1, 255]])
            for operation in (masked_bce_with_logits, ConfusionCounts().update):
                with self.subTest(value=value, operation=operation), self.assertRaisesRegex(ValueError, "finite"):
                    operation(logits, labels)

    def test_integer_logits_and_empty_spatial_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            masked_bce_with_logits(torch.zeros((1, 1), dtype=torch.int64), torch.zeros((1, 1)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            masked_bce_with_logits(torch.empty((0, 2)), torch.empty((0, 2)))


class ReviewedMetricTests(unittest.TestCase):
    def test_aggregate_counts_precede_ratios_instead_of_averaging_image_scores(self):
        counts = ConfusionCounts()
        counts.update(torch.tensor([[2., -2.]]), torch.tensor([[1, 0]]))
        counts.update(torch.tensor([[-2.]*9+[2.]*3]), torch.tensor([[1]*9+[0]*3]))
        summary = counts.summary()
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"], summary["tn"]), (1, 3, 9, 1))
        self.assertEqual(summary["supervised_pixels"], 14)
        self.assertAlmostEqual(summary["recall"], .1)
        self.assertAlmostEqual(summary["precision"], .25)
        self.assertAlmostEqual(summary["dice"], 1/7)
        self.assertAlmostEqual(summary["iou"], 1/13)
        self.assertAlmostEqual(summary["accuracy"], 1/7)
        self.assertAlmostEqual(summary["balanced_accuracy"], (.1+.25)/2)

    def test_all_ignored_update_does_not_create_true_negatives_or_defined_scores(self):
        counts = ConfusionCounts()
        self.assertEqual(counts.update(torch.full((2, 2), float("nan")), torch.full((2, 2), 255)), 0)
        summary = counts.summary()
        self.assertEqual(summary["supervised_pixels"], 0)
        self.assertEqual(summary["ignored_pixels"], 4)
        self.assertEqual(summary["empty_updates"], 1)
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"], summary["tn"]), (0, 0, 0, 0))
        for key in ("precision", "recall", "specificity", "f1", "dice", "iou", "accuracy", "balanced_accuracy"):
            self.assertIsNone(summary[key])

    def test_absent_positive_class_does_not_receive_a_fabricated_overlap_score(self):
        counts = ConfusionCounts()
        counts.update(torch.tensor([[-3., -2.]]), torch.tensor([[0, 0]]))
        summary = counts.summary()
        self.assertEqual(summary["accuracy"], 1.)
        self.assertEqual(summary["specificity"], 1.)
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["dice"])
        self.assertIsNone(summary["balanced_accuracy"])

    def test_zero_logit_is_positive_at_default_half_probability_threshold(self):
        counts = ConfusionCounts()
        counts.update(torch.zeros((1, 2)), torch.tensor([[1, 0]]))
        self.assertEqual((counts.tp, counts.fp, counts.fn, counts.tn), (1, 1, 0, 0))
        thresholded = ConfusionCounts(threshold=.8)
        thresholded.update(torch.tensor([[1., 2.]]), torch.tensor([[1, 0]]))
        self.assertEqual((thresholded.tp, thresholded.fp, thresholded.fn, thresholded.tn), (0, 1, 1, 0))

    def test_failed_update_leaves_previous_valid_counts_unchanged(self):
        counts = ConfusionCounts()
        counts.update(torch.tensor([[1.]]), torch.tensor([[1]]))
        before = counts.summary()
        with self.assertRaises(ValueError):
            counts.update(torch.tensor([[float("nan")]]), torch.tensor([[0]]))
        self.assertEqual(counts.summary(), before)

    def test_invalid_probability_thresholds_are_rejected(self):
        for threshold in (0., 1., -1., float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ConfusionCounts(threshold=threshold)


if __name__ == "__main__":
    unittest.main()
