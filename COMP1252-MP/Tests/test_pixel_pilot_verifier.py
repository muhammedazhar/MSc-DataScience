"""Independent pooled diagnostics must not hide weak reviewed examples."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from verify_pixel_pilot import aggregate_checkpoint, verify_steps


def sigmoid(values):
    return 1/(1+np.exp(-values))


class PixelPilotVerificationTests(unittest.TestCase):
    def test_pixel_weighted_loss_is_not_equal_image_mean(self):
        labels = np.array([[[1, 0, 255, 255]], [[1, 1, 0, 0]]], dtype='uint8')
        logits = np.array([[[4., -4., 11., -11.]], [[-2., -2., -2., -2.]]])
        result = aggregate_checkpoint(logits, sigmoid(logits), labels, ['a', 'b'], 2.)
        p = np.array([4., -2., -2.])
        n = np.array([-4., -2., -2.])
        expected = (2*np.logaddexp(0., -p).sum()+np.logaddexp(0., n).sum())/6
        self.assertAlmostEqual(result['aggregate']['weighted_bce'], expected)
        image_mean = sum(row['weighted_bce'] for row in result['examples'])/2
        self.assertGreater(abs(image_mean-expected), .01)
        self.assertEqual([result['aggregate']['metrics'][key] for key in ('tp', 'fp', 'fn', 'tn')], [1, 0, 2, 3])
        changed = logits.copy()
        changed[labels == 255] *= -1
        self.assertEqual(result, aggregate_checkpoint(changed, sigmoid(changed), labels, ['a', 'b'], 2.))

    def test_pooled_fit_does_not_hide_failed_small_example(self):
        labels = np.zeros((2, 2, 20), dtype='uint8')
        labels[0, 0] = 1
        labels[1] = 255
        labels[1, 0, :2] = [1, 0]
        logits = np.where(labels == 1, 7., -7.)
        logits[1, 0, 0] = -7.
        result = aggregate_checkpoint(logits, sigmoid(logits), labels, ['large', 'small'], 1.)
        self.assertTrue(result['aggregate']['fit']['passed'])
        self.assertFalse(result['examples'][1]['fit']['passed'])
        self.assertFalse(result['all_examples_fit'])
        self.assertFalse(result['fit_passed'])
        self.assertTrue(np.isfinite(result['aggregate']['weighted_bce']))

    def test_corrupt_probability_and_event_coverage_fail(self):
        labels = np.array([[[1, 0]], [[1, 0]]], dtype='uint8')
        logits = np.array([[[3., -3.]], [[4., -4.]]])
        probability = sigmoid(logits)
        for ids in (['a'], ['a', 'a']):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'event coverage'):
                aggregate_checkpoint(logits, probability, labels, ids, 1.)
        bad = probability.copy()
        bad[1, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            aggregate_checkpoint(logits, bad, labels, ['a', 'b'], 1.)

    def test_step_receipts_bind_order_support_scale_and_zero_gradients(self):
        labels = np.array([[[1, 0, 255]], [[1, 0, 0]]], dtype='uint8')
        objective, protocol = {'total_reviewed_pixels': 5}, {'attempts': 4, 'cycles': 2}
        rows = [{'attempt': i+1, 'event_id': ['a', 'b'][i % 2],
                 'reviewed_pixels': [2, 3][i % 2], 'step_scale': [4/5, 6/5][i % 2],
                 'scaled_loss_before_step': .5, 'gradient_norm': .2, 'parameter_delta_l2': .1,
                 'zero_parameter_delta': False, 'ignored_gradient_max_abs': 0.,
                 'optimizer_step_called': True} for i in range(4)]
        self.assertEqual(verify_steps(rows, labels, ['a', 'b'], objective, protocol), [])
        for field, invalid in [('event_id', 'b'), ('reviewed_pixels', 3), ('step_scale', 1.),
                               ('ignored_gradient_max_abs', .01), ('scaled_loss_before_step', float('inf'))]:
            altered = deepcopy(rows)
            altered[0][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_steps(altered, labels, ['a', 'b'], objective, protocol)
        rows[-1].update(parameter_delta_l2=0., zero_parameter_delta=True)
        self.assertEqual(verify_steps(rows, labels, ['a', 'b'], objective, protocol), [4])


if __name__ == '__main__':
    unittest.main()
