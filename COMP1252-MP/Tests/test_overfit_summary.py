"""Independent saved-logit reconstruction and resource receipt regressions."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from summarize_pilot_overfit import recompute_checkpoint, resource_receipt


def sigmoid(values):
    return 1/(1+np.exp(-values))


class IndependentCheckpointTests(unittest.TestCase):
    def test_ranking_signal_can_exist_while_every_reviewed_prediction_is_positive(self):
        labels = np.array([[1, 0, 255]], dtype='uint8')
        logits = np.array([[3., 1., 12.]])
        result = recompute_checkpoint(logits, sigmoid(logits), labels, pos_weight=2.)
        self.assertEqual([result['metrics'][k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 0, 0])
        self.assertFalse(result['fit']['passed'])
        self.assertEqual(result['probability_diagnostics']['training_pairwise_auc'], 1.)
        expected = (2*np.log1p(np.exp(-3))+np.log1p(np.exp(1)))/2
        self.assertAlmostEqual(result['weighted_bce'], expected)
        changed = logits.copy()
        changed[labels == 255] = -123.
        self.assertEqual(result, recompute_checkpoint(changed, sigmoid(changed), labels, 2.))

    def test_a_completed_no_fit_result_is_retained_and_not_a_numerical_failure(self):
        labels = np.array([[1, 0, 255]], dtype='uint8')
        for logits, passed in [(np.array([[7., -7., 0.]]), True),
                               (np.array([[-7., -9., 0.]]), False)]:
            with self.subTest(passed=passed):
                result = recompute_checkpoint(logits, sigmoid(logits), labels, 1.)
                self.assertEqual(result['fit']['passed'], passed)
                self.assertTrue(np.isfinite(result['weighted_bce']))
                self.assertEqual(result['metrics']['positive_pixels'], 1)
                self.assertEqual(result['metrics']['stable_pixels'], 1)

    def test_corrupted_probabilities_nonfinite_logits_and_wrong_support_are_rejected(self):
        labels = np.array([[1, 0, 255]], dtype='uint8')
        logits = np.array([[3., -3., 0.]])
        bad = sigmoid(logits)
        bad[0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            recompute_checkpoint(logits, bad, labels, 1.)
        for invalid in [np.array([[3., -3., np.nan]]), np.array([[3., -3.]])]:
            with self.assertRaises(ValueError):
                recompute_checkpoint(invalid, sigmoid(invalid), labels, 1.)
        with self.assertRaises(ValueError):
            recompute_checkpoint(logits, sigmoid(logits), np.zeros_like(labels), 1.)


class ResourceReceiptTests(unittest.TestCase):
    def receipt(self):
        return {'device': 'cpu', 'gpu_used': False, 'batch_size': 1, 'dataloader_workers': 0,
                'native_threads': 1, 'torch_intraop_threads': 1, 'torch_interop_threads': 1,
                'resources': {'workers': 1, 'native_thread_limit': 1, 'gpu_used': False,
                              'nice_increment': 10, 'worker_limit_mib': 512,
                              'minimum_available_gib_limit': 1.5, 'runtime_limit_seconds': 300,
                              'worker_peak_rss_mib_sampled': 310., 'worker_high_water_rss_mib': 308.,
                              'minimum_available_gib_sampled': 3., 'elapsed_seconds': 4.}}

    def test_worst_observed_memory_is_used_and_unsafe_receipts_fail(self):
        original = self.receipt()
        self.assertEqual(resource_receipt(original)['peak_worker_mib'], 310.)
        for field, value in [('worker_peak_rss_mib_sampled', 513.),
                             ('minimum_available_gib_sampled', 1.4), ('elapsed_seconds', 301.)]:
            changed = deepcopy(original)
            changed['resources'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resource_receipt(changed)
        changed = deepcopy(original)
        changed['dataloader_workers'] = 1
        with self.assertRaises(ValueError):
            resource_receipt(changed)


if __name__ == '__main__':
    unittest.main()
