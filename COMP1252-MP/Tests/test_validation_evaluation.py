"""Saved validation predictions must preserve ignored/undefined and group semantics."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Solutions'))
from evaluate_validation_pilot import numpy_summary, verify_label_lock


def probability(logits):
    return 1/(1+np.exp(-logits))


class ValidationEvaluationTests(unittest.TestCase):
    def test_missing_label_lock_fails_closed_before_model_or_git_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            with self.assertRaisesRegex(ValueError,'Missing or oversized'):
                verify_label_lock(base,base/'protocol.json',base/'missing-lock.json',{}, {})

    def rows(self):
        return [{'sample_id':'loss-a','event_id':'a','group_id':'group-a','kind':'loss'},
                {'sample_id':'stable-a','event_id':'a','group_id':'group-a','kind':'stable_roi'}]

    def test_stable_only_undefined_metrics_are_not_promoted_to_perfect_clearing_scores(self):
        labels=np.array([[[0,0,255],[0,255,255]]],dtype='uint8')
        logits=np.full(labels.shape,-3.,dtype='float64')
        row=self.rows()[1]
        result=numpy_summary(logits,probability(logits),labels,[row])
        metrics=result['summaries']['stable_pairs']['metrics']
        self.assertEqual(metrics['tn'],3)
        self.assertEqual(metrics['specificity'],1.)
        self.assertEqual(metrics['reviewed_stable_false_positive_rate'],0.)
        for key in ('precision','recall','dice','iou','balanced_accuracy'):
            self.assertIsNone(metrics[key])
        self.assertEqual(result['summaries']['pooled']['group_count'],1)
        self.assertIsNone(result['summaries']['loss_pairs']['weighted_bce'])
        changed=logits.copy();changed[0,0,0]=3.
        failed=numpy_summary(changed,probability(changed),labels,[row])['summaries']['stable_pairs']['metrics']
        self.assertEqual(failed['fp'],1)
        self.assertEqual(failed['dice'],0.)
        self.assertIsNone(failed['recall'])

    def test_ignored_changes_leave_counts_losses_and_group_totals_identical(self):
        labels=np.array([[[1,1,1,1],[0,255,255,255]],[[0,0,0,255],[255,255,255,255]]],dtype='uint8')
        logits=np.array([[[2.,2.,-2.,2.],[-2.,9.,-9.,9.]],[[2.,-2.,-2.,9.],[9.,9.,9.,9.]]])
        result=numpy_summary(logits,probability(logits),labels,self.rows(),2.)
        altered=logits.copy();altered[labels==255]=float('nan')
        self.assertEqual(result,numpy_summary(altered,probability(altered),labels,self.rows(),2.))
        self.assertEqual([result['summaries']['pooled']['metrics'][key] for key in ('tp','fp','fn','tn')],[3,1,1,3])
        self.assertEqual(result['summaries']['pooled']['sample_count'],2)
        self.assertEqual(result['summaries']['pooled']['group_count'],1)
        self.assertEqual(len(result['summaries']['groups']),1)

    def test_pixel_pooled_loss_and_groups_are_reconstructed_without_equal_image_reweighting(self):
        labels=np.array([[[1,1,1,1],[0,255,255,255]],[[0,0,0,255],[255,255,255,255]]],dtype='uint8')
        logits=np.array([[[2.,2.,2.,2.],[-2.,0.,0.,0.]],[[1.,1.,1.,0.],[0.,0.,0.,0.]]])
        rows=self.rows();rows[1]['group_id']='group-b'
        result=numpy_summary(logits,probability(logits),labels,rows,2.)
        first=(8*np.logaddexp(0.,-2.)+np.logaddexp(0.,-2.))/5
        second=np.logaddexp(0.,1.)
        expected=(5*first+3*second)/8
        self.assertAlmostEqual(result['summaries']['pooled']['weighted_bce'],expected)
        self.assertGreater(abs(expected-(first+second)/2),.01)
        self.assertEqual(result['summaries']['pooled']['group_count'],2)
        self.assertEqual(len(result['summaries']['groups']),2)
        for invalid in ('probability','sample_ids','kind'):
            probabilities=probability(logits);changed=[dict(r) for r in rows]
            if invalid=='probability':probabilities[0,0,0]=.1
            elif invalid=='sample_ids':changed[1]['sample_id']=changed[0]['sample_id']
            else:changed[1]['kind']='train'
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                numpy_summary(logits,probabilities,labels,changed,2.)


if __name__=='__main__':unittest.main()
