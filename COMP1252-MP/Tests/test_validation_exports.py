"""Validation publication preserves unknown pixels and prediction-blind locks."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
from build_validation_pilot import (ARTIFACTS, bind_case_artifacts, canonical,
                                    digest, enforce_selection_policy, validation_supervision)


class ValidationExportTests(unittest.TestCase):
    def arrays(self):
        positive = np.zeros((64, 64), dtype=bool); positive[10:12, 10:12] = True
        stable = np.zeros((64, 64), dtype=bool); stable[:8, :8] = True
        empty = np.zeros((64, 64), dtype=bool)
        features = np.zeros((27, 64, 64), dtype='float32')
        return positive, stable, empty, features, np.ones((64, 64), dtype=bool), empty.copy()

    def test_partial_labels_never_expand_to_unknown_context(self):
        positive, stable, qa, features, valid, excluded = self.arrays()
        features[:, 30, 30] = np.nan; valid[30, 30] = False
        labels = validation_supervision('loss', positive, stable, qa, features, valid, excluded)
        self.assertEqual(int((labels == 1).sum()), 4)
        self.assertEqual(int((labels == 0).sum()), 64)
        self.assertEqual(int((labels == 255).sum()), 4096-68)
        self.assertEqual(int(labels[30, 30]), 255)
        self.assertEqual(labels.dtype, np.dtype('uint8'))

    def test_zero_positive_stable_pair_supervises_only_its_reviewed_control(self):
        positive, stable, qa, features, valid, excluded = self.arrays()
        positive[:] = False
        labels = validation_supervision('stable_roi', positive, stable, qa, features, valid, excluded)
        self.assertTrue(np.array_equal(labels == 0, stable))
        self.assertFalse((labels == 1).any())
        self.assertTrue((labels[~stable] == 255).all())
        with self.assertRaisesRegex(ValueError, 'reviewed positives'):
            validation_supervision('loss', positive, stable, qa, features, valid, excluded)
        positive[10:12, 10:12] = True
        with self.assertRaisesRegex(ValueError, 'stable-only'):
            validation_supervision('stable_roi', positive, stable, qa, features, valid, excluded)

    def test_reviewed_qa_nonfinite_and_catalogue_conflicts_cannot_be_silently_dropped(self):
        for change in ('qa', 'nonfinite', 'validity', 'catalogue', 'overlap'):
            positive, stable, qa, features, valid, excluded = self.arrays()
            if change == 'qa':
                qa[0, 0] = True
            elif change == 'nonfinite':
                features[0, 0, 0] = np.nan; valid[0, 0] = False
            elif change == 'validity':
                valid[30, 30] = False
            elif change == 'catalogue':
                excluded[0, 0] = True
            else:
                positive[0, 0] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validation_supervision('loss', positive, stable, qa, features, valid, excluded)

    def bound_fixture(self, temporary, split='validation'):
        directory = Path(temporary)/'event-bound'; directory.mkdir()
        for name in ARTIFACTS:
            (directory/name).write_bytes(('explicit review bytes: '+name).encode())
        case = {'event_id': 'event-bound', 'split': split, 'pair': {'split': split},
                'training_approved': False, 'evaluation_approved': False,
                'artifact_sha256': {name: digest(directory/name) for name in ARTIFACTS}}
        path = directory/'case.json'; path.write_text(json.dumps(case)+'\n')
        review = {'case_file': str(path.relative_to(ROOT)), 'case_sha256': digest(path),
                  'prepared_case_sha256': canonical(case), 'artifact_sha256': deepcopy(case['artifact_sha256'])}
        return path, case, review

    def test_raw_case_and_actual_artifact_bytes_are_both_review_locked(self):
        for change in ('case_formatting', 'artifact_bytes', 'artifact_map'):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                path, case, review = self.bound_fixture(temporary)
                bind_case_artifacts(review)
                if change == 'case_formatting':
                    path.write_text(path.read_text()+'\n')
                    self.assertEqual(canonical(json.loads(path.read_text())), canonical(case))
                elif change == 'artifact_bytes':
                    (path.parent/'timeline.png').write_bytes(b'changed after review')
                else:
                    review['artifact_sha256']['timeline.png'] = '0'*64
                with self.assertRaises(ValueError):
                    bind_case_artifacts(review)

    def test_training_and_test_evidence_cannot_be_repurposed_as_validation(self):
        for split in ('train', 'test'):
            with self.subTest(split=split), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                _, _, review = self.bound_fixture(temporary, split)
                with self.assertRaisesRegex(ValueError, 'validation evidence'):
                    bind_case_artifacts(review)

    def test_predeclared_alternate_requires_retained_primary_rejection(self):
        candidates = [{'event_id': 'event-primary', 'stratum': '36UXA', 'role': 'primary'},
                      {'event_id': 'event-alternate', 'stratum': '36UXA', 'role': 'alternate'}]
        dispositions = [{'event_id': 'event-primary', 'decision_status': 'rejected'},
                        {'event_id': 'event-alternate', 'decision_status': 'accepted'}]
        self.assertEqual(enforce_selection_policy(candidates, dispositions), ['event-alternate'])
        both = deepcopy(dispositions); both[0]['decision_status'] = 'accepted'
        with self.assertRaisesRegex(ValueError, 'primary is rejected'):
            enforce_selection_policy(candidates, both)
        with self.assertRaisesRegex(ValueError, 'retained review'):
            enforce_selection_policy(candidates, dispositions[1:])


if __name__ == '__main__':
    unittest.main()
