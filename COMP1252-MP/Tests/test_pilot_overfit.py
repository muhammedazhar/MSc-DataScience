"""Diagnostic gates and analytic controls for the bounded one-event fit.

These fixtures do not train a model or read satellite imagery. Root schedules
execution through the resource-guarded test runner.
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
from diagnose_pilot_overfit import (checkpoint_steps, constant_optimum,
                                   fit_assessment, probability_diagnostics,
                                   select_record)


def accepted_record(identifier="event-first", group="group-first"):
    return {"event_id": identifier, "group_id": group, "pair_id": "pair-"+identifier,
            "split": "train", "training_approved": True,
            "visibility_evidence": {"decision_status": "accepted", "training_approved": True}}


def counts(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "probability_threshold": .5}


class ConstantProbabilityTests(unittest.TestCase):
    def test_local_balance_has_half_probability_and_analytic_weighted_loss(self):
        result = constant_optimum(25, 256, 256/25)
        self.assertAlmostEqual(result["probability"], .5, places=14)
        self.assertAlmostEqual(result["logit"], 0., places=14)
        self.assertAlmostEqual(result["mean_weight"], 512/281, places=14)
        self.assertAlmostEqual(result["weighted_bce"], (512/281)*math.log(2), places=13)

    def test_global_weight_pushes_the_one_event_constant_optimum_above_half(self):
        weight = 2640/151
        result = constant_optimum(25, 256, weight)
        expected = (2640*25)/(2640*25+151*256)
        self.assertAlmostEqual(result["probability"], expected, places=14)
        self.assertAlmostEqual(result["logit"], math.log((2640*25)/(151*256)), places=14)
        self.assertGreater(result["probability"], .63)
        self.assertLess(result["probability"], .64)
        # A genuine closed-form minimum must beat both neighboring constants,
        # rather than simply agree with a hard-coded probability fixture.
        def objective(probability):
            return (-weight*25*math.log(probability)-256*math.log1p(-probability))/281
        self.assertAlmostEqual(result["weighted_bce"], objective(expected), places=13)
        self.assertLess(result["weighted_bce"], objective(expected-.05))
        self.assertLess(result["weighted_bce"], objective(expected+.05))

    def test_unweighted_constant_matches_the_observed_class_prior(self):
        result = constant_optimum(25, 256, 1.)
        prior = 25/281
        expected = -prior*math.log(prior)-(1-prior)*math.log1p(-prior)
        self.assertAlmostEqual(result["probability"], prior, places=14)
        self.assertAlmostEqual(result["mean_weight"], 1., places=14)
        self.assertAlmostEqual(result["weighted_bce"], expected, places=13)

    def test_missing_classes_noninteger_counts_and_invalid_weights_are_rejected(self):
        for positive, negative in ((0, 256), (25, 0), (-1, 256), (25, -1),
                                   (True, 256), (25, False), (25., 256), (25, 256.)):
            with self.subTest(positive=positive, negative=negative):
                with self.assertRaises(ValueError):
                    constant_optimum(positive, negative, 1.)
        for weight in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    constant_optimum(25, 256, weight)


class ReviewedFitGateTests(unittest.TestCase):
    def test_perfect_and_near_perfect_reviewed_fits_pass(self):
        perfect = fit_assessment(counts(25, 0, 0, 256), 25, 256)
        self.assertTrue(perfect["passed"])
        self.assertEqual(perfect["reasons"], [])
        self.assertEqual([perfect[k] for k in ("precision", "recall", "specificity", "dice")], [1.]*4)
        near = fit_assessment(counts(24, 1, 1, 255), 25, 256)
        self.assertTrue(near["passed"])
        self.assertAlmostEqual(near["recall"], .96)
        self.assertAlmostEqual(near["dice"], .96)

    def test_all_positive_and_all_negative_predictions_cannot_pass(self):
        all_positive = fit_assessment(counts(25, 256, 0, 0), 25, 256)
        self.assertFalse(all_positive["passed"])
        self.assertTrue(all_positive["reasons"])
        self.assertAlmostEqual(all_positive["dice"], 50/306)
        self.assertEqual(all_positive["specificity"], 0.)
        all_negative = fit_assessment(counts(0, 0, 25, 256), 25, 256)
        self.assertFalse(all_negative["passed"])
        self.assertEqual(all_negative["recall"], 0.)
        self.assertEqual(all_negative["dice"], 0.)

    def test_forged_metric_fields_do_not_override_the_confusion_counts(self):
        forged = {**counts(25, 256, 0, 0), "precision": 1., "recall": 1.,
                  "specificity": 1., "dice": 1., "accuracy": 1.}
        result = fit_assessment(forged, 25, 256)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["precision"], 25/281)
        self.assertEqual(result["specificity"], 0.)

    def test_gate_accepts_exact_predeclared_minimum(self):
        result = fit_assessment(counts(19, 1, 1, 19), 20, 20, minimum=.95)
        self.assertTrue(result["passed"])
        for name in ("precision", "recall", "specificity", "dice"):
            self.assertAlmostEqual(result[name], .95)

    def test_wrong_support_noninteger_counts_and_threshold_changes_are_errors(self):
        invalid = [counts(24, 0, 0, 256), counts(25, 1, 0, 256),
                   counts(-1, 0, 26, 256), counts(25., 0, 0, 256),
                   {**counts(25, 0, 0, 256), "probability_threshold": .6}]
        for metrics in invalid:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    fit_assessment(metrics, 25, 256)
        with self.assertRaises(ValueError):
            fit_assessment(counts(True, 0, 0, 2), 1, 2)
        for minimum in (0, -1, 1.01, True, float("nan"), float("inf")):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    fit_assessment(counts(25, 0, 0, 256), 25, 256, minimum=minimum)


class FixedDiagnosticSelectionTests(unittest.TestCase):
    def test_checkpoint_schedule_is_unique_bounded_and_includes_final_attempt(self):
        expected = {1: [0, 1], 26: [0, 26], 27: [0, 26, 27],
                    128: [0, 26, 128], 129: [0, 26, 128, 129],
                    256: [0, 26, 128, 256], 512: [0, 26, 128, 256, 512]}
        for limit, schedule in expected.items():
            with self.subTest(limit=limit):
                self.assertEqual(checkpoint_steps(limit), schedule)
                self.assertEqual(schedule, sorted(set(schedule)))
                self.assertEqual(schedule[-1], limit)
        for limit in (0, -1, 513, True, 26., float("nan")):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    checkpoint_steps(limit)

    def test_selection_is_a_single_exact_approved_training_event(self):
        rows = [accepted_record(), accepted_record("event-second", "group-second")]
        self.assertEqual(select_record(rows), 0)
        self.assertEqual(select_record(rows, "event-second"), 1)
        with self.assertRaises(ValueError):
            select_record(rows, "event-missing")
        with self.assertRaises(ValueError):
            select_record([])
        with self.assertRaises(ValueError):
            select_record([rows[0], dict(rows[0])], "event-first")
        for field, value in (("split", "test"), ("split", "validation"),
                             ("training_approved", False)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    select_record([{**accepted_record(), field: value}])


class ProbabilityEvidenceTests(unittest.TestCase):
    def example(self):
        labels = np.array([[1, 0, 255], [1, 0, 255]], dtype="uint8")
        probability = np.array([[.9, .2, np.nan], [.4, .4, np.inf]], dtype="float64")
        return probability, labels

    def test_class_summaries_and_pairwise_auc_ties_use_reviewed_pixels_only(self):
        probability, labels = self.example()
        result = probability_diagnostics(probability, labels)
        self.assertEqual(result["positive"]["count"], 2)
        self.assertEqual(result["negative"]["count"], 2)
        self.assertEqual(result["ignored_pixels"], 2)
        self.assertAlmostEqual(result["positive"]["mean"], .65)
        self.assertAlmostEqual(result["negative"]["mean"], .3)
        self.assertAlmostEqual(result["positive"]["min"], .4)
        self.assertAlmostEqual(result["positive"]["max"], .9)
        # Positive .9 wins both comparisons; positive .4 wins one and ties one.
        self.assertAlmostEqual(result["training_pairwise_auc"], .875)

    def test_arbitrary_ignored_probabilities_cannot_change_the_evidence(self):
        probability, labels = self.example()
        changed = probability.copy()
        changed[labels == 255] = [-100., 100.]
        self.assertEqual(probability_diagnostics(probability, labels),
                         probability_diagnostics(changed, labels))

    def test_invalid_reviewed_probabilities_shapes_and_unknown_labels_fail(self):
        probability, labels = self.example()
        for value in (-.01, 1.01, np.nan, np.inf):
            changed = probability.copy()
            changed[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    probability_diagnostics(changed, labels)
        with self.assertRaises(ValueError):
            probability_diagnostics(probability[:, :2], labels)
        bad_labels = labels.copy()
        bad_labels[0, 0] = 2
        with self.assertRaises(ValueError):
            probability_diagnostics(probability, bad_labels)
        for single_class in (np.zeros((2, 2), dtype="uint8"),
                             np.ones((2, 2), dtype="uint8"),
                             np.full((2, 2), 255, dtype="uint8")):
            with self.subTest(single_class=single_class.tolist()):
                with self.assertRaises(ValueError):
                    probability_diagnostics(np.full((2, 2), .5), single_class)


if __name__ == "__main__":
    unittest.main()
