import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from sisa_cost import cost_grid, expected_cost, normalized_expected_cost


ROOT = Path(__file__).resolve().parents[1]


class SISACostTests(unittest.TestCase):
    def test_closed_form_matches_average_of_all_deletion_locations(self):
        # Independently enumerate the cumulative stages replayed for every slice.
        n, epochs = 4800, 3
        for shards in (1, 4, 16):
            for slices in (1, 2, 4, 10):
                with self.subTest(shards=shards, slices=slices):
                    epoch_budget = 2 * epochs / (slices + 1)
                    by_location = [sum(epoch_budget * j * n / (shards * slices)
                                       for j in range(q, slices + 1))
                                   for q in range(1, slices + 1)]
                    self.assertAlmostEqual(expected_cost(shards, slices, n, epochs=epochs),
                                           sum(by_location) / slices)

    def test_no_slicing_limit_and_paper_configurations(self):
        self.assertEqual(normalized_expected_cost(16, 1), 1 / 16)
        self.assertAlmostEqual(normalized_expected_cost(4, 2), 5 / 24)
        self.assertAlmostEqual(normalized_expected_cost(2, 2), 5 / 12)
        self.assertAlmostEqual(normalized_expected_cost(10, 1), 0.1)
        self.assertAlmostEqual(normalized_expected_cost(100, 1), 0.01)
        self.assertAlmostEqual(normalized_expected_cost(4, 10**9), 2 / (3 * 4))

    def test_grid_and_independent_request_scaling(self):
        rows = cost_grid([4, 10], [1, 2], training_examples=4800, epochs=3, requests=25)
        self.assertEqual([(row["shards"], row["slices"]) for row in rows], [(4, 1), (4, 2), (10, 1), (10, 2)])
        self.assertEqual(rows[0]["expected_examples_per_request"], 3600)
        self.assertEqual(rows[0]["expected_total_examples"], 90000)
        self.assertEqual(expected_cost(4, 1, 4800, epochs=3, requests=25), 90000)
        self.assertEqual(expected_cost(4, 1, 4800, requests=0), 0)
        self.assertIsNone(cost_grid([4], [2])[0]["expected_examples_per_request"])

    def test_candidate_epochs_are_charged_against_a_fixed_common_reference(self):
        # Two slices, three nominal epochs: two whole epochs per prefix.
        expected = expected_cost(4, 2, 4800, epochs=3)
        self.assertEqual(expected, 3000)
        self.assertEqual(normalized_expected_cost(4, 2, epochs=3), expected / 4800)
        self.assertAlmostEqual(normalized_expected_cost(4, 2, epochs=3, reference_epochs=3), 5 / 24)
        low = cost_grid([4], [2], training_examples=4800, epochs=3)[0]
        high = cost_grid([4], [2], training_examples=4800, epochs=6)[0]
        self.assertEqual(high["normalized_expected_cost"], 2 * low["normalized_expected_cost"])
        self.assertEqual(high["expected_examples_per_request"], 2 * low["expected_examples_per_request"])
        changed_reference = cost_grid([4], [2], training_examples=4800, epochs=3, reference_epochs=3)[0]
        self.assertEqual(changed_reference["expected_examples_per_request"], expected)
        self.assertAlmostEqual(changed_reference["normalized_expected_cost"], low["normalized_expected_cost"] / 3)
        self.assertEqual(changed_reference["reference_epochs"], 3)

    def test_invalid_counts_and_nonfinite_epochs_are_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalized_expected_cost(value, 2)
                with self.assertRaises(ValueError):
                    normalized_expected_cost(2, value)
                with self.assertRaises(ValueError):
                    expected_cost(2, 2, value)
        for epochs in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                cost_grid([2], [2], epochs=epochs)
            with self.assertRaises(ValueError):
                cost_grid([2], [2], reference_epochs=epochs)
        for requests in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                cost_grid([2], [2], requests=requests)
        with self.assertRaises(ValueError):
            expected_cost(4, 4, 15)
        with self.assertRaises(ValueError):
            cost_grid([], [1])
        with self.assertRaises(ValueError):
            cost_grid([1], [])

    def test_cli_csv_json_and_invalid_input(self):
        command = [sys.executable, str(ROOT / "sisa_cost.py"), "--shards", "4", "10", "--slices", "1", "2"]
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True, check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["expected_total_examples"], "")
        result = subprocess.run(command + ["--training-examples", "4800", "--requests", "25", "--format", "json"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)[0]["expected_total_examples"], 30000)
        result = subprocess.run(command + ["--epochs", "3", "--reference-epochs", "2", "--format", "json"],
                                capture_output=True, text=True, check=True)
        row = json.loads(result.stdout)[0]
        self.assertEqual(row["normalized_expected_cost"], 0.375)
        self.assertEqual(row["reference_epochs"], 2)
        result = subprocess.run(command + ["--epochs", "nan"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("epochs must be finite and positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
