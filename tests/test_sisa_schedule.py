"""Schedule eligibility and early rejection for both SISA training paths."""
from pathlib import Path
import subprocess
import sys
import unittest

from sisa_schedule import epochs_per_slice


ROOT = Path(__file__).resolve().parents[1]


class SISAScheduleTests(unittest.TestCase):
    def test_valid_integer_schedules_preserve_nominal_example_budget(self):
        # Equal prefixes must sum to E full-shard passes without any rounding.
        for slices in range(1, 11):
            for epochs in range(1, 21):
                with self.subTest(slices=slices, epochs=epochs):
                    if 2 * epochs % (slices + 1):
                        with self.assertRaisesRegex(ValueError, "Fractional epochs"):
                            epochs_per_slice(epochs, slices)
                    else:
                        passes = epochs_per_slice(epochs, slices)
                        self.assertGreaterEqual(passes, 1)
                        self.assertAlmostEqual(
                            sum(passes * prefix / slices for prefix in range(1, slices + 1)), epochs)

    def test_invalid_types_and_nonpositive_counts(self):
        for value in (0, -1, 1.5, 2.0, True, float("nan"), float("inf"), "3"):
            for epochs, slices in ((value, 1), (1, value)):
                with self.assertRaises(ValueError):
                    epochs_per_slice(epochs, slices)

    def test_four_slices_requires_multiples_of_five_nominal_epochs(self):
        with self.assertRaisesRegex(ValueError, "positive multiple of 5"):
            epochs_per_slice(3, 4)
        self.assertEqual(epochs_per_slice(5, 4), 2)
        self.assertEqual(epochs_per_slice(10, 4), 4)

    def test_sisa_cli_rejects_invalid_schedule_before_dataset_or_model_setup(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "sisa.py"), "--train", "--slices", "4", "--epochs", "1",
             "--dataset", "/nonexistent/dataset", "--model", "nonexistent"],
            cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Fractional epochs per slice are not allowed", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
