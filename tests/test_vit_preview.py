"""Validate the six-run GPU preview before launching expensive training."""
from pathlib import Path
import subprocess
import sys
import unittest

from deletion_requests import validate_workload
from dataset_splits import load_manifest
from studies.shared_comparison.preview import ROOT, STUDY, prepare_plan

BLOCK = ROOT.parent / "BlockUnlearn/unlearning"


@unittest.skipUnless((BLOCK / "run_sweep.py").exists(), "Sibling BlockUnlearn checkout required")
class ViTPreviewTests(unittest.TestCase):
    def test_six_commands_share_requests_full_pools_and_device_isolation(self):
        plan = prepare_plan("unit_preview_plan", ["0", "1"], BLOCK)
        self.assertEqual(len(plan["commands"]), 6)
        self.assertEqual(plan["protocol"]["epochs"], 1)
        self.assertEqual(plan["protocol"]["sisa"]["slices"], 2)
        for dataset, device in (("cifar10", "0"), ("cifar100", "1")):
            entry = plan["protocol"]["datasets"][dataset]
            workload = plan["workloads"][dataset]
            validate_workload(workload, load_manifest(ROOT / "splits" / entry["split"]))
            self.assertEqual(len(workload["batches"]), 3)
            commands = [item for item in plan["commands"] if item["dataset"] == dataset]
            self.assertEqual([item["method"] for item in commands], ["sisa", "su", "bm"])
            for item in commands:
                self.assertEqual(item["env"]["CUDA_VISIBLE_DEVICES"], device)
                if item["method"] == "sisa":
                    self.assertEqual(item["env"]["DELETION_REQUEST_MANIFEST"], entry["workload"])
                    self.assertEqual(item["env"]["SISA_SEED"], "1")
                else:
                    self.assertIn("unlearning_args.request_manifest=\"" + entry["workload"] + "\"", item["argv"])
                    self.assertIn("dataset_args.max_train_examples=null", item["argv"])
                    self.assertIn("dataset_args.max_test_examples=null", item["argv"])
                    self.assertEqual(item["argv"][item["argv"].index("--device") + 1], "cuda:0")
        self.assertFalse((STUDY / "runs/unit_preview_plan").exists())

    def test_dry_run_from_any_directory_creates_no_run(self):
        completed = subprocess.run([sys.executable, str(STUDY / "preview.py"),
                                    "--tag", "unit_preview_dry_run", "--dry-run"],
                                   cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("six runs, 1 epoch, 3 deletions/dataset", completed.stdout)
        self.assertFalse((STUDY / "runs/unit_preview_dry_run").exists())


if __name__ == "__main__":
    unittest.main()
