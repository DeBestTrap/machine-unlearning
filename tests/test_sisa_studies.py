import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from studies.artifacts import publish_reports, study_directory
from studies.sisa import build_plan


ROOT = Path(__file__).resolve().parents[1]


class SISAStudyTests(unittest.TestCase):
    def settings(self, slug):
        study = study_directory(slug)
        return study, json.loads((study / "configs/recipes.json").read_text())

    def test_each_study_owns_its_reports_and_original_launch_recipes(self):
        ownership = {
            "sisa_purchase": ["report-purchase.csv"],
            "sisa_vit_cifar10": ["report-vit.csv", "r-vit.csv"],
            "sisa_roberta_sst2": ["report-roberta.csv"],
        }
        for slug, reports in ownership.items():
            study, settings = self.settings(slug)
            self.assertTrue((study / "run.sh").is_file())
            self.assertTrue((study / "configs/original-launchers/run.sh").is_file())
            for name in reports:
                self.assertFalse((ROOT / name).exists())
                self.assertFalse((ROOT / name).is_symlink())
                self.assertTrue((study / "reports/historical" / name).is_file())

    def test_recipe_grids_preserve_the_available_launch_settings(self):
        study, settings = self.settings("sisa_vit_cifar10")
        commands = build_plan(settings, "shards", None, 1, "unit_grid", study, device="cuda:2")
        self.assertEqual([item["resolved_config"]["shards"] for item in commands], [1, 4, 16, 32, 64])
        self.assertTrue(all(item["env"]["CUDA_VISIBLE_DEVICES"] == "2" for item in commands))
        self.assertTrue(all(Path(item["argv"][14]).is_relative_to(study / "runs/unit_grid/reports") for item in commands))
        for slug, expected in (("sisa_purchase", [1e-6, 3e-6, 1e-5]),
                               ("sisa_roberta_sst2", [1e-3, 1e-4, 1e-5])):
            study, settings = self.settings(slug)
            commands = build_plan(settings, "lrs", 2, 0, "unit_grid", study, device="cpu")
            self.assertEqual([item["resolved_config"]["learning_rate"] for item in commands], expected)
            self.assertTrue(all(item["env"]["CUDA_VISIBLE_DEVICES"] == "" for item in commands))
            self.assertEqual(len({item["container"] for item in commands}), len(commands))

    def test_study_launchers_accept_positional_arguments_from_another_directory(self):
        for slug in ("sisa_purchase", "sisa_vit_cifar10", "sisa_roberta_sst2"):
            study, _ = self.settings(slug)
            tag = f"unit_{slug}"
            result = subprocess.run(["bash", str(study / "run.sh"),
                                     "2", "1", "--dry-run", "--tag", tag],
                                    cwd="/tmp", capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("/studies/", result.stdout)
            self.assertIn(f"/runs/{tag}/reports/", result.stdout)
        study, _ = self.settings("sisa_purchase")
        result = subprocess.run([sys.executable, str(study / "summarize.py")],
                                cwd="/tmp", capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("accuracy", result.stdout)

    def test_report_index_links_to_saved_results_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            run = study / "runs/first"
            (run / "reports").mkdir(parents=True)
            sources = {run / "reports/sisa.csv": "accuracy\n0.9\n",
                       run / "results.csv": "step,accuracy\n0,0.8\n"}
            for path, text in sources.items():
                path.write_text(text)
            before = {path: path.read_bytes() for path in sources}
            publish_reports(study, run)
            publish_reports(study, run)
            self.assertEqual((study / "reports/first/sisa.csv").resolve(), run / "reports/sisa.csv")
            self.assertEqual((study / "reports/first/results.csv").resolve(), run / "results.csv")
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
