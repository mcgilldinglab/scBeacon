"""Orchestration tests with explicit fake subprocesses, not scientific validation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from pipeline import PreparedInputPipeline
from workflow import DATASETS, config_for


class LivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.guard = patch("pipeline.compute_only")
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.addCleanup(self.temp.cleanup)

    def make_run(self, dataset="ipf", use_cuda=False):
        config = json.loads(config_for(dataset).read_text())
        data = self.base / "inputs" / DATASETS[dataset] / "02_data"
        data.mkdir(parents=True)
        for key in ("condition1_input", "condition2_input", "combined_lognorm"):
            (data / Path(config[key]).name).touch()
        return PreparedInputPipeline.create(dataset, self.base / "inputs", self.base / "new_run", use_cuda=use_cuda)

    def test_configuration_cannot_fall_back_to_archived_results(self):
        run = self.make_run()
        for name in ("embedding", "reference_dir", "expected_clean_pairs"):
            self.assertNotIn(name, run.config)
        self.assertFalse(run.config["use_archived_display_annotations"])
        self.assertFalse(run.config["use_archived_display_pairs"])
        for key in ("condition1_input", "condition2_input", "combined_lognorm"):
            self.assertTrue(Path(run.config[key]).is_absolute())
            self.assertEqual(Path(run.config[key]).suffix, ".h5ad")
        self.assertEqual(run.config["iterations"], 33)
        self.assertEqual(run.config["pretrain_epochs"], 40)

    def test_all_commands_execute_and_use_the_new_model(self):
        run = self.make_run()
        for stage in run.STAGES:
            command = run.command(stage, allow_network=stage == "enrich")
            self.assertIn("--execute", command)
            self.assertNotIn("render", command)
        pairs = run.command("pairs-deg")
        self.assertEqual(pairs[pairs.index("--embedding") + 1], str(run.embedding))
        self.assertIn("02_training", str(run.embedding))
        self.assertIn("--allow-network", run.command("enrich", allow_network=True))

    def test_input_validation_reads_no_existing_embedding(self):
        run = self.make_run()
        with patch("pipeline.validate_inputs", return_value={"embedding_shape": None}) as check:
            self.assertIsNone(run.check_inputs()["embedding_shape"])
            self.assertNotIn("embedding", check.call_args.args[0])

    def test_full_stage_handoffs_with_simulated_subprocesses(self):
        run = self.make_run()
        called = []

        def fake_child(command, **kwargs):
            stage = command[2]
            called.append(stage)
            for path in run._required_outputs(stage):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            if stage == "pairs-deg":
                (run.stage_dir(stage) / "run.json").write_text(json.dumps({"embedding": str(run.embedding)}))
            run.stage_dir(stage).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0)

        with patch("pipeline.subprocess.run", side_effect=fake_child):
            run.train()
            run.pairs_deg()
            run.qc()
            run.markers()
            run.enrich(allow_network=True)
            run.plot()
        self.assertEqual(called, list(run.STAGES))
        for stage in run.STAGES:
            status = json.loads((run.root / "08_logs" / (stage + ".json")).read_text())
            self.assertEqual(status["status"], "completed")
            self.assertFalse(status["archived_numerical_fallback"])

    def test_missing_prerequisite_does_not_launch_a_subprocess(self):
        run = self.make_run()
        with patch("pipeline.subprocess.run") as child:
            with self.assertRaises(RuntimeError):
                run.pairs_deg()
            child.assert_not_called()

    def test_network_opt_in_is_required(self):
        run = self.make_run()
        with patch("pipeline.subprocess.run") as child:
            with self.assertRaises(ValueError):
                run.enrich()
            child.assert_not_called()

    def test_training_failure_is_recorded_without_result_substitution(self):
        run = self.make_run()
        with patch("pipeline.subprocess.run", side_effect=subprocess.CalledProcessError(1, "simulated training failure")):
            with self.assertRaises(subprocess.CalledProcessError):
                run.train()
        record = json.loads((run.root / "08_logs/train.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(run.embedding.exists())

    def test_gpu_requirement_does_not_silently_start_cpu_training(self):
        run = self.make_run(use_cuda=True)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch("pipeline.subprocess.run") as child:
            with self.assertRaises(RuntimeError):
                run.train()
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
