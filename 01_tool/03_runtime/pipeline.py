"""Real prepared-H5AD to model-to-figure stages for the notebook tutorials.

No archived embedding, DEG, coordinates or enrichment values are substituted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from workflow import ROOT, DATASETS, compute_only, config_for, fresh_output, validate_inputs


class PreparedInputPipeline:
    """Execute the preserved shared scientific stages in a new output tree."""

    STAGES = {
        "train": "02_training",
        "pairs-deg": "03_pairs_deg",
        "qc": "04_qc",
        "markers": "05_markers",
        "enrich": "06_enrichment",
        "plot-run": "07_figures",
    }

    def __init__(self, run_root):
        self.root = Path(run_root).expanduser().resolve()
        self.config_path = self.root / "01_configuration/run.json"
        self.config = json.loads(self.config_path.read_text())

    @classmethod
    def create(cls, dataset, input_root, run_root, use_cuda=True, display_pair_labels=None):
        """input_root contains 01_ipf/, 02_covid/, 03_lung/ with 02_data/."""
        compute_only()
        if dataset not in DATASETS:
            raise ValueError(f"Choose one of {tuple(DATASETS)}")
        cfg = json.loads(config_for(dataset).read_text())
        input_base = Path(input_root).expanduser().resolve() / DATASETS[dataset] / "01_configuration"
        for key in ("condition1_input", "condition2_input", "combined_lognorm"):
            cfg[key] = str((input_base / cfg[key]).resolve())
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(
                    f"Missing prepared input {cfg[key]}. Set SCBEACON_INPUT_ROOT to the "
                    "directory containing the dataset folders. A code-only Git clone "
                    "does not contain the large H5AD input package."
                )
        # These keys would otherwise permit a historical model/figure fallback.
        for key in ("embedding", "reference_dir", "expected_clean_pairs"):
            cfg.pop(key, None)
        cfg["use_cuda"] = bool(use_cuda)
        cfg["use_archived_display_annotations"] = False
        cfg["use_archived_display_pairs"] = False
        if display_pair_labels is not None:
            cfg["display_pair_labels"] = list(display_pair_labels)
        # Gene identifier -> symbol is static feature annotation, not a result.
        symbols = ROOT / "03_reproduction" / DATASETS[dataset] / "03_reference/gene_symbols.tsv"
        if symbols.is_file():
            cfg["gene_symbol_map"] = str(symbols.resolve())
        root = Path(run_root).expanduser().resolve()
        for key in ("condition1_input", "condition2_input", "combined_lognorm"):
            if root.is_relative_to(Path(cfg[key]).parent):
                raise ValueError("Run outputs must not be placed inside the prepared-input directory")
        fresh_output(root)
        cfg["output_dir"] = str(root / cls.STAGES["train"])
        config_dir = root / "01_configuration"
        config_dir.mkdir()
        (config_dir / "run.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (root / "08_logs").mkdir()
        return cls(root)

    @classmethod
    def open(cls, run_root):
        """Explicitly reopen one's own run to continue at an unfinished stage.

        Completed stages are never silently reused or reported as newly trained.
        """
        compute_only()
        return cls(run_root)

    @property
    def embedding(self):
        return self.stage_dir("train") / "03_contrastive/contrastive_gene_gene_sets/iter_33_contrastive_final.npy"

    @property
    def clean_pairs(self):
        return self.stage_dir("qc") / "clean_pairs.tsv"

    def stage_dir(self, stage):
        return self.root / self.STAGES[stage]

    def check_inputs(self):
        compute_only()
        # No embedding exists yet and none is read during input validation.
        return validate_inputs(self.config)

    def command(self, stage, allow_network=False):
        command = [sys.executable, str(ROOT / "scbeacon.py"), stage,
                   "--config", str(self.config_path), "--output", str(self.stage_dir(stage)), "--execute"]
        if stage == "pairs-deg":
            command += ["--embedding", str(self.embedding)]
        elif stage in ("qc", "markers", "enrich", "plot-run"):
            command += ["--pair-run", str(self.stage_dir("pairs-deg"))]
        if stage in ("markers", "enrich"):
            command += ["--clean-pairs", str(self.clean_pairs)]
        if stage == "enrich" and allow_network:
            command += ["--allow-network"]
        if stage == "plot-run":
            command += ["--qc-run", str(self.stage_dir("qc")),
                        "--marker-run", str(self.stage_dir("markers")),
                        "--enrichment-run", str(self.stage_dir("enrich"))]
        return command

    def _required_outputs(self, stage):
        out = self.stage_dir(stage)
        if stage == "train":
            return [self.embedding]
        if stage == "pairs-deg":
            return [out / "run.json", out / "formal_pairs.tsv", out / "02_deg/deg_unit_summary.tsv"]
        if stage == "qc":
            return [self.clean_pairs, out / "all_tests.tsv"]
        if stage == "markers":
            return [out / "heatmap_columns.tsv", out / "heatmap_zscore.tsv"]
        if stage == "plot-run":
            return [out / "run_manifest.json", out / "02_individual_panels/render_manifest.json"]
        return []

    def _execute(self, stage, allow_network=False):
        compute_only()
        if stage == "enrich" and not allow_network:
            raise ValueError("Set allow_network=True to submit DEG symbols to ToppGene on a network-enabled compute node or personal workstation. No archived enrichment is substituted.")
        dependencies = {"pairs-deg": "train", "qc": "pairs-deg", "markers": "qc", "enrich": "markers", "plot-run": "enrich"}
        if stage in dependencies:
            previous = self.root / "08_logs" / (dependencies[stage] + ".json")
            if not previous.is_file() or json.loads(previous.read_text()).get("status") != "completed":
                raise RuntimeError(f"Complete this run's {dependencies[stage]} stage first")
        output = self.stage_dir(stage)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {output}. Use a new run root, or explicitly continue at an unfinished stage.")
        log = self.root / "08_logs" / (stage + ".log")
        command = self.command(stage, allow_network)
        record = {"stage": stage, "command": command, "status": "running",
                  "archived_numerical_fallback": False, "output": str(output), "log": str(log)}
        started = time.monotonic()
        print(f"Executing {stage}; full output: {log}", flush=True)
        try:
            with log.open("x") as handle:
                subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
            for artifact in self._required_outputs(stage):
                if not artifact.is_file():
                    raise FileNotFoundError(f"Stage returned but its required output is missing: {artifact}")
            if stage == "pairs-deg":
                paired_embedding = Path(json.loads((output / "run.json").read_text())["embedding"]).resolve()
                if paired_embedding != self.embedding.resolve():
                    raise ValueError("Pair stage did not use the embedding trained in this run")
            record["status"] = "completed"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            if log.is_file():
                with log.open() as handle:
                    print("".join(deque(handle, maxlen=20)), flush=True)
            raise
        finally:
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            (self.root / "08_logs" / (stage + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        return output

    def train(self):
        """Run full pretraining and all 33 training iterations; not a preview."""
        compute_only()
        if self.config.get("use_cuda", True):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("The training tutorial needs a CUDA-capable compute allocation. No CPU training or archived embedding fallback was started.")
        return self._execute("train")

    def pairs_deg(self):
        return self._execute("pairs-deg")

    def qc(self):
        return self._execute("qc")

    def markers(self):
        return self._execute("markers")

    def enrich(self, allow_network=False):
        return self._execute("enrich", allow_network)

    def plot(self):
        return self._execute("plot-run")
