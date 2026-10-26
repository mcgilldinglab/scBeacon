"""Lightweight tests; no training, DEG recomputation, live API or baseline runs."""

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from workflow import config_for, settings, fresh_output, load_module
from results import table, summary


class ReleaseTests(unittest.TestCase):
    def test_counts_cohorts_and_plot_tables(self):
        for dataset, expected_n, expected_pairs in (("ipf", 18030, 9), ("covid", 56485, 20), ("lung", 88144, 23)):
            with self.subTest(dataset=dataset):
                report = summary(dataset)
                self.assertEqual(report["n_cells"], expected_n)
                self.assertEqual(report["clean_pairs"], expected_pairs)
                cells = table(dataset, "umap.tsv.gz")
                self.assertTrue(cells.cell_id.is_unique)
                self.assertTrue(np.isfinite(cells[["umap_1", "umap_2"]]).all().all())
                rows = table(dataset, "heatmap_rows.tsv")
                self.assertEqual(len(rows), 2 * expected_pairs)
                self.assertTrue((rows.groupby("pair_id").size() == 2).all())
                columns = table(dataset, "heatmap_columns.tsv")
                matrix = table(dataset, "heatmap_zscore.tsv")
                self.assertEqual(matrix.shape, (len(rows), len(columns) + 1))
                self.assertEqual(matrix.row_id.tolist(), rows.row_id.tolist())
                self.assertEqual(matrix.columns[1:].tolist(), columns.column_id.tolist())
                self.assertTrue(np.isfinite(matrix.iloc[:, 1:]).all().all())
                terms = table(dataset, "selected_pathways.tsv")
                self.assertTrue((terms.groupby("pair_label").size() == 10).all())
                self.assertTrue(terms.qvalue.between(0, 1).all())
                self.assertEqual(set(terms.unit_id) - set(table(dataset, "clean_pairs.tsv").pair_id), set())

    def test_portable_configuration(self):
        for dataset in ("ipf", "covid", "lung"):
            raw = json.loads(config_for(dataset).read_text())
            for key in ("condition1_input", "condition2_input", "combined_lognorm", "embedding", "reference_dir"):
                self.assertFalse(Path(raw[key]).is_absolute())
                self.assertTrue(Path(settings(config_for(dataset))[key]).exists())

    def test_protected_output(self):
        with self.assertRaises(ValueError):
            fresh_output(ROOT / "03_reproduction")

    def test_stats_against_frozen_background(self):
        stats = load_module(ROOT / "01_tool/04_downstream/edge_statistics.py", "test_stats")
        for dataset in ("ipf", "covid", "lung"):
            cfg = settings(config_for(dataset))
            original = Path(cfg["reference_dir"]).parent / "05_original_source_tables"
            background = pd.read_csv(original / f"{dataset}_knn30_all_cluster_pair_background.tsv", sep="\t")
            formal = pd.read_csv(original / f"{dataset}_formal_kl_pairs_knn30_empirical_pvalues.tsv", sep="\t")
            def matrix(row, col, size, fraction):
                n = np.rint(background[fraction] * background[size] * 30).astype(int)
                return pd.DataFrame({"r": background[row].astype(str), "c": background[col].astype(str), "n": n}).pivot_table(index="r", columns="c", values="n", aggfunc="sum", fill_value=0)
            gs = next(c for c in ("group_cluster_size_from_assignment", "group_cluster_size") if c in background)
            rs = next(c for c in ("reference_cluster_size_from_assignment", "reference_cluster_size") if c in background)
            forward = matrix("group_cluster", "reference_cluster", gs, "knn30_group_to_reference_fraction")
            reverse = matrix("reference_cluster", "group_cluster", rs, "knn30_reference_to_group_fraction")
            fp, rp = [], []
            for row in formal.itertuples():
                fp.append(stats.fisher_from_matrix(forward, str(row.group_cluster), str(row.reference_cluster))[-1])
                rp.append(stats.fisher_from_matrix(reverse, str(row.reference_cluster), str(row.group_cluster))[-1])
            fq, rq = stats.bh_qvalues(fp), stats.bh_qvalues(rp)
            kept = set()
            for i, row in enumerate(formal.itertuples()):
                size = min(float(getattr(row, "reference_cluster_size")), float(getattr(row, "group_cluster_size")))
                if size >= 100 and fq[i] < .05 and rq[i] < .05:
                    kept.add(row.pair_id)
            self.assertEqual(kept, set(table(dataset, "clean_pairs.tsv").pair_id))

    def test_directed_edge_count_toy(self):
        from downstream import directed_edges
        rng = np.random.default_rng(7)
        x = rng.normal(size=(35, 3))
        y = rng.normal(size=(40, 3))
        a = np.array([str(i % 2) for i in range(len(x))])
        b = np.array([str(i % 3) for i in range(len(y))])
        actual = directed_edges(x, y, a, b)
        expected = actual * 0
        for i in range(len(x)):
            nearest = np.argsort(((y - x[i]) ** 2).sum(axis=1))[:30]
            for j in nearest:
                expected.loc[a[i], b[j]] += 1
        pd.testing.assert_frame_equal(actual, expected)

    def test_pathway_values_against_archived_toppgene(self):
        selector = load_module(ROOT / "01_tool/04_downstream/pathway_display.py", "test_pathway_display")
        for dataset in ("ipf", "covid", "lung"):
            frozen = table(dataset, "selected_pathways.tsv")
            cfg = settings(config_for(dataset))
            for unit, expected in frozen.groupby("unit_id", sort=False):
                path = Path(cfg["reference_dir"]).parent / "04_archived_downstream/02_toppgene" / unit / "06_toppgene_annotations.tsv"
                actual = pd.read_csv(path, sep="\t")
                actual = actual.loc[actual.Category.eq("Pathway") & actual.Source.isin(["Reactome Pathways", "WikiPathways"])].copy()
                actual["qvalue_fdr_bh"] = pd.to_numeric(actual.QValueFDRBH)
                actual["pvalue"] = pd.to_numeric(actual.PValue)
                actual["term_clean"] = actual.Name.map(selector.clean_term)
                actual = selector.select_nonoverlap_terms(actual.loc[actual.qvalue_fdr_bh < .05])
                expected = expected.sort_values("rank")
                self.assertEqual(actual.ID.tolist(), expected.ID.tolist(), (dataset, unit))
                np.testing.assert_allclose(actual.qvalue_fdr_bh, expected.qvalue, rtol=1e-12)

    def test_compute_guard_without_allocation(self):
        import os
        import subprocess
        env = {k: v for k, v in os.environ.items() if k not in {"SLURM_JOB_ID", "PBS_JOBID", "SCBEACON_ALLOW_LOCAL_COMPUTE"}}
        result = subprocess.run([sys.executable, str(ROOT / "scbeacon.py"), "qc", "--config", str(config_for("ipf")), "--pair-run", "/tmp/not_a_real_pair_run", "--output", "/tmp/not_a_real_output", "--execute"], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Use a compute allocation", result.stderr)

    def test_scientific_entry_imports_without_execution(self):
        runner = load_module(ROOT / "01_tool/02_training/run_balanced_seed_task.py", "test_training_recipe")
        core = ROOT / "01_tool/01_shared_scbeacon_core"
        config, autoencoder, trainer = runner.import_core(core)
        self.assertTrue(callable(config))
        self.assertTrue(callable(autoencoder))
        self.assertTrue(callable(trainer))
        self.assertTrue(callable(runner.import_train(core)))
        for name in ("formal_pairs", "toppgene_api", "pathway_display"):
            load_module(ROOT / "01_tool/04_downstream" / (name + ".py"), "test_import_" + name)
        import prepare
        import new_results

    def test_plan_handoffs_use_new_embedding(self):
        import subprocess
        result = subprocess.run([sys.executable, str(ROOT / "scbeacon.py"), "plan", "--dataset", "lung", "--output", "/tmp/example_new_training_run"], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        commands = plan["commands"]
        pairs = next(command for command in commands if "pairs-deg" in command)
        self.assertIn("/tmp/example_new_training_run/01_train/03_contrastive/contrastive_gene_gene_sets/iter_33_contrastive_final.npy", pairs)
        self.assertEqual(commands[-1][2], "plot-run")


if __name__ == "__main__":
    unittest.main()
