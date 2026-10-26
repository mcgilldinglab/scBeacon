#!/usr/bin/env python3
"""Generate reviewable notebooks with genuine cached-result execution outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from workflow import compute_only


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SETUP = '''%matplotlib inline
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "scbeacon.py").is_file())
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from results import summary, table, umap, network, heatmap, pathways
'''


def write_execute(path, cells, execute, refresh=False):
    if path.exists() and not refresh:
        raise FileExistsError(f"Choose a new output name rather than overwriting a reviewed notebook: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    book = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}, "scbeacon": {"execution_mode": "archived_source_values", "scientific_training_executed": False, "live_enrichment_executed": False}})
    if execute:
        client = NotebookClient(book, timeout=600, resources={"metadata": {"path": str(path.parent)}})
        # Use the activated interpreter, not an arbitrary registered global kernel.
        client.km = client.create_kernel_manager()
        client.km.ip = "127.0.0.1"
        client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.execute()
    nbf.validate(book)
    nbf.write(book, path)
    return {"notebook": str(path.relative_to(ROOT)), "executed": execute, "code_cells": sum(c.cell_type == "code" for c in book.cells), "output_cells": sum(bool(c.get("outputs", [])) for c in book.cells), "errors": sum(o.output_type == "error" for c in book.cells for o in c.get("outputs", []))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--refresh-generated-notebooks", action="store_true")
    args = p.parse_args()
    compute_only()
    reports = []
    reports.append(write_execute(ROOT / "04_docs/02_notebooks/01_quickstart.ipynb", [
        md("# scBeacon Quickstart\n\n## Goal\nInspect three real prepared-input examples and learn how to invoke the shared tool.\n\n**Mode:** executed frozen-result tutorial, not GPU training, DEG recomputation or live ToppGene. Read the user guide before launching an expensive run."),
        md("## Setup\nRun on a compute node or your own workstation with requirements-notebooks installed. No large H5AD files or external API are needed for these default cells."),
        code(SETUP),
        md("## Steps\n### 1. Inspect the archived cohorts\nCell counts and clean-pair counts below are read from the actual frozen tables."),
        code('reports = [summary(d) for d in ("ipf", "covid", "lung")]\ndisplay(pd.DataFrame(reports)[["dataset", "n_cells", "selected_seed", "iteration", "clean_pairs"]])'),
        md("### 2. Inspect one real pair result\nThe selected IPF cross-type population match is not a demonstrated lineage trajectory."),
        code('display(table("ipf", "clean_pairs.tsv")[["pair_id", "pair_cluster_label", "forward_fisher_qvalue_bh", "reverse_fisher_qvalue_bh"]])'),
        md("### 3. Configure a new run\nThe JSON template uses raw counts for training and a separately prepared normalized-log matrix for DEG. Relative paths resolve against the JSON location. Training resolution selection uses the supplied annotations."),
        code('import json\nexample = json.loads((ROOT / "01_tool/05_examples/user_pair.json").read_text())\ndisplay(pd.Series(example, name="configuration"))'),
        md("### 4. Preview before execution\nCopy the JSON to your working directory and set real paths. The following are commands to run, not commands executed in this notebook. --execute is explicit; training requires GPU resources."),
        code('print("python scbeacon.py train --config /your/work/user_pair.json --output /your/work/train")\nprint("python scbeacon.py train --config /your/work/user_pair.json --output /your/work/train --execute")\nprint("python scbeacon.py plan --dataset covid --output /your/work/covid_run")'),
        md("## Checks\nNo training subprocesses, API requests or dataset modifications were performed."),
        code('assert [r["n_cells"] for r in reports] == [18030, 56485, 88144]\nassert [r["clean_pairs"] for r in reports] == [9, 20, 23]\nprint("Frozen cohorts verified; training was not run.")'),
        md("## Next Steps\nOpen each dataset's executed notebook for UMAP, pairs, marker heatmap and selected pathway results. See 04_docs/02_reproduction.md for full scientific rerun interfaces and their validation boundary."),
    ], args.execute, args.refresh_generated_notebooks))
    for dataset, numbered in (("ipf", "01_ipf"), ("covid", "02_covid"), ("lung", "03_lung")):
        cells = [
            md(f"# {dataset.upper()}: scBeacon Manuscript Results\n\n## tl;dr\nThis notebook exposes actual archived source values and redraws individual panels. It does not retrain a model or recompute enrichment. Display layout may differ from Illustrator; numerical values are retained."),
            md("## Context & Methods\nSeparate-condition Leiden1 KL candidates; cross-condition kNN30 directed edges; both cluster sizes >=100 and both one-sided Fisher BH q<.05. Disease-vs-control Wilcoxon DEG: adjusted p<.05, log2FC>1. The feature universe is the locked2,000 genes.\n\n### Key Assumptions\nFrozen UMAP coordinates and source expression/pathway tables are the reproduction starting point for these figure views. Shared graph edges and cells are not independent donor replicates. Cell-type annotation changes do not prove transitions."),
            code(SETUP + f'\nDATASET = "{dataset}"'),
            md("## Data\n### 1. Cohort and selected model\nCounts are computed from the supplied source tables."),
            code('report = summary(DATASET)\ndisplay(pd.Series(report, name="archived_result"))\ncells = table(DATASET, "umap.tsv.gz")\ndisplay(pd.crosstab(cells.cell_type, cells.condition))'),
            md("## Results\n### 2. Individual UMAP views\nAll cells are displayed; UMAP cloud is rasterized, text/axes vector. Coordinates are frozen and are not the kNN30 test space."),
            code('for condition in cells.condition.unique():\n    for by in ("cell_type", "cluster"):\n        print(condition, by)\n        fig = umap(DATASET, condition, by)\n        display(fig)\n        plt.close(fig)'),
            md("### 3. Final clean pairs\nDisplay arrows C->D do not reverse the original formal disease-to-reference matching calculation."),
            code('pairs = table(DATASET, "clean_pairs.tsv")\ndisplay(pairs[["pair_id", "pair_cluster_label", "reference_cluster_size", "group_cluster_size", "forward_fisher_qvalue_bh", "reverse_fisher_qvalue_bh"]])\nfig = network(DATASET)\ndisplay(fig)\nplt.close(fig)'),
            md("### 4. Marker mean-expression heatmap\nTwo rows per pair, control then disease; full gene-wise z-scores are stored. Color display clips at[-1,1]. Red gene labels preserve existing curated annotations; no new relevance assignments are inferred here."),
            code('columns = table(DATASET, "heatmap_columns.tsv")\ndisplay(columns[["pair_id", "gene_symbol", "pvals_adj", "logfoldchanges", "highlight"]].head(10))\nfig = heatmap(DATASET)\ndisplay(fig)\nplt.close(fig)'),
            md("### 5. Selected Reactome + WikiPathways\nAll passing mapped disease-up DEG contributed to the original enrichment. Top10 nonredundant display terms per selected pair are shown; bars use BH-adjusted p-values. These are archived API results, not a new online query."),
            code('terms = table(DATASET, "selected_pathways.tsv")\ndisplay(terms[["pair_label", "term", "PValue", "qvalue", "highlight"]])\nfor pair_label in terms.pair_label.unique():\n    fig = pathways(DATASET, pair_label)\n    display(fig)\n    plt.close(fig)'),
            md("## Checks\nEach heatmap pair must exist in the final clean set. Source values remain unchanged; no heavy analysis is launched."),
            code('rows = table(DATASET, "heatmap_rows.tsv")\nassert len(cells) == report["n_cells"]\nassert len(rows) == 2 * len(pairs)\nassert set(rows.pair_id) <= set(pairs.pair_id)\nassert set(terms.unit_id) <= set(pairs.pair_id)\nprint("Cohorts and selected-pair joins verified; cached numerical result mode.")'),
            md("## Takeaways\nUse these panels and tables for numerical figure reproduction. For retraining from the copied H5ADs, inspect the generated plan and run its stages in new output directories with suitable compute resources. Historical environment/snapshot identity and live database drift are documented in the reproduction guide."),
        ]
        reports.append(write_execute(ROOT / "03_reproduction" / numbered / "04_notebooks/01_manuscript_results.ipynb", cells, args.execute, args.refresh_generated_notebooks))
    report_path = ROOT / "06_metadata/notebook_execution.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
