# Train From Prepared H5AD Inputs To New Figures

## What This Tutorial Does

The primary tutorials now execute the real shared model and downstream stages:

1. Read both prepared raw-count H5ADs and their combined normalized-log H5AD.
2. Train the shared model: 40 pretraining epochs, then 33 iterations with 20
   reconstruction epochs and 5 contrastive epochs per iteration.
3. Use that new run's encoder embedding for separate-condition Leiden1 clusters,
   centroid KL pairs and disease-versus-control Wilcoxon DEG.
4. Apply kNN30 bidirectional Fisher/BH QC, minimum100 cells and both q<.05.
5. Compute marker mean expression and heatmap values from these retained pairs.
6. Submit all mapped passing disease-up DEG to live ToppGene.
7. Recompute UMAP and render the newly computed network, heatmap and pathways.

Intermediate TSV files are written by this new run, then consumed by later
stages. They are not historical tables used instead of model training.

## Choose A Notebook

- [General end-to-end tutorial](02_notebooks/02_train_to_figures.ipynb).
- [IPF](../03_reproduction/01_ipf/04_notebooks/02_train_to_figures.ipynb).
- [COVID](../03_reproduction/02_covid/04_notebooks/02_train_to_figures.ipynb).
- [Lung](../03_reproduction/03_lung/04_notebooks/02_train_to_figures.ipynb).

The `01_manuscript_results.ipynb` notebooks remain archived figure viewers. Their
saved outputs do not demonstrate execution of the training tutorial.

## Current Server Inputs

On the manuscript server the prepared input directory is:

```text
/home/kailu/links/projects/rrg-junding/kailu/scBeacon_codex/02_github_public_release/03_reproduction
```

It contains `01_ipf/02_data`, `02_covid/02_data` and `03_lung/02_data`. Set the
input root to the directory containing those three dataset folders, not to an
individual H5AD. Each run uses two raw-count H5ADs and the combined log-normalized
H5AD. It does not need `selected_embedding.npy`. Inputs stay read-only.

On another server, unpack the same prepared-input package under the same three
dataset folder names and set SCBEACON_INPUT_ROOT accordingly. Large H5ADs are
not in a code-only Git clone; there is not yet a published input archive URL.

## Compute And Environment

Run Jupyter in a GPU compute allocation with CUDA-capable PyTorch and both
requirements files installed. Activate an existing environment using
`source ~/<environment>/bin/activate`. Do not execute Python cells on an HPC
login node and do not set SCBEACON_ALLOW_LOCAL_COMPUTE there.

Set these in the compute-node terminal before starting Jupyter:

```bash
export SCBEACON_INPUT_ROOT=/home/kailu/links/projects/rrg-junding/kailu/scBeacon_codex/02_github_public_release/03_reproduction
export SCBEACON_RUN_ROOT=/your/new/writable/output/location
```

Choose an output filesystem with enough byte and file quota for training states,
DEG, API responses and figures. On the original server scratch quota was already
exhausted during packaging; do not assume scratch is usable without checking.
Notebook defaults use a new directory under `~/scbeacon_new_runs`; change this
before long training if home quota is unsuitable.

The live enrichment cell requires allow_network=True and outbound HTTPS to
ToppGene. It sends gene symbols, not matrices or cell metadata. If compute nodes
have no internet, explicitly reopen this run on a network-enabled compute node
or personal workstation with access to the run and inputs. Do not move Python
computation to the HPC login node and do not substitute archived responses.

## Execution And Continuation

Run the new notebook from top to bottom. Training cells launch subprocesses;
they are not command-printing previews. Inspect long-running output under
08_logs/train.log. Every successful stage has a JSON status record and required
output checks. A failed stage stops the notebook rather than loading old results.

To continue after a kernel restart, explicitly call
PreparedInputPipeline.open(RUN_ROOT) and begin at the next uncompleted stage.
Completed outputs are not overwritten or silently reused as new calculations.
For a fresh rerun, use a new run root. Do not remove checkpoints to force reuse.

## Output Tree

```text
01_configuration/run.json
02_training/03_contrastive/contrastive_gene_gene_sets/iter_33_contrastive_final.npy
03_pairs_deg/                   Formal pairs and full newly computed DEG
04_qc/                         All Fisher tests and retained clean pairs
05_markers/                    New mean-expression and z-score tables
06_enrichment/                 New per-pair queries and complete API responses
07_figures/01_source_values/    New UMAP coordinates and plotting tables
07_figures/02_individual_panels/
08_logs/                       Executed commands and stage status records
```

## No Historical Numerical Fallback

The new run configuration removes archived embedding, reference_dir and expected
clean-pair count. No old UMAP, pair list, DEG, pathway p-value or plot table is
loaded. The optional IPF gene-ID-to-symbol map is static feature annotation,
not an expression result. All clean pairs are plotted by default. Choosing
display_pair_labels explicitly affects presentation only and never forces a
pair through QC.

Training and downstream code are shared with the tool, not reimplemented in
notebooks. Final counts and term lists may differ with software/CUDA/API versions.
No old result is substituted to force 9/20/23 pairs or manuscript p-values.
Disease-up DEG means adjusted p<.05 and log2FC>1 relative to health/control.

## Validation Status

These new notebooks are executable tutorial definitions, not completed full-model
runs. They contain no invented training outputs. At authoring time there was no
active allocation and Slurm reported compute nodes unavailable, so their Python
cells, new orchestration tests and complete GPU training have not been executed.
The prior frozen-result notebook validation does not validate these new cells.

When compute is available, first run the new orchestration tests and input
validation there, then execute and save the live notebook using an appropriate
GPU allocation. Never label simulated-subprocess unit tests as model validation.

```bash
python -m unittest discover -s 05_tests -p test_live_pipeline.py -v
jupyter nbconvert --execute --to notebook --output-dir /your/executed_notebooks --ExecutePreprocessor.timeout=-1 03_reproduction/01_ipf/04_notebooks/02_train_to_figures.ipynb
```
