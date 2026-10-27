# Prepared-Input Reproduction

## Modes

1. **Full scientific pipeline:** start from the prepared H5ADs and execute
   `02_train_to_figures.ipynb` for IPF, COVID or lung. It trains a new model,
   recomputes all downstream stages and renders only this run's results. See
   [the end-to-end tutorial](06_end_to_end_tutorial.md). Full GPU execution of
   these new notebooks has not yet been validated.
2. **Offline numerical figure reproduction:** load the frozen source tables,
   inspect saved notebook outputs and redraw independent panels. This mode was
   executed when preparing the release; it is not new model training.
3. **Manual scientific stage interfaces:** start at the copied prepared H5ADs, train the shared model,
   recompute formal pairs, QC and DEG, then optionally call ToppGene. Interfaces
   are supplied, but full retraining was not performed to assemble this release.

| Dataset | Prepared cells | Seed | Iteration | Training resolution | Clean pairs |
|---|---:|---:|---:|---:|---:|
| IPF | 4,807 Control + 13,223 IPF | 23 | 33 | 0.1 | 9 |
| COVID | 30,665 Healthy + 25,820 COVID+ | 4 | 33 | 0.1 | 20 |
| Lung | 42,995 Normal + 45,149 Tumor | 38 | 33 | 0.3 | 23 |

These are full prepared cohorts. Earlier COVID public-atlas preparation capped
Epithelial cells at 10,000/condition and kept the retained non-epithelial lineages;
later QC removed 3 Healthy cells. Final model and figure contain all 56,485
prepared cells. This release does not perform another random draw.

## Inspect and Redraw

On a compute node with the environment activated:

```bash
python scbeacon.py inspect --dataset covid --inputs
python scbeacon.py render --dataset covid --output /your/work/covid_panels
```

Output is one plot per PDF/SVG/PNG: four UMAPs, one clean network, one heatmap,
and one pathway figure per selected pair. Fonts are 5pt; Arial if available,
otherwise the local fallback recorded in render_manifest.json. Text/lines are
vector; UMAP clouds are rasterized at300dpi. Vector objects do not have a pixel
dpi. These are clean redraws, not identical Illustrator page layouts.

## Scientific Rerun

```bash
python scbeacon.py plan --dataset ipf --output /your/work/ipf_new_run
```

The plan prints commands, not jobs. Run stages with appropriate compute resources.
`pairs-deg` accepts the new training embedding explicitly. Later QC uses that
path from its run.json; it never silently reverts to the archived embedding.
`markers` uses only retained pairs. `enrich` uses all passing DEG, not just top5.
The final `plot-run` stage recomputes UMAP using cosine neighbors10, spectral
initialization and min_dist0.8, selects current Reactome/WikiPathways terms with
the preserved non-overlap rule, and renders the newly computed results. It does
not substitute frozen coordinates or p-values. Previously curated display
annotations may transfer by exact barcode only. New gene/term relevance is not
automatically colored red. If a requested manuscript display pair is missing
after a new fit, the interface reports that mismatch rather than forcing it back.

For a recomputation beginning at the archived final embedding, omit training:

```bash
python scbeacon.py pairs-deg --config 03_reproduction/01_ipf/01_configuration/dataset.json --output /your/work/ipf_pairs --execute
python scbeacon.py qc --config 03_reproduction/01_ipf/01_configuration/dataset.json --pair-run /your/work/ipf_pairs --output /your/work/ipf_qc --execute
```

The old `01_manuscript_results.ipynb` notebooks do not execute those scientific
reruns; the new `02_train_to_figures.ipynb` notebooks do invoke the actual stages.
Frozen ToppGene
artifacts for displayed pairs are in 04_archived_downstream/02_toppgene. Exact
displayed values are in 03_reference; source identities are recorded in
06_metadata/source_copy_map.tsv. Full clean-pair ranked DEG is archived separately.

The core and training recipe are available and shared for all three datasets.
Exact historical target-server source/environment identity, notably the lung
seed38 target snapshot, is not certified. This is a provenance limitation, not
a missing implementation. See the actual validation report, and do not equate
successful cached notebooks with independent GPU retraining validation.
