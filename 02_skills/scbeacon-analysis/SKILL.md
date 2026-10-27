---
name: scbeacon-analysis
description: Run or explain two-condition scBeacon analysis, prepared-input IPF/COVID/lung reproduction, pair QC, DEG, enrichment and frozen-result notebooks using the shared scBeacon release. Use for scBeacon workflows, not unrelated single-cell or pathogen analyses.
---

# scBeacon Analysis

Locate the release root containing scbeacon.py and 01_tool. For new data read
04_docs/01_user_guide.md; for prepared examples read04_docs/02_reproduction.md.
Use the same shared core in both modes; do not implement a replacement model.

## Choose the Mode

- When the user asks to run from prepared inputs through the model to figures,
  use 04_docs/06_end_to_end_tutorial.md and the 02_train_to_figures.ipynb
  notebooks. They call PreparedInputPipeline and execute real training and all
  new downstream stages. Set the input root to the actual prepared H5AD package
  and choose a new output root. Do not replace this request with render, frozen
  table loading, printed commands or an archived results notebook.
- Inspect manuscript results through executed notebooks, inspect or render.
  Explain that frozen values are not a new training run.
- For user data, validate raw-count inputs and create a JSON in a new workspace.
  Preview train without --execute; confirm compute resources and authorization
  before expensive training.
- For reproduction, use the supplied dataset config and plan. Explicitly pass
  the selected/new embedding; do not substitute seed0 for manuscript seeds.

The new full-training notebooks have not yet been executed end-to-end. Report
that boundary until a real compute-node run has completed. Existing cached
notebook outputs and simulated orchestration tests do not certify model reruns.

## Preserve Meaning

Read04_docs/03_methods.md before changing stages or explaining statistics. The
encoder uses log1p(raw), decoder raw counts, DEG separate normalized-log data.
Do not renormalize the selected feature subset. The training-resolution recipe
uses labels for evaluation; do not advertise it as label-free.

Final QC is Leiden1 KL candidates, cross-condition Euclidean kNN30 edge counts,
both clusters >=100 and both directed Fisher BH q<.05. No cell-type whitelist or
forced ATII transition is permitted. Explain its explicit parameter choices and
that alignment is not proof of lineage conversion.

Live ToppGene requires --allow-network; frozen values preserve manuscript
p-values. Keep VIRAL_ host-marker exclusions distinct from training/DEG inputs.

## Boundaries

Keep release/source/input files immutable during analysis. Write only to new
external output directories; do not delete checkpoints to fix runs. On HPC use
a compute allocation and source ~/<environment>/bin/activate. Never bypass the
compute guard on a login node. Do not upload, commit, push or globally install
the skill without a separate request. Report actual execution mode and failures;
cached plotting is not training reproduction.
