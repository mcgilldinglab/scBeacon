# User Guide

## Input Contract

Supply two AnnData H5AD files, one per condition, containing nonnegative integer
raw counts in `.X`, with identical ordered features and unique cell identifiers.
Each `.obs` needs an annotation column. The preserved balanced training-resolution
recipe uses annotation-based evaluation in resolution selection; annotations are
not exclusively for display in that recipe. Downstream KL/Fisher selection does
not use cell-type whitelists or force transitions.

Manuscript reproduction starts at the already prepared 2,000-feature H5ADs.
Do not refilter or reselect genes. For new data, jointly filter cells with fewer
than 200 detected genes and genes expressed in fewer than 3 cells. Normalize the
full-feature matrix to 10,000 counts/cell, log1p and select 2,000 HVGs with Scanpy's
default highly_variable_genes procedure. Preserve both raw and normalized-log
values for those same ordered features. Do not renormalize only the selected
2,000-feature subset as a replacement for the full-gene normalization.

The encoder uses log1p(raw), the decoder target raw counts, and DEG the separate
prepared normalized-log matrix. Combined rows and embedding rows must contain
condition1 first, then condition2, with `condition::barcode` combined cell IDs.

## Configuration and Training

Copy `01_tool/05_examples/user_pair.json` to your own working directory. Set your
input paths, condition names and annotation keys. Relative paths resolve against
the JSON location. Compatibility profiles preserve the available model variants:
`ipf09` or `covid09`; lung uses `covid09`. Changing preprocessing or profiles is
a new analysis, not a byte-identical manuscript reproduction.

For new full-feature raw inputs, set the two input paths in your JSON and run
the optional preparation interface on compute. It creates a prepared_config.json
with paths to the new locked raw/log matrices. This must not be run on the
already prepared manuscript inputs.

```bash
python scbeacon.py prepare --config /your/work/full_raw_pair.json --output /your/work/prepared --execute
```

```bash
python scbeacon.py train --config /your/work/user_pair.json --output /your/work/train
```

This previews the configuration without training. After allocating appropriate
GPU resources and activating the environment:

```bash
python scbeacon.py train --config /your/work/user_pair.json --output /your/work/train --execute
```

The preserved schedule uses pretrain40, annotation20 and contrastive5 epochs,
33 iterations and batch512. Balanced resolution selection chooses the training
resolution. Figure Leiden resolution1 refers to subsequent separate-condition
clustering, not a fixed training resolution. The CLI neither submits jobs nor
silently subsamples cells. It refuses nonempty output directories.

`iter_33_contrastive_final.npy` is a cell embedding, not pretrained model weights.
Training states/weights, when produced by the core, remain in the new run folder.

## Downstream Analysis

See `plan --dataset ...` and the reproduction guide for stage handoffs. Parameters
are explicit, not parameter-free: Leiden1, kNN30, minimum100 cells/cluster, and
both directed Fisher BH q<.05. DEG is Wilcoxon, disease against healthy reference;
adjusted p<.05 and log2FC>1. Positive log2FC means disease-up. Cell-level tests do
not make cells from the same donor independent biological replicates.

ToppGene receives all mapped passing disease-up DEG, not only the five heatmap
markers. It is a live external service: mappings and annotations change. Live
replay requires `--allow-network`; missing symbols and failures are recorded.
To preserve manuscript p-values use the frozen responses and selected-term tables.

The skill is `02_skills/scbeacon-analysis`; it is not installed globally by this
release preparation. The executed tutorials use the same CLI and core. Default
notebook execution reads frozen results and never trains or calls ToppGene.
