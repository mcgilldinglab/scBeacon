# Scientific Method Contract

## Representation and Pairing

User and reproduction modes call the same preserved shared core. For final
pairing, separately cluster each condition's final scBeacon embedding using
Scanpy neighbors and Leiden resolution1, downstream random state0. Average
embeddings within clusters, standardize the two centroid matrices separately
with sklearn.preprocessing.scale and use the preserved PyTorch
F.kl_div(log_softmax(input), softmax(target)) operation. Keep formal
disease/group-to-health/reference candidates. PyTorch's target/input KL
convention is preserved in code, not inferred from arrow names. Network C->D
arrows are presentation, not evidence of cell ancestry.

## Clean Rule

For each cell, find30 Euclidean nearest cells in the opposite condition on the
scBeacon embedding, not UMAP. Count directed cell-neighbor edges between clusters.
For each formal pair, in each direction, use this 2x2 contingency table:

| | Matched target cluster | Other target clusters |
|---|---:|---:|
| Source cluster | a | b |
| Other source clusters | c | d |

Apply one-sided Fisher exact test (greater), then separate BH corrections for
forward and reverse p-values across all formal pairs before size filtering.
Keep pairs with both clusters >=100 and both q<.05. Statistical function blocks
are copied unchanged from the finalized producer. No cell-type whitelist,
ATII-specific exception or custom connectivity-score threshold is added.

The rule has explicit choices k30/size100/q.05 and is not parameter-free. Graph
edges sharing cells are dependent; the operational Fisher filter is not
donor-level biological validation or proof of cell-type transitions.

## DEG and Heatmap

Use prepared normalized-log expression for Scanpy rank_genes_groups, Wilcoxon,
use_raw=False, pts=True, disease against healthy reference. The feature universe
is the locked2,000 genes, not whole transcriptome. Positive log2FC means disease-up.
Require adjusted p<.05 and log2FC>1.

Select five eligible disease-up markers per clean pair, ordered by adjusted p
ascending and log2FC descending. Plot mean normalized-log expression per matched
cluster and gene-wise z-scores across displayed rows, control then disease per
pair. A displayed color limit[-1,1] does not modify source z-scores. COVID host
markers exclude VIRAL_ before marker selection; this is not host-only training
or a change to the original DEG multiple-testing feature universe.

## Enrichment and Highlights

ToppGene uses all mapped passing disease-up DEG. Reactome Pathways and
WikiPathways were combined and ranked by BH q-value, with a token-overlap
display heuristic to reduce redundant top10 terms per pair. This heuristic is
not a significance test. Frozen terms retain p/q-values, IDs and databases.

Red labels preserve existing curated disease-relevance annotations and available
rationale/citations; they do not affect p-values or establish superiority to a
baseline. UMAP coordinates use cosine neighbors10, spectral initialization and
min_dist0.8; they are never used for the Euclidean kNN30 pair-QC statistic.
