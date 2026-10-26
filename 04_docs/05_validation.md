# Validation Performed

All executable checks ran in a CPU compute allocation, not a login node.
No model training, real-data DEG recalculation, baseline fitting or live ToppGene
request was performed.

- Four notebooks executed top-to-bottom with saved table/figure outputs and no
  error outputs. Each dataset notebook contains eight embedded plot outputs.
- Nine tests cover cohort/heatmap joins, relative-path configs, output protection,
  compute guards, shared scientific-module imports, stage handoffs, toy directed
  kNN edge counts, archived Fisher clean membership and archived ToppGene term
  identities/q-values.
- Copied prepared inputs and the selected embedding have matching ordered cell
  and feature dimensions. Actual input counts are in validation_report.json.
- All24 individual panels passed text/legend canvas-boundary checks after the
  long cluster legends were split into two columns.
- Original source/data files referenced by the copy map were rehashed unchanged.
  Copies are independent files, not source symlinks or hardlinks.

Review the machine-readable records in06_metadata rather than treating this
summary as certification of a fresh training run. The shared model retains
existing literal-identity SyntaxWarnings in KLCLRv4.py; they do not prevent the
import checks, and model source was not edited to suppress them.

The notebook validation environment is recorded in notebook_environment.json.
It combines the existing analysis interpreter with separately installed notebook
dependencies. It is not claimed to be the exact historical training environment.

## Recheck

Inside your compute allocation, with analysis/notebook dependencies available:

```bash
python -m unittest discover -s 05_tests -v
python 05_tests/check_figure_layouts.py
```

The build_notebooks.py script refuses overwriting reviewed notebooks by default.
For an intentional refresh of generated notebook outputs use
`--execute --refresh-generated-notebooks`. The same deliberate refresh flag for
the full validation renderer is `--refresh-generated-panels`.
