# scBeacon

One shared implementation for two-condition scBeacon analysis, plus three
prepared-input manuscript examples: IPF, COVID airway and lung cancer.

## Start Here

- [User guide and input contract](04_docs/01_user_guide.md)
- [Reproduce the three dataset analyses](04_docs/02_reproduction.md)
- [Methods and interpretation](04_docs/03_methods.md)
- [Release boundaries and remaining publication decisions](04_docs/04_release_scope.md)
- [Codex skill](02_skills/scbeacon-analysis/SKILL.md)
- [Executed tutorial](04_docs/02_notebooks/01_quickstart.ipynb)
- [IPF results notebook](03_reproduction/01_ipf/04_notebooks/01_manuscript_results.ipynb)
- [COVID results notebook](03_reproduction/02_covid/04_notebooks/01_manuscript_results.ipynb)
- [Lung results notebook](03_reproduction/03_lung/04_notebooks/01_manuscript_results.ipynb)

The result notebooks are inside each dataset's `04_notebooks` directory under
`03_reproduction`. Their saved outputs are real executions that **read frozen
source values**, not new training runs. Open them on GitHub/Jupyter without
running cells to inspect the figures and tables. Individual plots are exported
under each dataset's `06_individual_panels`; Illustrator row layouts are not
the only available figure outputs.

## Installation

Use Python 3.11 and an isolated environment. On the original HPC, activate
environments with `source ~/<environment>/bin/activate` and run analytical code
inside a compute allocation, not on a login node.

```bash
python -m venv ~/scbeacon_public
source ~/scbeacon_public/bin/activate
python -m pip install -r 04_docs/01_environment/requirements-notebooks.txt
```

For training and downstream analysis, also install
`04_docs/01_environment/requirements-analysis.txt`. Choose a PyTorch build that
matches your GPU/driver. These are dependency specifications, not a claim that
the exact historical environment has been recovered. See `06_metadata` for the
environment actually used for notebook validation.

## Commands

Run from this directory, after activating your environment:

```bash
python scbeacon.py inspect --dataset ipf
python scbeacon.py inspect --dataset covid --inputs
python scbeacon.py plan --dataset lung --output /your/work/lung_run
python scbeacon.py render --dataset ipf --output /your/work/ipf_panels
```

`train`, `pairs-deg`, `qc`, `markers` and `enrich` are plan-only unless
`--execute` is supplied. Live ToppGene calls also require `--allow-network`.
The plan command prints stage commands; it does not submit jobs. All execution
outputs must be new directories outside this release. Nothing is deleted or
overwritten automatically. On a personal workstation only, explicitly set
`SCBEACON_ALLOW_LOCAL_COMPUTE=1`; never set it on an HPC login node.

## Contents

```text
01_tool/           Shared model, preserved training recipe, portable interfaces
02_skills/         scBeacon analysis skill
03_reproduction/   Dataset-specific configurations, data, references, notebooks
04_docs/           User guide, methods, tutorial and environment specifications
05_tests/          Interface and frozen-value validation
06_metadata/       Copy identities, checksums and validation reports
```

Large H5AD/NPY input files are present in this local copy but intentionally
excluded by `.gitignore`. A code-only Git clone therefore needs the separately
distributed prepared-input package before retraining. The frozen source tables
and executed notebooks can be inspected without those large files. No external
data archive URL is invented here; choose and document the archive before public
release. Do not use `git add -f` to bypass this boundary.

This repository is a manuscript reproducibility candidate. Uploading the
repository does not constitute a final software release, a license choice or
authorization to redistribute the large input datasets.
