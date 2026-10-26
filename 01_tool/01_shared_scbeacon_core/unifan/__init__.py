"""Minimal package init for the portable COVID baseline.

The original research tree eagerly imported a longer list of historical
modules here, but this portable baseline only ships the subset used by
`train_covid.py`. Keeping package init light avoids import-time failures
from missing legacy files while preserving the actual training code paths.
"""

__all__ = [
    "KLCLRv4",
    "annoclusterV4",
    "autoencoderV2",
    "classifier",
    "datasets1",
    "layers",
    "networks",
    "trainer",
    "utils",
]
