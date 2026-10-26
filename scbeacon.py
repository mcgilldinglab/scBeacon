#!/usr/bin/env python3
"""Portable entry point; scientific training and downstream code live in 01_tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "01_tool/03_runtime"))
from workflow import main

if __name__ == "__main__":
    main()
