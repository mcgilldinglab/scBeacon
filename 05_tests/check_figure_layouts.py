#!/usr/bin/env python3
"""Check rendered text/legend extents for clipping, without changing figures."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from results import table, umap, network, heatmap, pathways
from workflow import compute_only


def main():
    compute_only()
    reports = []
    for dataset in ("ipf", "covid", "lung"):
        figures = []
        for condition in table(dataset, "umap.tsv.gz").condition.unique():
            for by in ("cell_type", "cluster"):
                figures.append((f"{condition}_{by}", umap(dataset, condition, by)))
        figures += [("network", network(dataset)), ("heatmap", heatmap(dataset))]
        figures += [(str(label), pathways(dataset, label)) for label in table(dataset, "selected_pathways.tsv").pair_label.unique()]
        for name, figure in figures:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            box = figure.bbox
            clipped = []
            for axis in figure.axes:
                objects = [*axis.texts]
                if axis.axison:
                    xlo, xhi = sorted(axis.get_xlim())
                    ylo, yhi = sorted(axis.get_ylim())
                    objects.extend(label for value, label in zip(axis.get_xticks(), axis.get_xticklabels()) if xlo <= value <= xhi)
                    objects.extend(label for value, label in zip(axis.get_yticks(), axis.get_yticklabels()) if ylo <= value <= yhi)
                    objects.extend([axis.xaxis.label, axis.yaxis.label, axis.title])
                if axis.get_legend() is not None:
                    objects.append(axis.get_legend())
                for obj in objects:
                    if not obj.get_visible() or hasattr(obj, "get_text") and not obj.get_text():
                        continue
                    ext = obj.get_window_extent(renderer)
                    if ext.x0 < box.x0 - 1 or ext.y0 < box.y0 - 1 or ext.x1 > box.x1 + 1 or ext.y1 > box.y1 + 1:
                        clipped.append(obj.get_text() if hasattr(obj, "get_text") else "legend")
            reports.append({"dataset": dataset, "figure": name, "clipped_objects": clipped})
            plt.close(figure)
    (ROOT / "06_metadata/figure_layout_check.json").write_text(json.dumps(reports, indent=2))
    if any(row["clipped_objects"] for row in reports):
        raise ValueError(json.dumps([row for row in reports if row["clipped_objects"]], indent=2))
    print(f"Checked {len(reports)} individual panels: no text or legend outside the figure canvas.")


if __name__ == "__main__":
    main()
