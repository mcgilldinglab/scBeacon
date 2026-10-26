"""Offline views of frozen manuscript source values, one scientific plot per file."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"ipf": "01_ipf", "covid": "02_covid", "lung": "03_lung"}


def folder(dataset):
    return ROOT / "03_reproduction" / DATASETS[dataset]


def table(dataset, name):
    path = dataset / name if isinstance(dataset, Path) else folder(dataset) / "03_reference" / name
    return pd.read_csv(path, sep="\t")


def style():
    fonts = {x.name for x in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in fonts else "Liberation Sans" if "Liberation Sans" in fonts else "DejaVu Sans"
    mpl.rcParams.update({"font.family": font, "font.size": 5, "axes.labelsize": 5, "axes.titlesize": 5, "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300, "axes.linewidth": 0.5})
    return font


def summary(dataset):
    cells = table(dataset, "umap.tsv.gz")
    pairs = table(dataset, "clean_pairs.tsv")
    terms = table(dataset, "selected_pathways.tsv")
    cfg = json.loads((folder(dataset) / "01_configuration/dataset.json").read_text())
    return {"dataset": dataset, "result_mode": "archived_source_values", "n_cells": len(cells), "condition_counts": {str(k): int(v) for k, v in cells.condition.value_counts().items()}, "selected_seed": cfg["seed"], "iteration": 33, "clean_pairs": len(pairs), "selected_enrichment_pairs": list(terms.pair_label.unique())}


def umap(dataset, condition, by="cell_type"):
    style()
    cells = table(dataset, "umap.tsv.gz")
    frame = cells.loc[cells.condition == condition]
    colors = table(dataset, "colors.tsv")
    palette = colors.loc[colors.kind == by].set_index("label").color.to_dict()
    fig, ax = plt.subplots(figsize=(91.5 / 25.4, 65 / 25.4))
    fig.subplots_adjust(left=.12, bottom=.16, right=.68, top=.95)
    labels = sorted(frame[by].unique(), key=(lambda x: int(x[1:])) if by == "cluster" else str)
    for label in labels:
        part = frame.loc[frame[by] == label]
        ax.scatter(part.umap_1, part.umap_2, s=1.2, linewidths=0, color=palette.get(label, "#777777"), rasterized=True, label=label)
    for axis, column in ((ax.set_xlim, "umap_1"), (ax.set_ylim, "umap_2")):
        lower, upper = cells[column].min(), cells[column].max()
        padding = max((upper - lower) * .08, 1e-6)
        axis(lower - padding, upper + padding)
    ax.set(xlabel="UMAP1", ylabel="UMAP2", xticks=[], yticks=[])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, markerscale=2.5, handletextpad=.4, ncol=2 if len(labels) > 17 else 1, columnspacing=.8, handlelength=.7, labelspacing=.35)
    return fig


def network(dataset):
    style()
    pairs = table(dataset, "clean_pairs.tsv")
    left = sorted(pairs.control_cluster.astype(int).unique())
    right = sorted(pairs.ipf_cluster.astype(int).unique())
    a = dict(zip(left, np.linspace(.94, .06, len(left))))
    b = dict(zip(right, np.linspace(.94, .06, len(right))))
    fig, ax = plt.subplots(figsize=(61 / 25.4, 100 / 25.4))
    fig.subplots_adjust(left=.05, right=.95, top=.94, bottom=.04)
    for row in pairs.itertuples():
        ax.add_patch(FancyArrowPatch((.25, a[int(row.control_cluster)]), (.75, b[int(row.ipf_cluster)]), arrowstyle="-|>", mutation_scale=4, linewidth=.5, color="#557886", shrinkA=2, shrinkB=2))
    for pos, nodes, prefix, x in ((a, left, "C", .25), (b, right, "D", .75)):
        for node in nodes:
            ax.plot(x, pos[node], "o", markersize=2.8, color="#2A6F7F")
            ax.text(x + (-.04 if prefix == "C" else .04), pos[node], prefix + str(node), va="center", ha="right" if prefix == "C" else "left")
    conditions = table(dataset, "umap.tsv.gz").condition.drop_duplicates().tolist()
    ax.text(.25, 1, conditions[0], ha="center")
    ax.text(.75, 1, conditions[1], ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1.04))
    ax.set_axis_off()
    return fig


def heatmap(dataset):
    style()
    values = table(dataset, "heatmap_zscore.tsv").set_index("row_id")
    rows = table(dataset, "heatmap_rows.tsv").set_index("row_id").loc[values.index]
    cols = table(dataset, "heatmap_columns.tsv").set_index("column_id").loc[values.columns]
    height = {"ipf": 65, "covid": 90, "lung": 130}.get(dataset, max(65, 2.8 * len(rows)))
    fig, ax = plt.subplots(figsize=(183 / 25.4, height / 25.4))
    fig.subplots_adjust(left=.19, right=.95, bottom=min(.32, 21 / height), top=.90)
    mesh = ax.pcolormesh(np.arange(len(cols) + 1), np.arange(len(rows) + 1), values.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, rasterized=False)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + .5, cols.gene_symbol, rotation=90)
    ax.set_yticks(np.arange(len(rows)) + .5, rows.row_label)
    ax.tick_params(length=0, pad=2)
    for tick, flag in zip(ax.get_xticklabels(), cols.highlight):
        if bool(flag):
            tick.set_color("#C62828")
    for i in range(2, len(rows), 2):
        ax.axhline(i, color="white", linewidth=.5)
    for pair, group in cols.reset_index().groupby("pair_id", sort=False):
        indices = np.flatnonzero(cols.pair_id.to_numpy() == pair)
        start, stop = int(indices.min()), int(indices.max()) + 1
        label = rows.loc[rows.pair_id == pair, "pair_label"].iloc[0]
        ax.text((start + stop) / 2, -0.8, label.replace("->", "\n->\n"), ha="center", va="bottom", clip_on=False, fontsize=5, linespacing=.85)
        ax.axvline(start, color="white", linewidth=.5)
    color_ax = fig.add_axes([.965, .38, .008, .23])
    bar = fig.colorbar(mesh, cax=color_ax, ticks=[-1, 0, 1])
    bar.ax.set_title("Z", fontsize=5)
    return fig


def pathways(dataset, pair_label):
    style()
    data = table(dataset, "selected_pathways.tsv")
    data = data.loc[data.pair_label == pair_label].sort_values("rank")
    fig, ax = plt.subplots(figsize=(91.5 / 25.4, 55 / 25.4))
    fig.subplots_adjust(left=.66, right=.94, top=.90, bottom=.18)
    y = np.arange(len(data))
    ax.barh(y, -np.log10(data.qvalue.clip(lower=1e-300)), height=.65, color="#537E8A")
    ax.set_yticks(y, [textwrap.fill(x, 40) for x in data.term])
    for label, flag in zip(ax.get_yticklabels(), data.highlight):
        if bool(flag):
            label.set_color("#C62828")
    ax.invert_yaxis()
    ax.set_xlabel("-log10 adjusted p-value")
    ax.set_title(pair_label)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=2, pad=2)
    return fig


def export_panels(dataset, output, reference=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(reference) if reference else dataset
    cells = table(source, "umap.tsv.gz")
    font = style()
    saved = []
    def save(fig, name):
        for extension in ("pdf", "svg", "png"):
            path = output / f"{dataset}_{name}.{extension}"
            fig.savefig(path, dpi=300)
            saved.append(path.name)
        plt.close(fig)
    for n, condition in enumerate(cells.condition.unique(), 1):
        for by in ("cell_type", "cluster"):
            save(umap(source, condition, by), f"umap_condition{n}_{by}")
    save(network(source), "clean_pair_network")
    save(heatmap(source), "marker_heatmap")
    for pair in table(source, "selected_pathways.tsv").pair_label.unique():
        save(pathways(source, pair), "pathways_" + pair.replace("->", "_to_"))
    (output / "render_manifest.json").write_text(json.dumps({"mode": "new_scientific_run" if reference else "redrawn_from_frozen_numerical_values", "font": font, "font_size_pt": 5, "raster_dpi": 300, "vector_except_umap_cloud": True, "files": saved}, indent=2))
    return saved
