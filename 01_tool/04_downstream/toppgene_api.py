"""Unchanged ToppGene client blocks; no RF or benchmark code."""
from __future__ import annotations
import argparse
import json
import math
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
LOOKUP_URL = "https://toppgene.cchmc.org/API/lookup?pretty=true"
ENRICH_URL = "https://toppgene.cchmc.org/API/enrich?pretty=true"

@dataclass(frozen=True)
class DegCollection:
    collection: str
    deg_test: str
    unit_version: str
    unit_type: str
    table_dir: Path
    pair_table: Path | None = None


def ordered_unique(items: list[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value.lower() != "nan" and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def post_json(url: str, payload: dict[str, object], max_retries: int, sleep_seconds: float, timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(1, int(max_retries) + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "Codex-ToppGene/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            time.sleep(float(sleep_seconds) * attempt)
    raise RuntimeError(last_error)


def flatten_toppgene(annotations: list[dict[str, object]], collection: DegCollection, unit_id: str, unit_type: str) -> pd.DataFrame:
    rows = []
    for item in annotations:
        genes = item.get("Genes", []) or []
        rows.append(
            {
                "collection": collection.collection,
                "deg_test": collection.deg_test,
                "unit_version": collection.unit_version,
                "unit_id": unit_id,
                "unit_type": unit_type,
                "Category": item.get("Category", ""),
                "ID": item.get("ID", ""),
                "Name": item.get("Name", ""),
                "PValue": item.get("PValue", ""),
                "QValueFDRBH": item.get("QValueFDRBH", ""),
                "QValueFDRBY": item.get("QValueFDRBY", ""),
                "QValueBonferroni": item.get("QValueBonferroni", ""),
                "GenesInTerm": item.get("GenesInTerm", ""),
                "GenesInQuery": item.get("GenesInQuery", ""),
                "GenesInTermInQuery": item.get("GenesInTermInQuery", ""),
                "Source": item.get("Source", ""),
                "URL": item.get("URL", ""),
                "GeneSymbols": ";".join(str(g.get("Symbol", "")) for g in genes),
                "GeneEntrez": ";".join(str(g.get("Entrez", "")) for g in genes),
            }
        )
    return pd.DataFrame(rows)


def run_toppgene(
    gene_symbols: list[str],
    collection: DegCollection,
    unit_id: str,
    unit_type: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = ordered_unique(gene_symbols)
    pd.DataFrame({"gene_symbol": genes}).to_csv(output_dir / "01_input_genes.tsv", sep="\t", index=False)
    status = "completed"
    detail = ""
    mapped_entrez: list[int] = []
    annotations = pd.DataFrame()
    if len(genes) < args.min_mapped_genes:
        status = "skipped_insufficient_input_genes"
        lookup = {"Genes": []}
    else:
        try:
            lookup = post_json(LOOKUP_URL, {"Symbols": genes}, args.max_retries, args.sleep_seconds, args.request_timeout_seconds)
        except Exception as exc:
            status = "lookup_failed"
            detail = repr(exc)
            lookup = {"Genes": []}
    with open(output_dir / "02_lookup_raw.json", "w", encoding="utf-8") as handle:
        json.dump(lookup, handle, indent=2, ensure_ascii=False)
    lookup_rows = []
    mapped_submitted = set()
    for gene in lookup.get("Genes", []) or []:
        submitted = str(gene.get("Submitted", "")).strip()
        entrez = gene.get("Entrez", "")
        if submitted:
            mapped_submitted.add(submitted)
        if entrez != "":
            mapped_entrez.append(int(entrez))
        lookup_rows.append({"Submitted": submitted, "OfficialSymbol": gene.get("OfficialSymbol", ""), "Entrez": entrez})
    pd.DataFrame(lookup_rows).to_csv(output_dir / "03_lookup_mapped.tsv", sep="\t", index=False)
    pd.DataFrame({"Submitted": [g for g in genes if g not in mapped_submitted]}).to_csv(
        output_dir / "04_lookup_missing.tsv", sep="\t", index=False
    )
    mapped_entrez = [int(x) for x in ordered_unique(mapped_entrez)]
    if status == "completed" and len(mapped_entrez) < args.min_mapped_genes:
        status = "skipped_insufficient_mapped_genes"
    if status == "completed":
        try:
            time.sleep(float(args.sleep_seconds))
            enrich = post_json(ENRICH_URL, {"Genes": mapped_entrez}, args.max_retries, args.sleep_seconds, args.request_timeout_seconds)
            with open(output_dir / "05_toppgene_raw.json", "w", encoding="utf-8") as handle:
                json.dump(enrich, handle, indent=2, ensure_ascii=False)
            annotations = flatten_toppgene(enrich.get("Annotations", []) or [], collection, unit_id, unit_type)
        except Exception as exc:
            status = "enrich_failed"
            detail = repr(exc)
    if annotations.empty:
        annotations = pd.DataFrame(columns=["collection", "deg_test", "unit_version", "unit_id", "unit_type", "Category", "Name", "QValueFDRBH"])
    annotations.to_csv(output_dir / "06_toppgene_annotations.tsv", sep="\t", index=False)
    qvals = pd.to_numeric(annotations.get("QValueFDRBH", pd.Series(dtype=float)), errors="coerce")
    sig = annotations.loc[qvals < args.fdr_threshold].copy() if not annotations.empty else annotations.copy()
    sig.to_csv(output_dir / "07_toppgene_significant_fdrbh.tsv", sep="\t", index=False)
    top_q = math.nan
    top_term = ""
    top_category = ""
    if qvals.notna().any():
        idx = qvals.idxmin()
        top_q = float(qvals.loc[idx])
        top_term = str(annotations.loc[idx].get("Name", ""))
        top_category = str(annotations.loc[idx].get("Category", ""))
    summary = {
        "collection": collection.collection,
        "deg_test": collection.deg_test,
        "unit_version": collection.unit_version,
        "unit_id": unit_id,
        "unit_type": unit_type,
        "input_gene_count": len(genes),
        "mapped_gene_count": len(mapped_entrez),
        "n_annotations": int(annotations.shape[0]),
        "n_significant_fdrbh": int(sig.shape[0]),
        "top_category": top_category,
        "top_term": top_term,
        "top_qvalue_fdrbh": top_q,
        "status": status,
        "detail": detail,
        "unit_output_dir": str(output_dir),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "08_toppgene_summary.tsv", sep="\t", index=False)
    time.sleep(float(args.sleep_seconds))
    return summary, annotations

