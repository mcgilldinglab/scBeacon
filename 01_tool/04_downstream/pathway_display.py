"""Preserved pathway display heuristic; not a statistical significance test."""
import re
import pandas as pd

TOP_N = 10


TOKEN_OVERLAP_THRESHOLD = 0.72


JACCARD_THRESHOLD = 0.55


STOPWORDS = {
    "and",
    "are",
    "cell",
    "cells",
    "disease",
    "diseases",
    "for",
    "from",
    "gene",
    "genes",
    "human",
    "homo",
    "in",
    "of",
    "pathway",
    "pathways",
    "process",
    "regulation",
    "sapiens",
    "signaling",
    "the",
    "to",
    "via",
    "with",
}


def clean_term(text: object) -> str:
    value = str(text).strip()
    value = re.sub(r"^WP[_: -]*", "WP ", value, flags=re.IGNORECASE)
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def display_term(text: object) -> str:
    value = clean_term(text)
    value = re.sub(r"^REACTOME\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^WP\s+", "", value, flags=re.IGNORECASE)
    return value


def term_tokens(text: object) -> set[str]:
    value = clean_term(text).lower()
    value = value.replace("tgf-beta", "tgfbeta")
    value = value.replace("tgf beta", "tgfbeta")
    value = re.sub(r"\bwp\s*\d+\b", " ", value)
    value = re.sub(r"\breactome\b|\bwikipathways\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    tokens = []
    for token in value.split():
        if len(token) <= 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return set(tokens)


def is_redundant(candidate: set[str], selected: list[set[str]]) -> bool:
    if not candidate:
        return False
    for existing in selected:
        if not existing:
            continue
        overlap = len(candidate & existing)
        smaller_fraction = overlap / max(1, min(len(candidate), len(existing)))
        jaccard = overlap / max(1, len(candidate | existing))
        if smaller_fraction >= TOKEN_OVERLAP_THRESHOLD or jaccard >= JACCARD_THRESHOLD:
            return True
    return False


def select_nonoverlap_terms(unit_terms: pd.DataFrame) -> pd.DataFrame:
    unit_terms = unit_terms.sort_values(
        ["qvalue_fdr_bh", "pvalue", "Source", "term_clean"], kind="mergesort"
    ).copy()
    selected_indices: list[int] = []
    selected_tokens: list[set[str]] = []
    for idx, row in unit_terms.iterrows():
        tokens = term_tokens(row["term_clean"])
        if is_redundant(tokens, selected_tokens):
            continue
        selected_indices.append(idx)
        selected_tokens.append(tokens)
        if len(selected_indices) >= TOP_N:
            break
    return unit_terms.loc[selected_indices].copy()

