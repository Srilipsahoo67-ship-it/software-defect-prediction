"""Shared dataset normalization helpers."""

import pandas as pd


def normalize_bug_labels(labels: pd.Series) -> pd.Series:
    """Convert defect counts and common boolean strings to nullable binary labels."""
    numeric = pd.to_numeric(labels, errors="coerce")
    text_labels = (
        labels.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "yes": 1, "no": 0})
    )
    values = numeric.fillna(text_labels)
    binary = (values > 0).astype("Int64")
    return binary.where(values.notna(), pd.NA)
