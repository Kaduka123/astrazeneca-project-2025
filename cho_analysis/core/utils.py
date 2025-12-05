# cho_analysis/core/utils.py
"""Utility functions for CHO cell line analysis."""

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directory(directory_path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory.

    Returns:
        The directory path.
    """
    Path.mkdir(directory_path, exist_ok=True)
    return directory_path


def get_gene_info(df: pd.DataFrame, gene_id: str) -> dict[str, str | pd.Series]:
    """Get information about a specific gene from the expression DataFrame.

    Args:
        df: Expression DataFrame.
        gene_id: Gene ID or symbol to look up.

    Returns:
        Dictionary containing gene information.

    Raises:
        ValueError: If gene is not found.
    """
    result: dict[str, str | pd.Series] = {}

    # Check if the gene ID is directly in the index
    if gene_id in df.index:
        result["gene_id"] = gene_id
        result["expression"] = df.loc[gene_id]

        # Include metadata if available
        if isinstance(df, pd.DataFrame) and "sym" in df.columns:
            result["symbol"] = df.loc[gene_id, "sym"]

        return result

    # Check if there's a 'sym' column that might contain the gene
    if "sym" in df.columns:
        gene_rows = df[df["sym"] == gene_id]
        if not gene_rows.empty:
            gene_row = gene_rows.iloc[0]
            result["gene_id"] = gene_row.name
            result["symbol"] = gene_id

            # Get expression values (exclude metadata columns)
            if "ensembl_transcript_id" in df.columns:
                expr_cols = df.columns[3:]  # Assume first 3 columns are metadata
                result["expression"] = gene_row[expr_cols]
            else:
                # If we don't know the structure, just return all numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns
                result["expression"] = gene_row[numeric_cols]

            return result

    # If we didn't find the gene, raise an error
    msg = f"Gene '{gene_id}' not found in the expression data"
    raise ValueError(msg)


def normalize_expression(
    df: pd.DataFrame, method: str = "log1p", exclude_cols: list[str] | None = None
) -> pd.DataFrame:
    """Normalize expression values using various methods.

    Args:
        df: Expression DataFrame.
        method: Normalization method ('log1p', 'log2', 'zscore', 'quantile').
        exclude_cols: List of column names to exclude from normalization.

    Returns:
        Normalized DataFrame.

    Raises:
        ValueError: If an unknown normalization method is specified.
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Determine which columns to normalize
    if exclude_cols is None:
        exclude_cols: list[str] = []

    if "ensembl_transcript_id" in df.columns:
        # Assume the first 3 columns are metadata
        metadata_cols: list[str] = ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
        exclude_cols.extend([col for col in metadata_cols if col in df.columns])

    # Get columns to normalize
    numeric_cols: list[str] = df.select_dtypes(include=["number"]).columns
    cols_to_normalize: list[str] = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_normalize:
        return result

    # Apply normalization method
    if method == "log1p":
        result[cols_to_normalize] = np.log1p(df[cols_to_normalize])
    elif method == "log2":
        # Add small value to avoid log(0)
        result[cols_to_normalize] = np.log2(df[cols_to_normalize] + 1e-10)
    elif method == "zscore":
        # Z-score normalization (zero mean, unit variance)
        result[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[
            cols_to_normalize
        ].std()
    elif method == "quantile":
        # Quantile normalization
        from scipy.stats import rankdata

        # Apply quantile normalization to each column separately
        for col in cols_to_normalize:
            # Get ranks
            ranks = rankdata(df[col], method="average")
            # Scale to [0, 1]
            result[col] = ranks / len(ranks)
    else:
        msg: str = f"Unknown normalization method: {method}"
        raise ValueError(msg)

    return result


def find_constant_genes(
    df: pd.DataFrame, threshold: float = 0.2, exclude_cols: list[str] | None = None
) -> tuple[list[str], pd.DataFrame]:
    """Find genes with constant expression across samples.

    Args:
        df: Expression DataFrame.
        threshold: Maximum coefficient of variation to consider a gene constant.
        exclude_cols: List of column names to exclude from the calculation.

    Returns:
        Tuple of (list of constant gene IDs, DataFrame with only constant genes).
    """
    # Determine which columns to analyze
    if exclude_cols is None:
        exclude_cols = []

    if "ensembl_transcript_id" in df.columns:
        # Assume the first 3 columns are metadata
        metadata_cols = ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
        exclude_cols.extend([col for col in metadata_cols if col in df.columns])

    # Get columns to analyze
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_analyze = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_analyze:
        return [], pd.DataFrame()

    # Calculate coefficient of variation for each gene
    means = df[cols_to_analyze].mean(axis=1)
    stds = df[cols_to_analyze].std(axis=1)

    # Avoid division by zero
    cv = stds / (means + 1e-10)

    # Find genes with CV below threshold
    constant_genes = cv[cv <= threshold].index.tolist()

    # Return list of constant genes and the filtered DataFrame
    return constant_genes, df.loc[constant_genes]
