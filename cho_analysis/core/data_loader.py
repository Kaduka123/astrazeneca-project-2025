# cho_analysis/core/data_loader.py
"""Data loading utilities for CHO cell line analysis."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import SeqIO conditionally
try:
    from Bio import SeqIO

    SEQIO_AVAILABLE = True
except ImportError:
    SEQIO_AVAILABLE = False

from cho_analysis.core.config import get_correlation_config, get_file_path

logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and basic processing of CHO cell line RNA-seq data."""

    def __init__(self):
        """Initialize DataLoader."""
        self.expression_data: pd.DataFrame | None = None
        self.manifest_data: pd.DataFrame | None = None
        self.cds_sequences: list[tuple[str, str]] | None = None
        self.utr5_sequences: list[tuple[str, str]] | None = None
        self.utr3_sequences: list[tuple[str, str]] | None = None
        if not SEQIO_AVAILABLE:
            logger.warning("Biopython not found. Sequence loading will be unavailable.")

    def load_expression_data(self) -> pd.DataFrame | None:
        """Load expression data from the file specified in config."""
        logger.info("Attempting to load expression data...")
        expression_path: Path | None = None
        try:
            expression_path = get_file_path("expression")
            self.expression_data = pd.read_csv(expression_path, sep="\t")
            if self.expression_data.empty:
                logger.warning(f"Expression data file is empty: {expression_path}")
            required_meta_cols = ["ensembl_transcript_id", "sym"]
            if not all(col in self.expression_data.columns for col in required_meta_cols):
                logger.warning(
                    f"Expression data at '{expression_path}' missing required metadata columns (e.g., {required_meta_cols})."
                )
            n_genes = self.expression_data.shape[0]
            potential_meta_cols = self.expression_data.select_dtypes(include="object").columns
            n_samples = self.expression_data.shape[1] - len(potential_meta_cols)
            logger.info(
                f"Loaded expression data: {n_genes} genes x {n_samples} samples (estimated) from {expression_path}"
            )
            return self.expression_data
        except FileNotFoundError:
            msg = f"Expression data file not found: {expression_path}"
            logger.error(msg)
            self.expression_data = None
            raise FileNotFoundError(msg) from None
        except (pd.errors.EmptyDataError, ValueError) as e:
            msg = f"Expression data file is empty or invalid: {expression_path}. Error: {e!s}"
            logger.error(msg)
            self.expression_data = None
            raise ValueError(msg) from e
        except pd.errors.ParserError as e:
            msg = f"Error parsing expression data file: {expression_path}. Error: {e!s}"
            logger.error(msg)
            self.expression_data = None
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading expression data from {expression_path}: {e!s}"
            logger.exception(msg)
            self.expression_data = None
            raise RuntimeError(msg) from e

    def load_manifest_data(self) -> pd.DataFrame | None:
        """Load sample metadata from the MANIFEST file specified in config."""
        logger.info("Attempting to load manifest data...")
        manifest_path: Path | None = None
        try:
            manifest_path = get_file_path("manifest")
            self.manifest_data = pd.read_csv(manifest_path, sep="\t")
            if self.manifest_data.empty:
                logger.warning(f"Manifest file is empty: {manifest_path}")
            if "Sample" not in self.manifest_data.columns:
                logger.warning(f"Manifest file '{manifest_path}' missing expected 'Sample' column.")
            logger.info(
                f"Loaded manifest data: {self.manifest_data.shape[0]} samples with {self.manifest_data.shape[1]} attributes from {manifest_path}"
            )
            return self.manifest_data
        except FileNotFoundError:
            msg = f"Manifest file not found: {manifest_path}"
            logger.exception(msg)
            self.manifest_data = None
            raise FileNotFoundError(msg) from None
        except (pd.errors.EmptyDataError, ValueError) as e:
            msg = f"Manifest file is empty or invalid: {manifest_path}. Error: {e!s}"
            logger.exception(msg)
            self.manifest_data = None
            raise ValueError(msg) from e
        except pd.errors.ParserError as e:
            msg = f"Error parsing manifest data file: {manifest_path}. Error: {e!s}"
            logger.exception(msg)
            self.manifest_data = None
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading manifest data from {manifest_path}: {e!s}"
            logger.exception(msg)
            self.manifest_data = None
            raise RuntimeError(msg) from e

    def _load_fasta_sequences(self, file_key: str) -> list[tuple[str, str]] | None:
        """Helper to load sequences from a FASTA file identified by its config key."""
        if not SEQIO_AVAILABLE:
            logger.error(f"Cannot load sequences for '{file_key}': Biopython is not installed.")
            return None
        logger.info(f"Attempting to load FASTA sequences using key '{file_key}'...")
        sequences = []
        fasta_path: Path | None = None
        try:
            fasta_path = get_file_path(file_key)
            with open(fasta_path) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    if not record.id or not record.seq:
                        logger.warning(
                            f"Skipping record with missing ID or sequence in {fasta_path}"
                        )
                        continue
                    sequences.append((record.id, str(record.seq)))
            if not sequences:
                logger.warning(f"FASTA file '{fasta_path}' loaded but contains no valid sequences.")
            else:
                logger.info(f"Loaded {len(sequences)} sequences from '{fasta_path}'.")
            return sequences
        except KeyError as e:  # Specific catch for get_file_path failure
            logger.exception(f"Configuration error finding path for key '{file_key}': {e}")
            return None
        except FileNotFoundError:
            logger.exception(f"FASTA file not found for key '{file_key}' at path: {fasta_path}")
            return None
        except Exception as e:
            logger.exception(
                f"Error loading or parsing sequences from {fasta_path} (key: '{file_key}'): {e}",
            )
            return None

    def load_cds_sequences(self) -> list[tuple[str, str]] | None:
        """Loads CDS sequences from the FASTA file specified in config."""
        self.cds_sequences = self._load_fasta_sequences("cds_sequences")
        return self.cds_sequences

    def load_utr5_sequences(self) -> list[tuple[str, str]] | None:
        """Loads 5' UTR sequences from the FASTA file specified in config."""
        self.utr5_sequences = self._load_fasta_sequences("utr5_sequences")
        return self.utr5_sequences

    def load_utr3_sequences(self) -> list[tuple[str, str]] | None:
        """Loads 3' UTR sequences from the FASTA file specified in config."""
        self.utr3_sequences = self._load_fasta_sequences("utr3_sequences")
        return self.utr3_sequences

    def load_all_data(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Load primary data files (expression and manifest)."""
        logger.info("Loading all primary data (Expression & Manifest)...")
        try:
            self.load_expression_data()
        except Exception:
            logger.exception("Failed during expression data loading step.")
        try:
            self.load_manifest_data()
        except Exception:
            logger.exception("Failed during manifest data loading step.")
        if self.expression_data is None or self.manifest_data is None:
            logger.warning("One or both primary data files (Expression/Manifest) failed to load.")
        return self.expression_data, self.manifest_data

    def load_all_sequence_data(self) -> tuple[list[tuple[str, str]] | None, ...]:
        """Loads all sequence data files (CDS, UTR5, UTR3)."""
        logger.info("Loading all sequence data (CDS, UTR5, UTR3)...")
        self.load_cds_sequences()
        self.load_utr5_sequences()
        self.load_utr3_sequences()
        if self.cds_sequences is None or self.utr5_sequences is None or self.utr3_sequences is None:
            logger.warning("One or more sequence types failed to load.")
        return self.cds_sequences, self.utr5_sequences, self.utr3_sequences

    def get_basic_stats(self) -> dict[str, Any]:
        """Get basic statistics about the loaded primary data."""
        logger.debug("Calculating basic data statistics...")
        if self.expression_data is None:
            logger.debug("Expression data not loaded, attempting load for stats...")
            try:
                self.load_expression_data()
            except Exception:
                msg = "Cannot calculate stats: Expression data failed to load."
                raise RuntimeError(msg)
        if self.manifest_data is None:
            logger.debug("Manifest data not loaded, attempting load for stats...")
            try:
                self.load_manifest_data()
            except Exception:
                logger.exception("Failed to load manifest data for stats calculation.")

        stats: dict[str, Any] = {"target_gene_present": False}
        if self.expression_data is not None:
            n_genes = self.expression_data.shape[0]
            potential_meta_cols = self.expression_data.select_dtypes(include="object").columns
            n_samples = self.expression_data.shape[1] - len(potential_meta_cols)
            expr_numeric = self.expression_data.select_dtypes(include=np.number)
            stats["num_genes"] = n_genes
            stats["num_samples"] = n_samples if n_samples >= 0 else self.expression_data.shape[1]
            stats["mean_expression"] = (
                float(expr_numeric.mean().mean()) if not expr_numeric.empty else 0.0
            )
            stats["median_expression"] = (
                float(expr_numeric.median().median()) if not expr_numeric.empty else 0.0
            )
            target_gene = get_correlation_config().get("target_gene", "PRODUCT-TG")
            if "sym" in self.expression_data.columns:
                target_row = self.expression_data[self.expression_data["sym"] == target_gene]
                if not target_row.empty:
                    stats["target_gene_present"] = True
                    target_numeric = target_row.select_dtypes(include=np.number)
                    stats["target_gene_mean"] = (
                        float(target_numeric.mean().mean()) if not target_numeric.empty else 0.0
                    )
                    stats["target_gene_median"] = (
                        float(target_numeric.median().median()) if not target_numeric.empty else 0.0
                    )
        else:
            stats.update(
                {"num_genes": 0, "num_samples": 0, "mean_expression": 0.0, "median_expression": 0.0}
            )
        if self.manifest_data is not None and "Lab" in self.manifest_data.columns:
            stats["num_labs"] = self.manifest_data["Lab"].nunique()
            stats["sample_batches"] = self.manifest_data["Lab"].value_counts().to_dict()
        else:
            stats.update({"num_labs": 0, "sample_batches": {}})
        logger.debug(f"Basic stats calculated: {stats}")
        return stats

    def validate_and_fix_residual_df(self, df):
        """Ensure the residual effects dataframe is properly formatted for visualization."""
        if df is None or df.empty:
            logger.warning("Empty or None residual effects dataframe")
            return df

        logger.info("DEBUGGING INDEX ISSUE:")
        logger.info(f"Type of df: {type(df)}")
        logger.info(f"Type of df.index: {type(df.index)}")
        logger.info(f"Sample of df.index[:5]: {list(df.index[:5])}")
        logger.info(f"df.index.dtype: {df.index.dtype}")

        # Convert the index completely, force to a standard format
        try:
            # Create a completely new dataframe with string index
            fixed_df = pd.DataFrame(
                df.values, index=[str(idx) for idx in df.index], columns=df.columns
            )

            logger.info(
                f"After conversion: index type: {type(fixed_df.index)}, dtype: {fixed_df.index.dtype}"
            )

            # Fill NaNs in significance column
            if "kruskal_pvalue_adj" in fixed_df.columns:
                fixed_df["kruskal_pvalue_adj"] = fixed_df["kruskal_pvalue_adj"].fillna(1.0)

            return fixed_df
        except Exception as e:
            logger.exception(f"Error fixing DataFrame: {e}")
            return df  # Return original as fallback
