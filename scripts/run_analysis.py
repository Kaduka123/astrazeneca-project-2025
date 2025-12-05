#!/usr/bin/env python
# scripts/run_analysis.py

"""Main entry point for CHO cell line analysis pipeline."""

import argparse
import functools
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Third-party imports
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Rich imports for enhanced CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.stats import pearsonr

# Project-specific imports
# NOTE: .env is loaded by docker compose, no need for python-dotenv here
from cho_analysis.core.config import (
    get_batch_correction_config,
    get_correlation_config,
    get_logging_config,
    get_path,
    get_sequence_analysis_config,
)
from cho_analysis.core.data_loader import DataLoader

# Logging setup now uses RichHandler via logging.py
from cho_analysis.core.logging import setup_logging
from cho_analysis.core.visualization_utils import SAVE_DPI
from cho_analysis.task1.advanced_batch_correction import AdvancedBatchCorrection
from cho_analysis.task1.batch_correction import BatchCorrection
from cho_analysis.task1.correlation import GeneCorrelationAnalysis
from cho_analysis.task1.marker_panels import MarkerPanelOptimization
from cho_analysis.task1.ranking import GeneRanking
from cho_analysis.task1.visualization import CorrelationVisualization
from cho_analysis.task2.sequence_analysis import SequenceFeatureAnalysis
from cho_analysis.task2.visualization import SequenceFeatureVisualization

# --- Global Setup ---
logger = setup_logging(__name__)  # Gets logger configured with RichHandler
console = Console()  # Global Rich console instance

# ===================================================
#  === Utility Functions (Modified for Rich) ===
# ===================================================


def print_header(title: str) -> None:
    """Prints a title header using Rich Panel."""
    console.print(
        Panel(f"[bold cyan]{title}[/]", border_style="bold cyan", expand=False, padding=(0, 5))
    )


def print_section(title: str) -> None:
    """Prints a section header using Rich Rule."""
    console.rule(f"[bold blue]:arrow_right: {title} [/]", style="blue")


def _ensure_dir_exists(output_dir: Path) -> bool:
    # (Keep this helper as is)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {output_dir}: {e!s}")
            return False
    return True


def save_dataframe(
    df: pd.DataFrame | None, filename: str, output_dir: Path, index: bool = False
) -> None:
    """Saves a DataFrame to CSV, ensuring parent directory exists."""
    if df is None:
        logger.debug(f"Skipping save '{filename}': DataFrame is None.")
        return
    if df.empty:
        logger.info(f":cross_mark: Skipping save '{filename}': DataFrame is empty.")
        return

    filepath = output_dir / filename
    # --- Ensure the specific output directory exists ---
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Optional: Add writability check here too if needed
        # logger.debug(f"Ensured parent directory exists: {filepath.parent}")
    except OSError as e:
        logger.exception(f":cross_mark: Failed to create parent directory for '{filepath}': {e}")
        return  # Cannot save if directory creation failed

    # --- Attempt saving ---
    try:
        save_index = index if not isinstance(df.index, pd.MultiIndex) else True
        df.to_csv(filepath, index=save_index)
        logger.info(
            f":floppy_disk: [green]Saved table:[/green] {filepath.resolve()} (Index: {save_index})"
        )
    except OSError as e:
        # Catch permission errors specifically during write
        if e.errno == 13:  # EACCES (Permission denied)
            logger.exception(
                f":cross_mark: [bold red]Permission denied[/bold red] saving table '{filepath}'. Check volume permissions."
            )
        else:
            logger.exception(f":cross_mark: [bold red]Failed to save table '{filename}':[/] {e!s}")
    except Exception as e:
        logger.exception(
            f":cross_mark: [bold red]Unexpected error saving table '{filename}':[/] {e!s}"
        )


def save_visualization(
    fig: Figure | None, filename: str, output_dir: Path, dpi: int = SAVE_DPI
) -> None:
    """Saves a Matplotlib Figure, ensuring parent directory exists."""
    if fig is None:
        logger.warning(f":cross_mark: Skipping save '{filename}': Figure object is None.")
        return

    filepath = output_dir / filename
    # --- Ensure the specific output directory exists ---
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # logger.debug(f"Ensured parent directory exists: {filepath.parent}")
    except OSError as e:
        logger.exception(f":cross_mark: Failed to create parent directory for '{filepath}': {e}")
        plt.close(fig)  # Close figure even if saving failed
        return

    # --- Attempt saving ---
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        logger.info(
            f":chart_increasing: [green]Saved visualization:[/green] {filepath.resolve()} (DPI: {dpi})"
        )
    except OSError as e:
        if e.errno == 13:
            logger.exception(
                f":cross_mark: [bold red]Permission denied[/bold red] saving visualization '{filepath}'. Check volume permissions."
            )
        else:
            logger.exception(
                f":cross_mark: [bold red]Failed to save visualization '{filename}':[/] {e!s}"
            )
    except Exception as e:
        logger.exception(
            f":cross_mark: [bold red]Unexpected error saving visualization '{filename}':[/] {e!s}"
        )
    finally:
        # Free memory
        plt.close(fig)


def analysis_step(section_title: str) -> Callable[..., Any]:
    """Decorator using Rich for section headers and status."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Print section rule *before* execution
            console.rule(f"[bold blue]:play_button: {section_title} [/]", style="blue")
            start_time = time.time()
            result = None
            success = True
            try:
                result = func(*args, **kwargs)

            except Exception:
                logger.exception(f"Step '{section_title}' encountered a critical error")
                success = False  # Mark as failed
            finally:
                end_time = time.time()
                elapsed = end_time - start_time
                status_icon = ":white_check_mark:" if success else ":cross_mark:"
                status_color = "green" if success else "red"
                # Log completion status with icon and color
                logger.info(
                    f"{status_icon} [{status_color}]Step '{section_title}' {'finished' if success else 'failed'} in {elapsed:.2f}s.[/{status_color}]"
                )
            return result if success else None

        return wrapper

    return decorator


# ===================================================
#  === Task 1: Correlation Analysis Steps ===
# ===================================================
@analysis_step("1. Loading Data (Task 1)")
def load_data_task1() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    # (Keep internal logic as before)
    data_loader = DataLoader()
    expression_data, manifest_data = None, None
    try:
        expression_data, manifest_data = data_loader.load_all_data()
        if expression_data is None or expression_data.empty:
            logger.exception("Failed to load expression data.")
            return None, manifest_data
        # Log warning instead of failing if manifest is missing
        if manifest_data is None or manifest_data.empty:
            logger.warning("Manifest data not loaded. Platform-dependent steps may be affected.")

        meta_cols = [
            c
            for c in ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
            if c in expression_data.columns
        ]
        sample_count_est = expression_data.shape[1] - len(meta_cols)
        logger.info(
            f"Loaded expression data: {expression_data.shape[0]} genes x {sample_count_est} samples"
        )
        if manifest_data is not None:
            logger.info(
                f"Loaded manifest data: {manifest_data.shape[0]} samples x {manifest_data.shape[1]} attributes"
            )
    except FileNotFoundError as e:
        logger.exception(f"Data loading failed: File not found - {e}")
        return None, None
    except Exception as e:
        logger.exception(f"Unexpected error loading data for Task 1: {e}")
        return expression_data, manifest_data
    return expression_data, manifest_data


@analysis_step("2. Batch Correction")
def run_batch_correction(
    expression_data: pd.DataFrame,
    manifest_data: pd.DataFrame | None,
    target_gene: str,
    figures_dir: Path,
    tables_dir: Path,
    skip_batch_correction: bool,
    run_advanced_correction: bool,
    advanced_alpha: float,
    viz_runner: CorrelationVisualization,
) -> pd.DataFrame | None:
    """Detects and applies batch correction, optionally checking for and applying advanced correction.

    Returns:
        Corrected expression DataFrame, or the original if skipped/failed. Returns the most corrected version.
    """
    if skip_batch_correction:
        logger.info("Skipping batch correction as requested by flag.")
        return expression_data.copy()

    if manifest_data is None:
        logger.warning("Skipping batch correction: Manifest data is missing.")
        return expression_data.copy()

    final_corrected_data = expression_data.copy()
    initial_correction_applied = False
    advanced_correction_applied = False
    standard_correction_success = True  # Track if standard correction itself succeeded
    adv_batch_corrector = AdvancedBatchCorrection()
    batch_corrector = BatchCorrection()
    original_numeric_df = None
    target_gene_id = None  # Store the Ensembl ID if found

    try:
        # --- Initial Batch Effect Detection ---
        logger.info("Detecting initial batch effects...")
        manifest_copy = manifest_data.copy()
        platform_col = "platform"
        # Derive platform if necessary
        if platform_col not in manifest_copy.columns and "description" in manifest_copy.columns:
            logger.debug("Deriving 'platform' column from 'description'.")
            manifest_copy[platform_col] = (
                manifest_copy["description"]
                .str.extract(r"(HiSeq|NovaSeq|NextSeq)", expand=False)
                .fillna("Unknown")
            )
        elif platform_col not in manifest_copy.columns:
            logger.warning(
                "Cannot detect batch effects: '%s' or 'description' column missing.", platform_col
            )
            return expression_data.copy()  # Return original if no platform info

        # Perform detection
        batch_labels, platform_effect_pct, silhouette_score = batch_corrector.detect_batch_effect(
            expression_df=expression_data, manifest_df=manifest_copy
        )
        platforms = sorted([p for p in np.unique(batch_labels) if p != "Unknown"])

        if len(platforms) <= 1:
            logger.info(
                "Only one known platform detected or no platform info. No batch correction needed."
            )
            return expression_data.copy()

        logger.info(f"Detected {len(platforms)} platforms: {', '.join(platforms)}")
        logger.info(
            f"Initial Platform effect (ANOVA p<0.05): {platform_effect_pct:.1f}% genes affected"
        )
        logger.info(f"Initial Batch silhouette score (PCA): {silhouette_score:.4f}")

        # --- Standard Batch Correction (Conditional) ---
        apply_standard_correction = platform_effect_pct > 20 or abs(silhouette_score) > 0.1
        if apply_standard_correction:
            logger.info("Applying standard ComBat-Seq correction...")
            # Run standard correction
            corrected_data_standard, _ = batch_corrector.correct_batch(
                expression_df=expression_data,
                manifest_df=manifest_copy,
                target_col=target_gene,
                method="combat",
            )
            if corrected_data_standard is not None and not corrected_data_standard.empty:
                final_corrected_data = corrected_data_standard  # Update data to be used
                logger.info(":syringe: Standard batch correction applied successfully.")
                initial_correction_applied = True
                # Verify preservation immediately after correction
                _verify_target_preservation(expression_data, final_corrected_data, target_gene)
                # Plot standard correction PCA comparison
                try:
                    fig_bc = batch_corrector.compare_pca_stages(
                        raw_df=expression_data,
                        filtered_df=expression_data,  # Use raw as 'filtered' pre-correction
                        corrected_df=final_corrected_data,
                        manifest_df=manifest_copy,
                    )
                    save_visualization(fig_bc, "batch_correction_standard_pca.png", figures_dir)
                except Exception as e:
                    logger.warning(f"Standard PCA plot generation failed: {e}")
            else:
                logger.error("Standard batch correction failed. Proceeding with original data.")
                standard_correction_success = False
                final_corrected_data = expression_data.copy()  # Revert to original
                initial_correction_applied = False  # Mark as not applied
        else:
            logger.info("Standard batch correction skipped (effect deemed minimal).")

        # --- Prepare Original Numeric Data & Find Target ID ---
        # Needed for hierarchical preservation and final validation
        gene_id_col = "ensembl_transcript_id"
        sym_col = "sym"
        try:
            if gene_id_col in expression_data.columns:
                meta_cols = [
                    c
                    for c in [gene_id_col, sym_col, "ensembl_peptide_id"]
                    if c in expression_data.columns
                ]
                numeric_cols = [c for c in expression_data.columns if c not in meta_cols]
                # Find target ID robustly
                target_row = (
                    expression_data[expression_data[sym_col] == target_gene]
                    if sym_col in expression_data.columns
                    else pd.DataFrame()
                )
                if not target_row.empty:
                    target_gene_id = target_row[gene_id_col].iloc[0]
                # Prepare numeric df (genes as index)
                df_unique = expression_data.drop_duplicates(subset=[gene_id_col], keep="first")
                original_numeric_df = df_unique.set_index(gene_id_col)[numeric_cols].apply(
                    pd.to_numeric, errors="coerce"
                )
            # Check if index looks like gene IDs
            elif expression_data.index.name or not isinstance(expression_data.index, pd.RangeIndex):
                if target_gene in expression_data.index:
                    target_gene_id = target_gene  # ID is the symbol/name
                original_numeric_df = expression_data.select_dtypes(include=np.number).apply(
                    pd.to_numeric, errors="coerce"
                )
            else:
                msg = "Cannot determine gene ID source in original expression data."
                raise ValueError(msg)

            if original_numeric_df is None or original_numeric_df.empty:
                msg = "Preparation of original numeric data failed."
                raise ValueError(msg)
        except Exception as e:
            logger.exception(f"Error preparing original numeric data for advanced BC: {e}")
            run_advanced_correction = False  # Disable advanced checks if prep fails
        # Log warning if target ID wasn't found (crucial for preservation)
        if target_gene_id is None:
            logger.warning(
                f"Could not find ID for target gene '{target_gene}'. Hierarchical preservation might fail or be inaccurate."
            )

        # --- Advanced Correction Steps (Conditional) ---
        # Only run if standard correction was applied (or attempted but maybe failed slightly)
        # AND advanced checks are enabled AND we have the necessary data (original numeric & target ID)
        if (
            (initial_correction_applied or standard_correction_success)
            and run_advanced_correction
            and original_numeric_df is not None
        ):
            logger.info("--- Advanced Batch Correction Checks & Application ---")
            numeric_corrected_df = None  # Holds standard-corrected numeric data
            try:
                # Prepare numeric standard-corrected data (genes x samples)
                # Use the *current* state of final_corrected_data
                if gene_id_col in final_corrected_data.columns:
                    meta_cols_corr = [
                        c
                        for c in [gene_id_col, sym_col, "ensembl_peptide_id"]
                        if c in final_corrected_data.columns
                    ]
                    numeric_cols_corr = [
                        c for c in final_corrected_data.columns if c not in meta_cols_corr
                    ]
                    df_unique_corr = final_corrected_data.drop_duplicates(
                        subset=[gene_id_col], keep="first"
                    )
                    numeric_corrected_df = df_unique_corr.set_index(gene_id_col)[
                        numeric_cols_corr
                    ].apply(pd.to_numeric, errors="coerce")
                elif final_corrected_data.index.name or not isinstance(
                    final_corrected_data.index, pd.RangeIndex
                ):
                    numeric_corrected_df = final_corrected_data.select_dtypes(
                        include=np.number
                    ).apply(pd.to_numeric, errors="coerce")
                else:
                    msg = "Cannot determine structure of standard corrected data."
                    raise ValueError(msg)

                if numeric_corrected_df is None or numeric_corrected_df.empty:
                    msg = "Preparation of standard corrected numeric data failed."
                    raise ValueError(msg)

                # 1. Detect Residual Effects
                residual_effects_df = adv_batch_corrector.detect_residual_platform_effects(
                    corrected_df=numeric_corrected_df,
                    manifest_df=manifest_copy,
                    alpha=advanced_alpha,
                )
                if residual_effects_df is not None:
                    logger.info(
                        f"DEBUG: residual_effects_df shape: {residual_effects_df.shape}, index type: {type(residual_effects_df.index)}"
                    )
                    logger.info(
                        f"DEBUG: residual_effects_df columns: {residual_effects_df.columns}"
                    )
                    save_dataframe(
                        residual_effects_df, "residual_platform_effects.csv", tables_dir, index=True
                    )  # Save with gene index

                # Function to ensure residual_effects_df is properly formatted
                def validate_and_fix_residual_df(df):
                    """Ensure the residual effects dataframe is properly formatted for visualization."""
                    if df is None or df.empty:
                        logger.warning("Empty or None residual effects dataframe")
                        return df

                    # Make sure index is properly formatted (not a multi-index)
                    if isinstance(df.index, pd.MultiIndex):
                        logger.warning("Converting MultiIndex to regular Index")
                        df = df.reset_index().set_index("gene_id")

                    # Ensure index is string type
                    if df.index.dtype != "object":
                        logger.warning(f"Converting index from {df.index.dtype} to string")
                        df.index = df.index.astype(str)

                    # Check for NaN values in significance column
                    if (
                        "kruskal_pvalue_adj" in df.columns
                        and df["kruskal_pvalue_adj"].isnull().any()
                    ):
                        logger.warning(
                            f"NaN values in significance column: {df['kruskal_pvalue_adj'].isnull().sum()}"
                        )
                        # Fill NaN with a high value (non-significant)
                        df["kruskal_pvalue_adj"] = df["kruskal_pvalue_adj"].fillna(1.0)

                    return df

                # 2. Quantify Bias & Identify Problem Platform
                bias_metrics = adv_batch_corrector.quantify_platform_bias(
                    corrected_df=numeric_corrected_df, manifest_df=manifest_copy
                )
                problem_platform = None
                if bias_metrics:
                    avg_bias = {
                        p: df["cohen_d"].abs().mean()
                        for p, df in bias_metrics.items()
                        if not df.empty
                        and "cohen_d" in df.columns
                        and not df["cohen_d"].isnull().all()
                    }
                    if avg_bias:
                        problem_platform = max(avg_bias, key=avg_bias.get)
                        logger.info(
                            f"Potential problem platform for hierarchical correction: '{problem_platform}' (Avg Abs Cohen's d: {avg_bias.get(problem_platform, np.nan):.3f})"
                        )
                    else:
                        logger.info(
                            "Could not reliably determine problem platform from bias metrics."
                        )
                    # Save individual bias tables
                    for platform, bias_df in bias_metrics.items():
                        save_dataframe(
                            bias_df, f"platform_bias_{platform}.csv", tables_dir, index=True
                        )  # Save with gene index

                # 3. Plot Residual Heatmap (if effects detected)
                if residual_effects_df is not None and not residual_effects_df.empty:
                    try:
                        logger.info("DEBUG: Starting residual effects heatmap plot")
                        # Use the numeric corrected data for plotting
                        plot_df_heatmap = numeric_corrected_df.copy()
                        logger.info(
                            f"DEBUG: plot_df_heatmap shape: {plot_df_heatmap.shape}, index type: {type(plot_df_heatmap.index)}"
                        )

                        # Attempt to map symbols for better readability if possible
                        if (
                            "sym" in expression_data.columns
                            and gene_id_col in expression_data.columns
                        ):
                            id_to_sym_map = pd.Series(
                                expression_data["sym"].values, index=expression_data[gene_id_col]
                            ).to_dict()
                            logger.info(
                                f"DEBUG: id_to_sym_map created with {len(id_to_sym_map)} entries"
                            )

                            # Create new index with proper fallback to original id if not in map
                            new_index = []
                            for idx in plot_df_heatmap.index:
                                new_index.append(id_to_sym_map.get(idx, str(idx)))

                            # Assign the new index directly
                            plot_df_heatmap.index = new_index
                            logger.info(
                                "DEBUG: plot_df_heatmap index mapped to symbols using safe approach"
                            )

                        logger.info(
                            f"DEBUG: residual_effects_df significance_col: 'kruskal_pvalue_adj' in columns: {'kruskal_pvalue_adj' in residual_effects_df.columns}"
                        )
                        if "kruskal_pvalue_adj" in residual_effects_df.columns:
                            logger.info(
                                f"DEBUG: null values in kruskal_pvalue_adj: {residual_effects_df['kruskal_pvalue_adj'].isnull().sum()}"
                            )

                        # Fix and validate the residual effects DataFrame before visualization
                        fixed_residual_df = validate_and_fix_residual_df(residual_effects_df)
                        logger.info(
                            f"DEBUG: Validated residual_effects_df, shape: {fixed_residual_df.shape}, index type: {type(fixed_residual_df.index)}"
                        )

                        logger.info("DEBUG: About to call plot_platform_effect_heatmap")
                        logger.info(
                            f"DEBUG: plot_df_heatmap index sample: {list(plot_df_heatmap.index[:5])}"
                        )
                        logger.info(
                            f"DEBUG: fixed_residual_df index sample: {list(fixed_residual_df.index[:5])}"
                        )

                        fig_resid = viz_runner.plot_platform_effect_heatmap(
                            expression_df=plot_df_heatmap,  # Use symbol-indexed if available
                            manifest_df=manifest_copy,
                            residual_effects_df=fixed_residual_df,
                            top_n_genes=50,
                            significance_col="kruskal_pvalue_adj",
                        )
                        save_visualization(
                            fig_resid, "residual_platform_effects_heatmap.png", figures_dir
                        )
                    except Exception as e:
                        logger.warning(
                            f"Residual effects heatmap failed: {e}", exc_info=True
                        )  # Change to True to get full stack trace

                # 4. Apply Hierarchical Correction (If needed)
                apply_hierarchical = False
                sig_col_k = "kruskal_pvalue_adj"
                # Trigger if significant residual effects found AND problem platform identified AND target gene ID known
                if (
                    residual_effects_df is not None
                    and sig_col_k in residual_effects_df
                    and not residual_effects_df[sig_col_k].isnull().all()
                    and (residual_effects_df[sig_col_k] < advanced_alpha).mean()
                    > 0.05  # More than 5% genes sig. affected
                    and problem_platform is not None
                    and target_gene_id is not None
                ):
                    logger.info(
                        f"Significant residual effects detected. Applying hierarchical correction for '{problem_platform}'..."
                    )
                    apply_hierarchical = True

                if apply_hierarchical:
                    # Run hierarchical correction
                    hierarchical_corrected_numeric = adv_batch_corrector.hierarchical_batch_correction(
                        standard_corrected_df=numeric_corrected_df,  # Input: standard corrected numeric
                        original_expression_df=original_numeric_df,  # Input: original numeric
                        manifest_df=manifest_copy,
                        target_gene_id=target_gene_id,
                        problem_platform=problem_platform,
                    )

                    if (
                        hierarchical_corrected_numeric is not None
                        and not hierarchical_corrected_numeric.empty
                    ):
                        logger.info(
                            ":syringe::syringe: Hierarchical correction applied successfully."
                        )
                        advanced_correction_applied = True

                        # Reconstruct the final DataFrame with metadata
                        final_numeric = (
                            hierarchical_corrected_numeric  # This is genes(ID) x samples
                        )
                        original_metadata = None
                        # Retrieve metadata based on original structure
                        if gene_id_col in expression_data.columns:
                            meta_cols_orig = [
                                c
                                for c in [gene_id_col, sym_col, "ensembl_peptide_id"]
                                if c in expression_data.columns
                            ]
                            original_metadata = (
                                expression_data[meta_cols_orig]
                                .drop_duplicates(subset=[gene_id_col])
                                .set_index(gene_id_col)
                            )
                        elif (
                            sym_col in expression_data.columns
                        ):  # If ID was index but sym was column
                            original_metadata = pd.DataFrame(expression_data[sym_col])

                        # Join metadata back to the corrected numeric data
                        if original_metadata is not None:
                            final_corrected_data = original_metadata.join(
                                final_numeric, how="right"
                            ).reset_index()
                            # Try to reorder columns based on original input
                            original_cols_order = [
                                c
                                for c in expression_data.columns
                                if c in final_corrected_data.columns
                            ]
                            if original_cols_order:
                                final_corrected_data = final_corrected_data[original_cols_order]
                        else:  # No metadata to join back, just use numeric + ID
                            final_corrected_data = final_numeric.reset_index()
                            if final_corrected_data.columns[0] == "index":  # Rename index if needed
                                final_corrected_data = final_corrected_data.rename(
                                    columns={"index": gene_id_col}
                                )

                        # Re-verify target preservation after hierarchical correction
                        _verify_target_preservation(
                            expression_data, final_corrected_data, target_gene
                        )

                        # Plot PCA comparison after hierarchical correction
                        try:
                            if numeric_corrected_df is not None:  # Standard corrected numeric
                                # Prepare DFs for plotting (Samples x Genes)
                                common_samples_pca = numeric_corrected_df.columns.intersection(
                                    final_numeric.columns
                                )
                                # Use original numeric data (already Samples x Genes if prepped correctly)
                                expr_raw_pca = (
                                    original_numeric_df[common_samples_pca].T
                                    if original_numeric_df is not None
                                    else None
                                )
                                expr_std_corr_pca = numeric_corrected_df[common_samples_pca].T
                                expr_hier_corr_pca = final_numeric[
                                    common_samples_pca
                                ].T  # Use final corrected numeric

                                if expr_raw_pca is not None:
                                    fig_bc_adv = batch_corrector.compare_pca_stages(
                                        raw_df=expr_raw_pca,
                                        filtered_df=expr_std_corr_pca,
                                        corrected_df=expr_hier_corr_pca,
                                        manifest_df=manifest_copy,
                                    )
                                    save_visualization(
                                        fig_bc_adv,
                                        "batch_correction_hierarchical_pca.png",
                                        figures_dir,
                                    )
                                else:
                                    logger.warning(
                                        "Cannot plot hierarchical PCA: Original numeric data missing."
                                    )
                            else:
                                logger.warning(
                                    "Cannot plot hierarchical PCA: Standard corrected numeric data missing."
                                )
                        except Exception as e:
                            logger.warning(f"Hierarchical PCA plot failed: {e}")
                    else:
                        logger.error(
                            "Hierarchical correction failed. Using standard corrected data (if available)."
                        )
                        # Keep final_corrected_data as it was after standard correction
                elif run_advanced_correction:  # Log if advanced was enabled but criteria not met
                    logger.info(
                        "Advanced correction not triggered: Criteria (residual effects / problem platform / target ID) not met."
                    )

            except Exception as e:
                logger.exception(f"Error during advanced BC checks/application: {e}")
                if not initial_correction_applied:
                    final_corrected_data = expression_data.copy()

        # Log if advanced checks were skipped entirely
        elif run_advanced_correction and not (
            initial_correction_applied or standard_correction_success
        ):
            logger.info(
                "Skipping advanced correction checks: Standard correction was not applied or failed."
            )

        # --- Final Validation Step ---
        if (
            initial_correction_applied or advanced_correction_applied
        ):  # Only validate if *any* correction occurred
            logger.info("--- Validating Final Corrected Data vs Original ---")
            if original_numeric_df is not None:
                numeric_final_corrected_df = None  # Holds final corrected numeric data
                try:
                    # Prepare final corrected numeric data (genes x samples)
                    if gene_id_col in final_corrected_data.columns:
                        meta_cols_final = [
                            c
                            for c in [gene_id_col, sym_col, "ensembl_peptide_id"]
                            if c in final_corrected_data.columns
                        ]
                        numeric_cols_final = [
                            c for c in final_corrected_data.columns if c not in meta_cols_final
                        ]
                        df_unique_final = final_corrected_data.drop_duplicates(
                            subset=[gene_id_col], keep="first"
                        )
                        numeric_final_corrected_df = df_unique_final.set_index(gene_id_col)[
                            numeric_cols_final
                        ].apply(pd.to_numeric, errors="coerce")
                    elif final_corrected_data.index.name or not isinstance(
                        final_corrected_data.index, pd.RangeIndex
                    ):
                        numeric_final_corrected_df = final_corrected_data.select_dtypes(
                            include=np.number
                        ).apply(pd.to_numeric, errors="coerce")

                    if (
                        numeric_final_corrected_df is not None
                        and not numeric_final_corrected_df.empty
                    ):
                        # Run validation
                        validation_metrics = adv_batch_corrector.validate_correction(
                            original_df=original_numeric_df,
                            corrected_df=numeric_final_corrected_df,
                            manifest_df=manifest_copy,
                        )
                        logger.info(
                            f"Validation Metrics (Final Corrected vs Original): {validation_metrics}"
                        )
                        # Save validation metrics
                        try:
                            val_path = (
                                tables_dir / "correction_validation_metrics.json"
                            )  # Use corrected tables_dir path
                            _ensure_dir_exists(val_path.parent)
                            serializable_metrics = {
                                k: (
                                    float(v)
                                    if pd.notna(v) and isinstance(v, (int, float, np.number))
                                    else None
                                )
                                for k, v in validation_metrics.items()
                            }
                            # Filter out None values before saving
                            serializable_metrics = {
                                k: v for k, v in serializable_metrics.items() if v is not None
                            }

                            with open(val_path, "w", encoding="utf-8") as f:
                                json.dump(serializable_metrics, f, indent=4)
                            logger.info(f"Saved validation metrics to {val_path}")
                        except NameError:  # Should not happen now
                            logger.exception(
                                "Failed to save validation metrics: 'tables_dir' not defined in scope."
                            )
                        except Exception as e:
                            logger.exception(f"Failed to save validation metrics: {e}")
                    else:
                        logger.warning(
                            "Could not prepare final corrected numeric data for validation."
                        )
                except Exception as e:
                    logger.exception(
                        f"Error preparing final corrected numeric data for validation: {e}"
                    )
            else:
                logger.warning("Could not prepare original numeric data for validation.")
        else:
            logger.info("Skipping final validation as no correction was applied.")

        # Return the final state of the data
        return final_corrected_data

    except Exception as e:
        logger.exception(f"Batch correction process encountered a critical error: {e}")
        logger.warning("Returning original uncorrected data due to failure.")
        return expression_data.copy()  # Return original as fallback


@analysis_step("3. Correlation Analysis")
def run_correlation(
    expression_df: pd.DataFrame,
    manifest_data: pd.DataFrame | None,
    methods: list[str],
    top_n: int,
    target_gene: str,
    tables_dir: Path,
    run_bootstrap: bool,
    bootstrap_iterations: int,
    confidence_level: float,
) -> tuple[pd.DataFrame | None, anndata.AnnData | None]:
    """Runs correlation analysis, including optional bootstrapping, and saves results."""
    correlation_df, adata = None, None
    try:
        analyzer = GeneCorrelationAnalysis()

        # Prepare data - creates AnnData object
        logger.info(f"Preparing data for correlation analysis (Target: '{target_gene}')...")
        adata_prepared, _ = analyzer.prepare_data(
            expression_df=expression_df, manifest_df=manifest_data, target_col=target_gene
        )
        if adata_prepared is None:
            logger.error("Data preparation failed (AnnData is None), cannot proceed.")
            return None, None
        adata = adata_prepared

        # Define Rich Progress columns for correlation steps
        progress_cols = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ]

        # --- Calculate Correlations with Progress Bar ---
        logger.info(f"Calculating correlations using methods: {methods}...")
        _, correlation_df = analyzer.calculate_correlations(
            adata=adata,
            methods=methods,
            target_col=target_gene,
            run_bootstrap=run_bootstrap,
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
        )  # Bootstrap happens inside if enabled

        if correlation_df is None or correlation_df.empty:
            logger.error("Correlation analysis yielded no results.")
            return None, adata

        logger.info(
            f"Correlation analysis complete. Found {len(correlation_df)} gene-method results."
        )
        save_dataframe(correlation_df, "correlation_results_full.csv", tables_dir, index=False)

        # --- Log Top Genes Summary using Rich Table ---
        console.rule(f"[bold green]Top Correlated Genes Summary (vs {target_gene})[/]")
        processed_methods_logged = set()
        for method_name_config in methods:
            method_cap = method_name_config.capitalize()
            if method_cap in processed_methods_logged:
                continue

            method_subset = correlation_df[correlation_df["Correlation_Type"] == method_cap]
            if method_subset.empty:
                console.print(f"\n[yellow]:warning: No results found for method: {method_cap}[/]")
                processed_methods_logged.add(method_cap)
                continue

            sort_key = next(
                (
                    key
                    for key in [
                        "Correlation_Rank_Score",
                        f"Rank_Stability_{method_cap}_Perc",
                        "Correlation_Coefficient_Abs",
                    ]
                    if key in method_subset.columns and not method_subset[key].isnull().all()
                ),
                "Correlation_Coefficient_Abs",
            )
            if sort_key not in method_subset.columns:
                logger.warning(
                    f"Cannot find suitable sort key for '{method_cap}'. Skipping summary table."
                )
                processed_methods_logged.add(method_cap)
                continue

            top_genes = method_subset.sort_values(
                sort_key, ascending=False, na_position="last"
            ).head(min(10, top_n))

            # Create Rich Table
            table = Table(
                title=f"Top {len(top_genes)} Genes by {method_cap} (Sorted by {sort_key})",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Rank", style="dim", width=4)
            table.add_column("Symbol", style="cyan", no_wrap=True)
            table.add_column("Corr Coeff", justify="right")
            table.add_column("P-Value (Adj)", justify="right")
            if f"Rank_Stability_{method_cap}_Perc" in top_genes.columns:
                table.add_column("Stability (%)", justify="right")

            for i, row in enumerate(top_genes.itertuples(index=False), 1):
                symbol = getattr(row, "sym", getattr(row, "ensembl_transcript_id", "N/A"))
                coeff = getattr(row, "Correlation_Coefficient", np.nan)
                pval_adj = getattr(row, "FDR_Adjusted_P_Value", np.nan)
                pval_raw = getattr(row, "Correlation_P_Value", np.nan)
                pval = pval_adj if pd.notna(pval_adj) else pval_raw
                stability = getattr(row, f"Rank_Stability_{method_cap}_Perc", None)

                coeff_str = f"{coeff:+.3f}" if pd.notna(coeff) else "[dim]N/A[/]"
                pval_str = f"{pval:.2e}" if pd.notna(pval) else "[dim]N/A[/]"
                row_data = [str(i), symbol, coeff_str, pval_str]

                if f"Rank_Stability_{method_cap}_Perc" in top_genes.columns:
                    stab_str = f"{stability:.1f}" if pd.notna(stability) else "[dim]N/A[/]"
                    row_data.append(stab_str)

                table.add_row(*row_data)

            console.print(table)
            processed_methods_logged.add(method_cap)

        return correlation_df, adata

    except Exception as e:
        logger.exception(f"Correlation analysis step failed critically: {e}")
        adata_return = adata if "adata" in locals() else None
        return None, adata_return


# --- _verify_target_preservation (Keep as is) ---
def _verify_target_preservation(
    original_df: pd.DataFrame, corrected_df: pd.DataFrame, target_gene: str
) -> None:
    # (Keep internal logic as before)
    if "sym" not in original_df.columns or "sym" not in corrected_df.columns:
        logger.warning("Cannot verify target gene preservation ('sym' column missing).")
        return
    original_tg_row = original_df[original_df["sym"] == target_gene]
    corrected_tg_row = corrected_df[corrected_df["sym"] == target_gene]
    if original_tg_row.empty:
        logger.warning(f"Target gene '{target_gene}' missing in original data for check.")
        return
    if corrected_tg_row.empty:
        logger.warning(f"Target gene '{target_gene}' missing in corrected data for check.")
        return
    logger.info(f"Verifying preservation of target gene '{target_gene}'...")
    try:
        meta_cols = ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
        sample_cols_orig = [
            c
            for c in original_df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(original_df[c])
        ]
        sample_cols_corr = [
            c
            for c in corrected_df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(corrected_df[c])
        ]
        common_samples = sorted(set(sample_cols_orig) & set(sample_cols_corr))
        if not common_samples:
            logger.warning("No common numeric sample columns found for target preservation check.")
            return
        original_vals = pd.to_numeric(
            original_tg_row.iloc[0][common_samples], errors="coerce"
        ).values
        corrected_vals = pd.to_numeric(
            corrected_tg_row.iloc[0][common_samples], errors="coerce"
        ).values
        valid_mask = ~np.isnan(original_vals) & ~np.isnan(corrected_vals)
        original_vals_valid = original_vals[valid_mask]
        corrected_vals_valid = corrected_vals[valid_mask]
        if len(original_vals_valid) <= 1:
            logger.warning("Insufficient valid data points for target gene preservation check.")
            return
        corr, p_val = np.nan, np.nan
        if np.std(original_vals_valid) > 1e-9 and np.std(corrected_vals_valid) > 1e-9:
            try:
                corr, p_val = pearsonr(original_vals_valid, corrected_vals_valid)
            except ValueError as pe:
                logger.warning(f"Could not calculate Pearson correlation for target gene: {pe}")
            if pd.notna(corr):
                logger.info(f"Preservation check: Pearson correlation = {corr:.4f} (p={p_val:.2e})")
        else:
            logger.warning(
                "Cannot calculate target gene Pearson correlation (zero variance in valid data)."
            )
        if np.allclose(original_vals_valid, corrected_vals_valid, rtol=1e-5, atol=1e-8):
            logger.info(
                f":heavy_check_mark: Target gene '{target_gene}' values appear numerically preserved."
            )
        else:
            diff = np.abs(original_vals_valid - corrected_vals_valid)
            logger.warning(
                f":warning: Target gene '{target_gene}' values differ numerically (Corr: {corr:.4f})."
            )
            if len(diff) > 0:
                logger.info(
                    f"  Mean Abs Diff: {np.nanmean(diff):.6g}, Max Abs Diff: {np.nanmax(diff):.6g}"
                )
            else:
                logger.info("  No valid pairs to calculate difference metrics.")
    except Exception as e:
        logger.warning(f"Error during target gene preservation check: {e}", exc_info=False)


# --- run_panel_optimization ---
@analysis_step("Optional: Marker Panel Optimization")
def run_panel_optimization(
    ranked_markers_df: pd.DataFrame | None,
    expression_df: pd.DataFrame,
    adata: anndata.AnnData | None,
    target_gene: str,
    panel_sizes: list[int],
    panel_types: list[str],
    panel_candidates: int,
    tables_dir: Path,
    figures_dir: Path,
    viz_runner: CorrelationVisualization,
) -> dict[str, dict[str, Any]] | None:
    if ranked_markers_df is None or ranked_markers_df.empty:
        logger.warning("Skipping panel optimization: Ranked marker data unavailable.")
        return None
    if "gene_symbol" not in ranked_markers_df.columns:
        logger.warning("Skipping panel optimization: Ranked marker data missing 'gene_symbol'.")
        return None
    if adata is None:
        logger.warning("Skipping panel optimization: AnnData object unavailable.")
        return None
    if target_gene not in adata.obs.columns:
        logger.warning(
            f"Skipping panel optimization: Target gene '{target_gene}' not in AnnData observations."
        )
        return None

    panel_results = None
    try:
        optimizer = MarkerPanelOptimization()
        logger.info("Preparing data for panel optimization...")

        id_col = "ensembl_transcript_id"
        sym_col = "sym"
        id_to_sym_map = {}
        if id_col in expression_df.columns and sym_col in expression_df.columns:
            try:
                map_df = (
                    expression_df[[id_col, sym_col]]
                    .dropna()
                    .drop_duplicates(subset=[id_col])
                    .set_index(id_col)
                )
                id_to_sym_map = map_df[sym_col].to_dict()
            except Exception as e:
                logger.warning(f"Error creating ID->Symbol map from original data: {e}")
        if (
            not id_to_sym_map
            and id_col in ranked_markers_df.columns
            and "gene_symbol" in ranked_markers_df.columns
        ):
            try:
                map_df_ranked = (
                    ranked_markers_df[[id_col, "gene_symbol"]]
                    .dropna()
                    .drop_duplicates(subset=[id_col])
                    .set_index(id_col)
                )
                id_to_sym_map = map_df_ranked["gene_symbol"].to_dict()
            except Exception as e:
                logger.warning(f"Error creating ID->Symbol map from ranked data: {e}")
        if not id_to_sym_map:
            logger.error("Failed to create ID->Symbol map. Cannot proceed with panel optimization.")
            return None

        expr_symbols_as_cols = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )
        expr_symbols_as_cols.columns = [
            id_to_sym_map.get(gid, str(gid)) for gid in expr_symbols_as_cols.columns
        ]
        expr_symbols_as_cols = expr_symbols_as_cols.loc[
            :, ~expr_symbols_as_cols.columns.duplicated(keep="first")
        ]

        candidate_symbols = ranked_markers_df["gene_symbol"].unique().tolist()
        valid_expr_symbols = [
            sym for sym in candidate_symbols if sym in expr_symbols_as_cols.columns
        ]
        if not valid_expr_symbols:
            logger.error(
                "None of the candidate gene symbols found in the prepared expression data columns."
            )
            return None
        expr_for_panels = expr_symbols_as_cols[valid_expr_symbols]
        logger.info(f"Prepared expression data for panel optimization: {expr_for_panels.shape}")

        target_series = adata.obs[target_gene]

        logger.info(
            f"Running panel optimization with {expr_for_panels.shape[1]} candidate genes..."
        )
        panel_results = optimizer.design_marker_panels(
            ranked_genes_df=ranked_markers_df,
            expression_df=expr_for_panels,
            target_series=target_series,
            panel_sizes=panel_sizes,
            panel_types=panel_types,
            top_n_candidates=panel_candidates,
        )

        if panel_results:
            all_panel_data = []
            for p_type, panels in panel_results.items():
                for panel_name, metrics in panels.items():
                    if metrics:
                        metrics_copy = metrics.copy()
                        metrics_copy["panel_name"] = panel_name
                        metrics_copy["panel_type"] = p_type
                        genes_list = metrics_copy.pop("genes", [])
                        metrics_copy["genes_str"] = (
                            ", ".join(map(str, genes_list)) if genes_list else ""
                        )
                        all_panel_data.append(metrics_copy)
            if all_panel_data:
                panel_df = pd.DataFrame(all_panel_data)
                save_dataframe(panel_df, "marker_panel_evaluation.csv", tables_dir, index=False)
                try:
                    if hasattr(viz_runner, "plot_panel_comparison"):
                        fig_comp = viz_runner.plot_panel_comparison(panel_results)
                        save_visualization(fig_comp, "panel_comparison.png", figures_dir)
                    if hasattr(viz_runner, "plot_information_gain"):
                        fig_gain = viz_runner.plot_information_gain(panel_results)
                        save_visualization(fig_gain, "panel_information_gain.png", figures_dir)
                except Exception as viz_err:
                    logger.exception(f"Failed to generate panel visualizations: {viz_err}")
            else:
                logger.info("Panel optimization ran but produced no valid panels.")
        else:
            logger.info("Panel optimization did not return any results.")
    except Exception as e:
        logger.exception(f"Marker panel optimization step failed critically: {e}")
        return None
    return panel_results


# --- run_ranking (Modified for Rich Table) ---
@analysis_step("4. Consensus Gene Ranking")
def run_ranking(
    correlation_df: pd.DataFrame | None,
    expression_df: pd.DataFrame,
    methods: list[str],
    top_n: int,
    tables_dir: Path,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Ranks genes based on aggregated correlation, stability, and expression stats."""
    if correlation_df is None or correlation_df.empty:
        logger.warning("Skipping ranking: Correlation results unavailable.")
        return None, []

    ranked_markers_df, ranked_markers_list = None, []
    try:
        ranker = GeneRanking()
        capitalized_methods = [m.capitalize() for m in methods]
        ranking_results = ranker.run_ranking_analysis(
            correlation_df=correlation_df,
            expression_df=expression_df,
            methods=capitalized_methods,
            top_n=top_n,
        )

        ranked_markers_df = ranking_results.get("rank_statistics")
        ranked_markers_list = ranking_results.get("ranked_markers", [])  # List of symbols

        if ranked_markers_df is not None and not ranked_markers_df.empty:
            save_dataframe(ranked_markers_df, "ranked_markers.csv", tables_dir, index=False)

            top_to_show_df = ranked_markers_df.head(min(10, top_n))
            if not top_to_show_df.empty:
                console.rule("[bold green]Top Consensus Marker Genes[/]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank", style="dim", width=4)
                table.add_column("Symbol", style="cyan", no_wrap=True)
                table.add_column("Final Score", justify="right")
                table.add_column("Corr", justify="right", style="blue")
                table.add_column("Level", justify="right", style="green")
                table.add_column("Expr Stab", justify="right", style="yellow")
                table.add_column("Rank Stab", justify="right", style="magenta")

                for i, row in enumerate(top_to_show_df.itertuples(), 1):
                    table.add_row(
                        str(i),
                        getattr(row, "gene_symbol", "N/A"),
                        f"{getattr(row, 'final_score', np.nan):.3f}",
                        f"{getattr(row, 'correlation_score', np.nan):.2f}",
                        f"{getattr(row, 'expression_level_score', np.nan):.2f}",
                        f"{getattr(row, 'expression_stability_score', np.nan):.2f}",
                        f"{getattr(row, 'rank_stability_score', np.nan):.2f}",
                    )
                console.print(table)
            else:
                logger.info("No consensus markers identified after ranking.")
        else:
            logger.info("Gene ranking analysis produced no results.")
            ranked_markers_df = None
            ranked_markers_list = []

        return ranked_markers_df, ranked_markers_list

    except Exception as e:
        logger.exception(f"Gene ranking analysis failed: {e}")
        return None, []


# --- generate_visualizations_task1 ---
@analysis_step("5. Generating Task 1 Visualizations")
def generate_visualizations_task1(
    correlation_df: pd.DataFrame | None,
    adata: anndata.AnnData | None,
    ranked_markers_df: pd.DataFrame | None,
    methods: list[str],  # Methods used for correlation
    top_n: int,  # Used for deciding how many items in some plots
    target_gene: str,
    figures_dir: Path,
    viz_runner: CorrelationVisualization,
) -> None:
    if correlation_df is None or correlation_df.empty:
        logger.warning("Skipping Task 1 visualizations: Correlation results unavailable.")
        return

    default_method_from_config = get_correlation_config().get("default_method", "spearman")
    plot_method = methods[0] if methods else default_method_from_config
    plot_method_cap = plot_method.capitalize()
    logger.info(f"Using method '{plot_method_cap}' for relevant Task 1 visualizations.")

    viz_success_count = 0

    def _try_plot(
        plot_func: Callable[..., Figure | None], filename: str, *args: Any, **kwargs: Any
    ) -> None:
        nonlocal viz_success_count
        if not callable(plot_func):
            logger.error(f"Plot function for '{filename}' not callable.")
            return
        plot_func_name = getattr(plot_func, "__name__", "unnamed_plot_func")
        logger.info(f":bar_chart: Generating '{filename}' using {plot_func_name}...")
        try:
            fig = plot_func(*args, **kwargs)
            if isinstance(fig, Figure):
                save_visualization(fig, filename, figures_dir)
                viz_success_count += 1
            elif fig is None:
                logger.warning(f"Plot function {plot_func_name} for '{filename}' returned None.")
            else:
                logger.error(
                    f"Plot function {plot_func_name} for '{filename}' returned unexpected type: {type(fig)}"
                )
        except ImportError as import_err:
            logger.warning(f"Skipping plot '{filename}': Missing dependency '{import_err.name}'.")
        except Exception as e:
            logger.exception(f"Failed to generate plot '{filename}': {e}")

    # Correlation Plots
    _try_plot(
        viz_runner.plot_correlation_heatmap,
        f"correlation_heatmap_{plot_method}.png",
        correlation_df=correlation_df,
        top_n=min(25, top_n * 2),
        method=plot_method_cap,
        cluster=True,
    )
    if adata is not None:
        _try_plot(
            viz_runner.plot_scatter_matrix,
            f"gene_scatter_matrix_{plot_method}.png",
            adata=adata,
            correlation_df=correlation_df,
            top_n=min(5, top_n),
            method=plot_method_cap,
            target_col=target_gene,
        )
        if "platform" in adata.obs.columns:
            _try_plot(
                viz_runner.plot_expression_by_platform,
                f"expression_by_platform_{plot_method}.png",
                adata=adata,
                correlation_df=correlation_df,
                top_n=min(6, top_n),
                method=plot_method_cap,
                target_col=target_gene,
                platform_col="platform",
            )
        else:
            logger.info("Skipping expression-by-platform plot: 'platform' info missing.")
    else:
        logger.warning("Skipping AnnData-dependent plots: AnnData object unavailable.")
    _try_plot(
        viz_runner.plot_correlation_network,
        f"correlation_network_{plot_method}.png",
        correlation_df=correlation_df,
        top_n=min(20, top_n * 2),
        method=plot_method_cap,
        min_edge_weight_abs=0.4,
    )

    # Bootstrap Plots
    ci_lower_col = f"Correlation_CI_Lower_{plot_method_cap}"
    stability_col = f"Rank_Stability_{plot_method_cap}_Perc"
    if ci_lower_col in correlation_df.columns:
        _try_plot(
            viz_runner.plot_bootstrap_confidence,
            f"bootstrap_confidence_{plot_method}.png",
            correlation_df=correlation_df,
            top_n=min(20, top_n * 2),
            method=plot_method_cap,
        )
    else:
        logger.info(f"Skipping bootstrap confidence plot: Column '{ci_lower_col}' not found.")
    if stability_col in correlation_df.columns:
        _try_plot(
            viz_runner.plot_rank_stability_heatmap,
            f"rank_stability_heatmap_{plot_method}.png",
            correlation_df=correlation_df,
            top_n_genes=min(30, top_n * 2),
            top_n_ranks=top_n,
            method=plot_method_cap,
        )
    else:
        logger.info(f"Skipping rank stability plot: Column '{stability_col}' not found.")

    # Ranking Plot
    if ranked_markers_df is not None and not ranked_markers_df.empty:
        ranker_viz_instance = GeneRanking()
        req_cols_ranking = [
            "final_score",
            "gene_symbol",
            "correlation_score",
            "expression_level_score",
            "expression_stability_score",
            "rank_stability_score",
        ]
        if all(col in ranked_markers_df.columns for col in req_cols_ranking):
            _try_plot(
                ranker_viz_instance.visualize_ranking,
                "gene_ranking_scores.png",
                ranked_genes=ranked_markers_df,
                figsize=(14, 8),
            )
        else:
            logger.error(
                f"Cannot generate ranking plot: Missing columns: {[c for c in req_cols_ranking if c not in ranked_markers_df.columns]}"
            )
    else:
        logger.info("Skipping ranking plot: No ranking data.")

    logger.info(f"Task 1 Visualization finished. {viz_success_count} figures potentially saved.")


# --- run_binary_markers ---
@analysis_step("6. Binary Marker Identification")
def run_binary_markers(
    expression_df: pd.DataFrame, target_gene: str, tables_dir: Path
) -> pd.DataFrame | None:
    """Identifies binary markers and finds optimal thresholds."""
    console.print("[yellow]Binary Marker Identification can be computationally intensive.[/]")
    optimal_thresholds_df = None
    try:
        ranker = GeneRanking()
        logger.info("Identifying potential binary markers using AUC...")
        binary_markers_df = ranker.binary_marker_identification(
            expression_df=expression_df,
            target_col=target_gene,
            top_n=50,
            min_auc=0.7,
            prefilter_genes=5000,
        )

        if binary_markers_df is None or binary_markers_df.empty:
            logger.info("No binary markers identified meeting criteria.")
            return None

        save_dataframe(binary_markers_df, "binary_markers_auc.csv", tables_dir, index=False)

        # Log top markers using Rich Table
        top_binary_to_show = binary_markers_df.head(min(5, len(binary_markers_df)))
        if not top_binary_to_show.empty:
            console.rule("[bold green]Top Potential Binary Markers (by CV AUC)[/]")
            bin_table = Table(show_header=True, header_style="bold magenta")
            bin_table.add_column("Rank", style="dim", width=4)
            bin_table.add_column("Symbol", style="cyan")
            bin_table.add_column("CV AUC", justify="right")
            bin_table.add_column("AUC", justify="right")
            bin_table.add_column("Direction", justify="center")
            for i, row in enumerate(top_binary_to_show.itertuples(index=False), 1):
                bin_table.add_row(
                    str(i),
                    getattr(row, "gene_symbol", "N/A"),
                    f"{getattr(row, 'cv_auc', np.nan):.4f}",
                    f"{getattr(row, 'auc', np.nan):.4f}",
                    getattr(row, "direction", "N/A"),
                )
            console.print(bin_table)

        logger.info("Finding optimal thresholds for identified binary markers...")
        optimal_thresholds_df = ranker.find_optimal_thresholds(
            expression_df=expression_df,
            target_col=target_gene,
            marker_genes=binary_markers_df,
            num_thresholds=30,
        )

        if optimal_thresholds_df is not None and not optimal_thresholds_df.empty:
            save_dataframe(optimal_thresholds_df, "optimal_thresholds.csv", tables_dir, index=False)
            # Log top thresholds using Rich Table
            top_thresh_to_show = optimal_thresholds_df.head(min(5, len(optimal_thresholds_df)))
            console.rule("[bold green]Top Markers with Optimal Thresholds (by Balanced Acc)[/]")
            thresh_table = Table(show_header=True, header_style="bold magenta")
            thresh_table.add_column("Rank", style="dim", width=4)
            thresh_table.add_column("Symbol", style="cyan")
            thresh_table.add_column("Threshold", justify="right")
            thresh_table.add_column("Balanced Acc", justify="right")
            thresh_table.add_column("Sensitivity", justify="right")
            thresh_table.add_column("Specificity", justify="right")
            for i, row in enumerate(top_thresh_to_show.itertuples(index=False), 1):
                thresh_table.add_row(
                    str(i),
                    getattr(row, "gene_symbol", "N/A"),
                    f"{getattr(row, 'optimal_threshold', np.nan):.4g}",
                    f"{getattr(row, 'balanced_accuracy', np.nan):.4f}",
                    f"{getattr(row, 'sensitivity', np.nan):.3f}",
                    f"{getattr(row, 'specificity', np.nan):.3f}",
                )
            console.print(thresh_table)
        else:
            logger.info("Could not determine optimal thresholds.")
            optimal_thresholds_df = None

    except Exception as e:
        logger.exception(f"Error during binary marker identification: {e}")
        optimal_thresholds_df = None

    return optimal_thresholds_df


# --- Run Platform Consensus Analysis ---
@analysis_step("7. Platform Consensus Analysis")
def run_platform_consensus(
    expression_df: pd.DataFrame,
    manifest_data: pd.DataFrame | None,
    correlation_df: pd.DataFrame | None,
    target_gene: str,
    tables_dir: Path,
    figures_dir: Path,
    viz_runner: CorrelationVisualization,
) -> pd.DataFrame | None:
    """Analyzes marker consistency across platforms and visualizes results."""
    if manifest_data is None:
        logger.warning("Skipping platform consensus: Manifest data missing.")
        return None
    platform_col = "platform"
    manifest_copy = manifest_data.copy()
    if platform_col not in manifest_copy.columns:
        if "description" in manifest_copy.columns:
            logger.info("Deriving platform from 'description' for consensus.")
            manifest_copy[platform_col] = (
                manifest_copy["description"]
                .str.extract(r"(HiSeq|NovaSeq|NextSeq)", expand=False)
                .fillna("Unknown")
            )
        else:
            logger.warning(
                f"Skipping platform consensus: Missing '{platform_col}' or 'description'."
            )
            return None
    if manifest_copy[platform_col].nunique(dropna=True) <= 1:
        logger.info("Skipping platform consensus: Only one platform.")
        return None

    consensus_df = None
    try:
        analyzer = GeneCorrelationAnalysis()
        logger.info("Analyzing marker consistency across platforms...")
        consensus_df = analyzer.analyze_platform_consensus(
            expression_df=expression_df,
            manifest_df=manifest_copy,
            target_col=target_gene,
            top_n=100,
            min_correlation=0.5,
            max_p_value=0.05,
        )

        if consensus_df is None or consensus_df.empty:
            logger.info("No consistent cross-platform markers identified.")
            return None

        save_dataframe(consensus_df, "platform_consensus_markers.csv", tables_dir, index=False)

        # Log top consensus genes using Rich Table
        top_consensus_to_show = consensus_df.head(min(10, len(consensus_df)))
        if not top_consensus_to_show.empty:
            console.rule("[bold green]Top Cross-Platform Consensus Genes[/]")
            cons_table = Table(show_header=True, header_style="bold magenta")
            cons_table.add_column("Rank", style="dim", width=4)
            cons_table.add_column("Symbol", style="cyan")
            cons_table.add_column("Score", justify="right")
            cons_table.add_column("# Platforms", justify="center")
            cons_table.add_column("Direction", justify="center")
            cons_table.add_column("Avg Abs Corr", justify="right")
            for i, row in enumerate(top_consensus_to_show.itertuples(index=False), 1):
                direction = (
                    ":heavy_check_mark:"
                    if getattr(row, "same_direction", False)
                    else "[yellow]:warning:[/]"
                )
                cons_table.add_row(
                    str(i),
                    getattr(row, "symbol", "N/A"),
                    f"{getattr(row, 'consensus_score', np.nan):.3f}",
                    str(getattr(row, "platforms_count", "N/A")),
                    direction,
                    f"{getattr(row, 'avg_abs_correlation', np.nan):.3f}",
                )
            console.print(cons_table)

        # Visualize consensus ranking
        try:
            if hasattr(analyzer, "visualize_consensus_ranking"):
                consensus_fig = analyzer.visualize_consensus_ranking(consensus_df, top_n=15)
                save_visualization(consensus_fig, "platform_consensus_ranking.png", figures_dir)
        except Exception as e:
            logger.exception(f"Failed consensus visualization: {e}", exc_info=False)

        # Analyze method overlap
        if correlation_df is not None and not correlation_df.empty:
            logger.info("Analyzing overlap between methods and platform consensus...")
            try:
                overlap_results = analyzer.analyze_method_consensus_overlap(
                    correlation_df, consensus_df, top_n=100
                )
                if overlap_results and "overlap_stats" in overlap_results:
                    overlap_table = Table(
                        title="Method Overlap with Platform Consensus (Top 100)",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    overlap_table.add_column("Method", style="cyan")
                    overlap_table.add_column("Overlap (%)", justify="right")
                    overlap_table.add_column("Jaccard", justify="right")
                    for method, stats in overlap_results["overlap_stats"].items():
                        overlap_table.add_row(
                            method,
                            f"{stats.get('overlap_percentage', np.nan):.1f}",
                            f"{stats.get('jaccard_similarity', np.nan):.3f}",
                        )
                    console.print(overlap_table)
                    # Visualize overlap
                    try:
                        barplot_fig, venn_fig = analyzer.visualize_method_overlap(overlap_results)
                        save_visualization(
                            barplot_fig, "method_consensus_overlap_barplot.png", figures_dir
                        )
                        if venn_fig:
                            save_visualization(
                                venn_fig, "method_consensus_overlap_venn.png", figures_dir
                            )
                    except ImportError:
                        logger.warning("Skipping Venn diagram: matplotlib_venn not installed.")
                    except Exception as e:
                        logger.exception(
                            f"Error creating overlap visualization: {e}", exc_info=False
                        )
                else:
                    logger.info("Could not compute overlap stats.")
            except Exception as e:
                logger.exception(f"Error analyzing overlap: {e}", exc_info=False)
        else:
            logger.warning(
                "Skipping method/consensus overlap analysis: Correlation results missing."
            )

    except Exception as e:
        logger.exception(f"Error during platform consensus analysis: {e}")
        consensus_df = None  # Ensure None on error

    return consensus_df


# --- save_processed_data ---
@analysis_step("8. Saving Processed Data (Task 1 Output)")
def save_processed_data(processed_data: pd.DataFrame | None, output_dir: Path) -> None:
    if processed_data is None or processed_data.empty:
        logger.warning("Skipping save: Processed data from Task 1 is missing.")
        return
    # Ensure output directory exists
    if not _ensure_dir_exists(output_dir):
        logger.error(f"Cannot save processed data: Output directory inaccessible {output_dir}")
        return
    processed_data_path = output_dir / "expressions_processed.csv"
    try:
        processed_data.to_csv(processed_data_path, index=False)
        logger.info(
            f":floppy_disk: [green]Saved processed expression data (Task 1 output):[/] {processed_data_path.resolve()}"
        )
    except Exception as e:
        logger.exception(f":cross_mark: [bold red]Failed to save processed data:[/bold red] {e}")


# ===================================================
#  === Task 1 Orchestrator ===
# ===================================================
def run_task1(args: argparse.Namespace) -> bool:
    """Orchestrates the execution of Task 1 with Rich logging."""
    print_header("TASK 1: CORRELATION ANALYSIS FOR MARKER GENE IDENTIFICATION")
    task_start_time = time.time()
    overall_success = True

    # --- Config Loading ---
    try:
        corr_config = get_correlation_config()
        batch_config = get_batch_correction_config()
        target_gene = corr_config.get("target_gene", "PRODUCT-TG")
        methods_param = args.methods or corr_config.get("methods", ["spearman", "pearson"])
        methods = (
            [m.strip() for m in methods_param.split(",")]
            if isinstance(methods_param, str)
            else methods_param
        )
        top_n = args.top_n if args.top_n is not None else corr_config.get("top_n", 50)
        run_bootstrap = corr_config.get("run_bootstrap", True)
        bootstrap_iterations = corr_config.get("bootstrap_iterations", 1000)
        confidence_level = corr_config.get("confidence_level", 0.95)
        run_panels = corr_config.get("run_panel_optimization", True)
        panel_sizes = corr_config.get("panel_sizes", [3, 5, 10])
        panel_types = corr_config.get("panel_types", ["minimal_redundancy", "max_score"])
        panel_candidates = corr_config.get("panel_candidates", 100)
        skip_batch_correction = args.skip_batch_correction
        run_advanced_correction = batch_config.get("run_advanced", False)
        advanced_alpha = batch_config.get("advanced_alpha", 0.01)
        task1_figures_dir = get_path("figures_dir") / "task1"
        task1_tables_dir = get_path("tables_dir") / "task1"
        processed_data_dir = get_path("preprocessed_dir")
    except KeyError as e:
        logger.critical(f"Config key error: {e}. Check config/env vars.")
        return False
    except Exception as e:
        logger.critical(f"Critical configuration error Task 1: {e}")
        return False

    # --- Log Parameters using Rich ---
    param_table = Table(title="Effective Task 1 Parameters", show_header=False, box=None)
    param_table.add_column("Parameter", style="dim")
    param_table.add_column("Value")
    param_table.add_row("Target Gene", f"'{target_gene}'")
    param_table.add_row("Correlation Methods", f"{methods}")
    param_table.add_row("Top N Genes", f"{top_n}")
    param_table.add_row(
        "Skip Batch Correction",
        f"[red]{skip_batch_correction}[/]" if skip_batch_correction else "[green]False[/]",
    )
    param_table.add_row(
        "Run Bootstrap", f"{run_bootstrap} (Iters: {bootstrap_iterations}, CI: {confidence_level})"
    )
    param_table.add_row(
        "Run Panel Optimization",
        f"{run_panels} (Sizes: {panel_sizes}, Types: {panel_types}, N: {panel_candidates})",
    )
    param_table.add_row(
        "Run Advanced BC Check", f"{run_advanced_correction} (Alpha: {advanced_alpha})"
    )
    param_table.add_row("Figures Dir", str(task1_figures_dir))
    param_table.add_row("Tables Dir", str(task1_tables_dir))
    param_table.add_row("Processed Data Dir", str(processed_data_dir))
    console.print(param_table)

    # --- Instantiate Runners ---
    viz_runner = CorrelationVisualization()

    # --- Pipeline Execution ---
    ranked_markers_df = None
    correlation_df = None
    adata = None

    # Step 1: Load Data
    expression_data, manifest_data = load_data_task1()
    if expression_data is None:  # Need expression data to continue
        logger.error("Task 1 cannot proceed: Failed to load expression data.")
        return False

    # Step 2: Batch Correction
    # Pass manifest_data even if None, run_batch_correction handles it
    processed_expression_data = run_batch_correction(
        expression_data=expression_data,
        manifest_data=manifest_data,
        target_gene=target_gene,
        figures_dir=task1_figures_dir,
        tables_dir=task1_tables_dir,
        skip_batch_correction=skip_batch_correction,
        run_advanced_correction=run_advanced_correction,
        advanced_alpha=advanced_alpha,
        viz_runner=viz_runner,
    )
    if processed_expression_data is None:  # Signifies failure in correction step
        logger.error("Batch correction step failed critically. Attempting to use original data.")
        processed_expression_data = expression_data.copy()  # Fallback
        overall_success = False  # Mark overall task as having issues

    # Step 3: Correlation Analysis
    correlation_df, adata = run_correlation(
        expression_df=processed_expression_data,
        manifest_data=manifest_data,
        methods=methods,
        top_n=top_n,
        target_gene=target_gene,
        tables_dir=task1_tables_dir,
        run_bootstrap=run_bootstrap,
        bootstrap_iterations=bootstrap_iterations,
        confidence_level=confidence_level,
    )
    if correlation_df is None:
        logger.error("Correlation analysis failed. Dependent steps will be skipped.")
        overall_success = False
        # Still save processed data if it exists
        if isinstance(processed_expression_data, pd.DataFrame):
            save_processed_data(processed_expression_data, processed_data_dir)
        return False  # Cannot rank or visualize without correlations

    # Step 4: Gene Ranking
    ranked_markers_df, ranked_markers_list = run_ranking(
        correlation_df=correlation_df,
        expression_df=processed_expression_data,
        methods=methods,
        top_n=top_n,
        tables_dir=task1_tables_dir,
    )
    if ranked_markers_df is None:
        logger.warning("Gene ranking failed or produced no results.")
        # Allow continuing, but panel optimization might fail

    # Step 5: Panel Optimization
    if run_panels:
        if ranked_markers_df is not None and adata is not None:
            run_panel_optimization(
                ranked_markers_df=ranked_markers_df,
                expression_df=expression_data,  # Use original for mapping
                adata=adata,
                target_gene=target_gene,
                panel_sizes=panel_sizes,
                panel_types=panel_types,
                panel_candidates=panel_candidates,
                tables_dir=task1_tables_dir,
                figures_dir=task1_figures_dir,
                viz_runner=viz_runner,
            )
        else:
            logger.warning("Skipping panel optimization: Missing ranked markers or AnnData.")
    else:
        logger.info("Skipping marker panel optimization as per configuration.")

    # Step 6: Platform Consensus Analysis
    run_platform_consensus(
        expression_df=processed_expression_data,
        manifest_data=manifest_data,
        correlation_df=correlation_df,
        target_gene=target_gene,
        tables_dir=task1_tables_dir,
        figures_dir=task1_figures_dir,
        viz_runner=viz_runner,
    )

    # Step 7: Binary Marker Identification
    run_binary_markers(
        expression_df=processed_expression_data,
        target_gene=target_gene,
        tables_dir=task1_tables_dir,
    )

    # Step 8: Visualizations (Depends on various outputs)
    generate_visualizations_task1(
        correlation_df=correlation_df,
        adata=adata,
        ranked_markers_df=ranked_markers_df,
        methods=methods,
        top_n=top_n,
        target_gene=target_gene,
        figures_dir=task1_figures_dir,
        viz_runner=viz_runner,
    )

    # Step 9: Save Final Processed Data
    if isinstance(processed_expression_data, pd.DataFrame):
        # Prepare for saving (e.g., reset index, add symbols)
        processed_data_to_save = processed_expression_data.copy()
        id_col = "ensembl_transcript_id"
        sym_col = "sym"
        if (
            id_col not in processed_data_to_save.columns
            and processed_data_to_save.index.name == id_col
        ):
            processed_data_to_save = processed_data_to_save.reset_index()
        if (
            sym_col not in processed_data_to_save.columns
            and id_col in processed_data_to_save.columns
            and sym_col in expression_data.columns
        ):
            try:
                sym_map = pd.Series(
                    expression_data[sym_col].values, index=expression_data[id_col]
                ).to_dict()
                processed_data_to_save[sym_col] = processed_data_to_save[id_col].map(sym_map)
            except Exception as e:
                logger.warning(f"Could not map symbol column for saving: {e}")
        save_processed_data(processed_data_to_save, processed_data_dir)
    else:
        logger.error("Cannot save final processed data: Not a DataFrame.")

    # --- Task Completion Logging ---
    elapsed_time = time.time() - task_start_time
    final_status = "COMPLETED" if overall_success else "COMPLETED WITH ERRORS"
    console.rule(
        f"[bold {'green' if overall_success else 'red'}]TASK 1 {final_status}[/] in {elapsed_time:.2f}s",
        style="bold",
    )
    logger.info(
        f"Outputs: Tables='{task1_tables_dir}', Figures='{task1_figures_dir}', Processed='{processed_data_dir}'"
    )
    return overall_success


# ===================================================
#  === Task 2 Orchestrator ===
# ===================================================
@analysis_step("Task 2: Sequence Feature Analysis Orchestration")
def run_task2(args: argparse.Namespace) -> bool:
    """Orchestrates the execution of Task 2 with Rich logging."""
    task_start_time = time.time()
    overall_success = True
    analysis_results: dict[str, pd.DataFrame | None] = {}

    # --- Config and Paths ---
    try:
        task2_figures_dir = get_path("figures_dir") / "task2"
        task2_tables_dir = get_path("tables_dir") / "task2"
        processed_expr_path = get_path("preprocessed_dir") / "expressions_processed.csv"
        if not processed_expr_path.exists():
            logger.error(f"Input file for Task 2 missing: {processed_expr_path}")
            return False

        # Get Task 2 configuration
        sequence_config = get_sequence_analysis_config()

        # Feature significance settings
        run_feature_significance = sequence_config.get("run_feature_significance", False)
        significance_alpha = sequence_config.get("significance_alpha", 0.05)

        # Comparative analysis settings
        run_comparative_analysis = sequence_config.get("run_comparative_analysis", False)
        reference_type = sequence_config.get("comparative_reference_type", "high_expression")

        # Expression prediction settings
        run_expression_prediction = sequence_config.get("run_expression_prediction", False)
        prediction_target = sequence_config.get("prediction_target", "cv")
        model_type = sequence_config.get("prediction_model_type", "ensemble")
        feature_selection_method = sequence_config.get("feature_selection_method", "statistical")
        max_features = sequence_config.get("max_features", 15)

        # Transgene design settings
        run_transgene_design = sequence_config.get("run_transgene_design", False)
        optimization_target = sequence_config.get("optimization_target", "cv_minimization")

    except Exception as e:
        logger.critical(f"Config error Task 2: {e}")
        return False
    logger.info(f"Task 2 Output Dirs: Figures='{task2_figures_dir}', Tables='{task2_tables_dir}'")

    # --- Data Loading ---
    expression_df_task2 = None
    cds_seqs, utr5_seqs, utr3_seqs = None, None, None
    try:
        console.rule("[blue]1. Loading Data for Task 2[/]", style="blue")
        expression_df_task2 = pd.read_csv(processed_expr_path)
        logger.info(f"Loaded processed expression shape: {expression_df_task2.shape}")
        # Handle potential unnamed index column from saving Task 1 results
        if expression_df_task2.columns[0].startswith("Unnamed:"):
            expression_df_task2 = expression_df_task2.iloc[:, 1:]
            logger.debug("Removed unnamed index column from loaded processed data.")

        data_loader = DataLoader()
        cds_seqs, utr5_seqs, utr3_seqs = data_loader.load_all_sequence_data()
        if cds_seqs is None:  # CDS is essential for codon analysis
            logger.error("CDS sequences failed to load. Task 2 cannot proceed fully.")
            return False
    except Exception as e:
        logger.exception(f"Failed loading Task 2 data: {e}")
        return False

    # --- Base Sequence Analysis ---
    try:
        console.rule("[blue]2. Running Base Sequence Feature Analysis[/]", style="blue")
        seq_analyzer = SequenceFeatureAnalysis()
        analysis_results = seq_analyzer.run_analysis(
            expression_df_task2, cds_seqs, utr5_seqs, utr3_seqs
        )
        save_dataframe(
            analysis_results.get("consistent_genes"),
            "consistent_genes_stats.csv",
            task2_tables_dir,
            index=False,
        )
        save_dataframe(
            analysis_results.get("merged_data"),
            "merged_consistent_genes_sequences.csv",
            task2_tables_dir,
            index=False,
        )
        save_dataframe(
            analysis_results.get("combined_features"),
            "combined_sequence_features.csv",
            task2_tables_dir,
            index=False,
        )
        save_dataframe(
            analysis_results.get("codon_usage_table"),
            "codon_usage_frequencies.csv",
            task2_tables_dir,
            index=True,
        )

        if (
            analysis_results.get("combined_features") is None
            or analysis_results["combined_features"].empty
        ):
            logger.warning("Analysis produced no combined features.")
    except Exception as e:
        logger.exception(f"Error during Task 2 base analysis execution: {e}")
        overall_success = False

    # Get main data frames for enhanced analysis
    combined_features = analysis_results.get("combined_features")
    consistent_genes = analysis_results.get("consistent_genes")
    merged_data = analysis_results.get("merged_data")
    codon_usage_table = analysis_results.get("codon_usage_table")

    # If base analysis failed, we can't proceed with enhancements
    if not overall_success or combined_features is None or combined_features.empty:
        logger.error("Base analysis failed or produced no results, skipping enhancements")
        return False

    # --- Feature Significance Analysis ---
    significance_results = None
    if run_feature_significance:
        try:
            console.rule("[blue]3. Statistical Feature Significance Analysis[/]", style="blue")
            from cho_analysis.task2.feature_significance import (
                FeatureSignificanceAnalysis,
                FeatureSignificanceVisualization,
            )

            # Initialize analyzers
            significance_analyzer = FeatureSignificanceAnalysis()
            significance_viz = FeatureSignificanceVisualization()

            # Run analysis
            logger.info("Running feature significance analysis...")
            feature_significance = significance_analyzer.analyze_feature_significance(
                combined_features, alpha=significance_alpha
            )
            save_dataframe(
                feature_significance, "feature_significance.csv", task2_tables_dir, index=False
            )

            # Feature interactions
            feature_interactions = significance_analyzer.analyze_feature_interactions(
                combined_features
            )
            save_dataframe(
                feature_interactions, "feature_interactions.csv", task2_tables_dir, index=False
            )

            # Create variable genes for comparison
            variable_genes = None
            if consistent_genes is not None and "cv" in expression_df_task2.columns:
                try:
                    # Get the most variable genes (high CV)
                    high_cv_genes = expression_df_task2.nlargest(len(consistent_genes), "cv")
                    variable_gene_ids = high_cv_genes["ensembl_transcript_id"].tolist()

                    # Extract variable genes from merged data
                    if merged_data is not None and not merged_data.empty:
                        variable_genes = merged_data[
                            merged_data["ensembl_transcript_id"].isin(variable_gene_ids)
                        ]
                except Exception as e:
                    logger.warning(f"Error creating variable gene set: {e}")

            # UTR motif enrichment analysis
            if variable_genes is not None and merged_data is not None:
                # 5' UTR motif enrichment
                utr5_motifs = significance_analyzer.analyze_utr5_motif_enrichment(
                    merged_data, variable_genes
                )
                save_dataframe(
                    utr5_motifs, "utr5_motif_enrichment.csv", task2_tables_dir, index=False
                )

                # 3' UTR motif enrichment
                utr3_motifs = significance_analyzer.analyze_utr3_motif_enrichment(
                    merged_data, variable_genes
                )
                save_dataframe(
                    utr3_motifs, "utr3_motif_enrichment.csv", task2_tables_dir, index=False
                )

            # Visualizations
            if feature_significance is not None and not feature_significance.empty:
                # Feature significance volcano plot
                fig = significance_viz.plot_feature_significance(feature_significance)
                save_visualization(fig, "feature_significance_volcano.png", task2_figures_dir)

                # Feature correlation matrix
                fig = significance_viz.plot_feature_correlation_matrix(
                    combined_features, feature_significance
                )
                save_visualization(fig, "feature_correlation_matrix.png", task2_figures_dir)

            if feature_interactions is not None and not feature_interactions.empty:
                # Feature interaction network
                fig = significance_viz.plot_feature_interaction_network(feature_interactions)
                save_visualization(fig, "feature_interaction_network.png", task2_figures_dir)

            if "utr5_motifs" in significance_analyzer.results:
                # 5' UTR motif enrichment plot
                fig = significance_viz.plot_motif_enrichment(
                    significance_analyzer.results["utr5_motifs"], region="5'UTR"
                )
                save_visualization(fig, "utr5_motif_enrichment.png", task2_figures_dir)

            if "utr3_motifs" in significance_analyzer.results:
                # 3' UTR motif enrichment plot
                fig = significance_viz.plot_motif_enrichment(
                    significance_analyzer.results["utr3_motifs"], region="3'UTR"
                )
                save_visualization(fig, "utr3_motif_enrichment.png", task2_figures_dir)

            significance_results = significance_analyzer.results
            logger.info("Feature significance analysis completed")

        except Exception as e:
            logger.exception(f"Error during feature significance analysis: {e}")
            logger.warning("Continuing with other analyses...")

    # --- Comparative Sequence Analysis ---
    comparative_results = None
    if run_comparative_analysis and combined_features is not None:
        try:
            console.rule("[blue]4. Comparative Sequence Analysis[/]", style="blue")
            from cho_analysis.task2.comparative_analysis import (
                ComparativeSequenceAnalysis,
                ComparativeSequenceVisualization,
            )

            # Initialize analyzers
            comparative_analyzer = ComparativeSequenceAnalysis()
            comparative_viz = ComparativeSequenceVisualization()

            # Feature comparison between consistent and variable genes
            if variable_genes is not None:
                logger.info("Running comparative feature analysis...")
                feature_comparison = comparative_analyzer.compare_sequence_features(
                    combined_features, variable_genes
                )
                save_dataframe(
                    feature_comparison, "feature_comparison.csv", task2_tables_dir, index=False
                )

                # Feature comparison visualization
                fig = comparative_viz.plot_feature_comparison(feature_comparison)
                save_visualization(fig, "feature_comparison.png", task2_figures_dir)

            # Reference gene comparison
            logger.info("Comparing with reference gene profiles...")
            reference_comparison = comparative_analyzer.compare_with_reference_genes(
                combined_features, reference_type=reference_type
            )
            save_dataframe(
                reference_comparison,
                f"reference_comparison_{reference_type}.csv",
                task2_tables_dir,
                index=False,
            )

            # Reference similarity visualization
            fig = comparative_viz.plot_reference_similarity(
                reference_comparison, reference_type=reference_type
            )
            save_visualization(fig, "reference_similarity.png", task2_figures_dir)

            # RNA structure analysis (if sequences available)
            if merged_data is not None and not merged_data.empty:
                # Extract sequences for structure analysis
                utr5_sequences = {}
                utr3_sequences = {}

                if (
                    "UTR5_Seq" in merged_data.columns
                    and "ensembl_transcript_id" in merged_data.columns
                ):
                    for _, row in merged_data.iterrows():
                        if pd.notna(row["UTR5_Seq"]):
                            utr5_sequences[row["ensembl_transcript_id"]] = row["UTR5_Seq"]

                if (
                    "UTR3_Seq" in merged_data.columns
                    and "ensembl_transcript_id" in merged_data.columns
                ):
                    for _, row in merged_data.iterrows():
                        if pd.notna(row["UTR3_Seq"]):
                            utr3_sequences[row["ensembl_transcript_id"]] = row["UTR3_Seq"]

                if utr5_sequences or utr3_sequences:
                    logger.info("Analyzing RNA secondary structures...")
                    rna_structures = comparative_analyzer.analyze_rna_structure(
                        utr5_sequences, utr3_sequences
                    )
                    save_dataframe(
                        rna_structures, "rna_structure_analysis.csv", task2_tables_dir, index=False
                    )

                    # RNA structure visualization
                    fig = comparative_viz.plot_rna_structures(rna_structures)
                    save_visualization(fig, "rna_structures.png", task2_figures_dir)

            comparative_results = comparative_analyzer.results
            logger.info("Comparative sequence analysis completed")

        except Exception as e:
            logger.exception(f"Error during comparative sequence analysis: {e}")
            logger.warning("Continuing with other analyses...")

    # --- Sequence-Based Expression Prediction ---
    prediction_results = None
    prediction_model = None
    if run_expression_prediction and combined_features is not None:
        try:
            console.rule("[blue]5. Sequence-Based Expression Prediction[/]", style="blue")
            from cho_analysis.task2.sequence_modeling import (
                SequenceExpressionModeling,
                SequenceModelingVisualization,
            )

            # Initialize analyzers
            modeling = SequenceExpressionModeling()
            modeling_viz = SequenceModelingVisualization()

            # Handle potential comma-separated prediction targets (e.g., "cv,mean")
            prediction_targets = (
                [target.strip() for target in prediction_target.split(",")]
                if isinstance(prediction_target, str)
                else [prediction_target]
            )
            logger.info(
                f"Processing {len(prediction_targets)} prediction targets: {prediction_targets}"
            )

            # Create a mapping between target names in config and column names in dataframe
            target_map = {"mean": "mean_expression", "cv": "cv"}

            # Process each prediction target separately
            for target in prediction_targets:
                # Map the target name to the actual column name
                actual_column = target_map.get(target, target)

                if actual_column not in combined_features.columns:
                    logger.error(
                        f"Target column '{actual_column}' not found in combined_features. Available columns: {list(combined_features.columns)}"
                    )
                    continue

                logger.info(f"Processing prediction target: {target}")

                # Feature selection for this target
                logger.info(f"Selecting predictive features for {target}...")
                selected_features = modeling.select_predictive_features(
                    combined_features,
                    expression_metric=target,
                    selection_method=feature_selection_method,
                    max_features=max_features,
                )
                save_dataframe(
                    selected_features,
                    f"selected_predictive_features_{target}.csv",
                    task2_tables_dir,
                    index=False,
                )

                # Build prediction model for this target
                logger.info(f"Building {model_type} model for {target} prediction...")
                model_performance = modeling.build_expression_prediction_model(
                    combined_features, target=target, model_type=model_type
                )

                # Save model performance metrics
                if model_performance:
                    performance_df = pd.DataFrame([model_performance])
                    save_dataframe(
                        performance_df,
                        f"model_performance_metrics_{target}.csv",
                        task2_tables_dir,
                        index=False,
                    )

                # Use the model to predict expression from sequence features
                logger.info(f"Predicting {target} from sequence features...")
                try:
                    # Use the trained model to predict on the original data
                    predictions_df = modeling.predict_expression_from_sequence(
                        combined_features, target=target
                    )

                    # Save predictions
                    if predictions_df is not None and not predictions_df.empty:
                        save_dataframe(
                            predictions_df,
                            f"sequence_based_predictions_{target}.csv",
                            task2_tables_dir,
                            index=False,
                        )

                        # Plot prediction performance if visualization module available
                        if hasattr(modeling_viz, "plot_prediction_performance") and modeling.models:
                            # Get the model for visualization
                            model_for_viz = None
                            for name, model in modeling.models.items():
                                if name != "ensemble_avg" and model != "average":
                                    model_for_viz = model
                                    break

                            if model_for_viz is None and modeling.models:
                                model_for_viz = next(iter(modeling.models.values()))

                            # Extract features and target
                            X_features = combined_features[modeling.feature_cols].values
                            y_target = combined_features[actual_column].values

                            # Create prediction performance plot
                            fig = modeling_viz.plot_prediction_performance(
                                model_for_viz,
                                X_features,
                                y_target,
                                feature_names=modeling.feature_cols,
                                target_name=target,
                            )
                            save_visualization(
                                fig, f"prediction_performance_{target}.png", task2_figures_dir
                            )

                        logger.info(
                            f"Successfully generated predictions for {target} using {len(predictions_df)} sequences"
                        )
                    else:
                        logger.warning(f"Failed to generate valid predictions for {target}")

                except Exception as e:
                    logger.exception(f"Error during sequence-based prediction for {target}: {e}")
                    logger.warning("Continuing with other targets or analyses...")

                # Store results for downstream use
                if modeling.results.get("model_performance"):
                    # Save feature importance data
                    importance_df = modeling.results.get("selected_features")
                    if importance_df is not None and not importance_df.empty:
                        save_dataframe(
                            importance_df,
                            f"feature_importance_{target}.csv",
                            task2_tables_dir,
                            index=False,
                        )

                        # Generate feature importance visualization
                        if hasattr(modeling_viz, "plot_feature_importance"):
                            fig = modeling_viz.plot_feature_importance(importance_df)
                            save_visualization(
                                fig, f"feature_importance_{target}.png", task2_figures_dir
                            )

            # Store the last target's model and results for downstream tasks
            prediction_results = modeling.results
            prediction_model = None
            for name, model in modeling.models.items():
                if name != "ensemble_avg" and isinstance(model, object) and not isinstance(model, str):
                    prediction_model = model
                    break

            logger.info("Sequence-based expression prediction completed")

        except Exception as e:
            logger.exception(f"Error during sequence-based expression prediction: {e}")
            logger.warning("Continuing with other analyses...")

    # --- Transgene Design Recommendations ---
    if run_transgene_design and combined_features is not None:
        try:
            console.rule("[blue]6. Transgene Design Recommendations[/]", style="blue")
            from cho_analysis.task2.transgene_design import (
                TransgeneDesignRecommendations,
                TransgeneDesignVisualization,
            )

            # Initialize analyzers
            design = TransgeneDesignRecommendations()
            design_viz = TransgeneDesignVisualization()

            # Generate optimal feature profiles
            logger.info("Generating optimal feature profiles...")
            optimal_profiles = design.generate_optimal_feature_profiles(
                combined_features, prediction_model=prediction_model
            )
            save_dataframe(
                optimal_profiles, "optimal_feature_profiles.csv", task2_tables_dir, index=False
            )

            # Generate specific recommendations
            logger.info("Generating specific design recommendations...")
            recommendations = design.generate_specific_recommendations(combined_features)

            # Save recommendations to JSON
            if recommendations:
                with open(task2_tables_dir / "design_recommendations.json", "w") as f:
                    import json

                    json.dump(recommendations, f, indent=2)

            # Generate sequence templates
            logger.info("Generating sequence templates...")
            optimal_features = {}
            if optimal_profiles is not None and not optimal_profiles.empty:
                for _, row in optimal_profiles.iterrows():
                    if "feature" in row and "balanced_profile_mean" in row:
                        optimal_features[row["feature"]] = row["balanced_profile_mean"]

            templates = design.generate_sequence_templates(optimal_features)

            # Save templates to JSON
            if templates:
                with open(task2_tables_dir / "sequence_templates.json", "w") as f:
                    import json

                    json.dump(templates, f, indent=2)

            # Visualizations
            if optimal_profiles is not None and not optimal_profiles.empty:
                fig = design_viz.plot_optimal_feature_ranges(optimal_profiles)
                save_visualization(fig, "optimal_feature_ranges.png", task2_figures_dir)

            if recommendations:
                fig = design_viz.plot_design_rule_flowchart(recommendations)
                save_visualization(fig, "design_rule_flowchart.png", task2_figures_dir)

            if templates:
                fig = design_viz.plot_sequence_templates(templates)
                save_visualization(fig, "sequence_templates.png", task2_figures_dir)

            # Optimization impact analysis (if prediction model available)
            if prediction_model is not None:
                feature_ranges = {}
                if optimal_profiles is not None and not optimal_profiles.empty:
                    for _, row in optimal_profiles.iterrows():
                        if (
                            "feature" in row
                            and "balanced_profile_range" in row
                            and isinstance(row["balanced_profile_range"], tuple)
                        ):
                            feature_ranges[row["feature"]] = row["balanced_profile_range"]

                if feature_ranges:
                    logger.info("Analyzing optimization impact...")
                    impact_analysis = design.analyze_optimization_impact(
                        feature_ranges, prediction_model
                    )
                    save_dataframe(
                        impact_analysis, "optimization_impact.csv", task2_tables_dir, index=False
                    )

                    fig = design_viz.plot_optimization_impact(impact_analysis)
                    save_visualization(fig, "optimization_impact.png", task2_figures_dir)

            logger.info("Transgene design recommendations completed")

        except Exception as e:
            logger.exception(f"Error during transgene design recommendations: {e}")
            logger.warning("Some transgene design tasks may not have completed")

    # --- Base Visualizations (original) ---
    if overall_success and combined_features is not None and not combined_features.empty:
        console.rule("[blue]7. Generating Task 2 Base Visualizations[/]", style="blue")
        viz = SequenceFeatureVisualization()
        figures_generated = 0

        def _try_plot_task2(plot_func, filename, *args, **kwargs):
            nonlocal figures_generated
            plot_func_name = getattr(plot_func, "__name__", "unnamed_plot_func")
            logger.info(f":bar_chart: Generating '{filename}' using {plot_func_name}...")
            try:
                fig = plot_func(*args, **kwargs)
                if isinstance(fig, Figure):
                    save_visualization(fig, filename, task2_figures_dir)
                    figures_generated += 1
                elif fig is None:
                    logger.warning(
                        f"Plot function {plot_func_name} for '{filename}' returned None."
                    )
                else:
                    logger.error(
                        f"Plot function {plot_func_name} for '{filename}' returned unexpected type: {type(fig)}"
                    )
            except Exception as e:
                logger.exception(f"Failed plot '{filename}': {e}", exc_info=False)

        # Call plotting functions
        _try_plot_task2(viz.plot_utr_length_distribution, "utr_length_dist.png", combined_features)
        _try_plot_task2(viz.plot_gc_distribution, "gc_content_dist.png", combined_features)
        if "UTR5_length" in combined_features.columns:
            _try_plot_task2(
                viz.plot_cv_vs_feature,
                "cv_vs_utr5_length.png",
                combined_features,
                "UTR5_length",
                "5' UTR Length",
            )
        if "UTR3_length" in combined_features.columns:
            _try_plot_task2(
                viz.plot_cv_vs_feature,
                "cv_vs_utr3_length.png",
                combined_features,
                "UTR3_length",
                "3' UTR Length",
            )
        if "GC3_content" in combined_features.columns:
            _try_plot_task2(
                viz.plot_cv_vs_feature,
                "cv_vs_gc3.png",
                combined_features,
                "GC3_content",
                "CDS GC3 Content (%)",
            )
        if "CAI" in combined_features.columns:
            _try_plot_task2(
                viz.plot_cv_vs_feature,
                "cv_vs_cai.png",
                combined_features,
                "CAI",
                "Codon Adaptation Index (CAI)",
            )
        _try_plot_task2(
            viz.plot_regulatory_element_counts, "regulatory_elements.png", combined_features
        )
        if codon_usage_table is not None and not codon_usage_table.empty:
            _try_plot_task2(
                viz.plot_codon_usage_heatmap,
                "codon_usage_heatmap.png",
                codon_usage_table,
                combined_features,
                top_n=20,
            )
            _try_plot_task2(
                viz.plot_average_codon_usage, "avg_codon_usage.png", codon_usage_table, top_n=20
            )

        logger.info(f"Task 2 Base Visualization complete. {figures_generated} figures saved.")
    elif not overall_success:
        logger.error("Skipping Task 2 visualizations due to analysis errors.")
    else:  # Success but no features
        logger.warning("Skipping Task 2 visualizations: No combined feature data generated.")

    # Task Completion
    elapsed_time = time.time() - task_start_time
    final_status = "COMPLETED" if overall_success else "FAILED"
    console.rule(
        f"[bold {'green' if overall_success else 'red'}]TASK 2 {final_status}[/] in {elapsed_time:.2f}s",
        style="bold",
    )
    logger.info(f"Task 2 Outputs: Tables='{task2_tables_dir}', Figures='{task2_figures_dir}'")
    return overall_success


def apply_experiment_config(exp_number: int | None) -> int | None:
    """Loads experiment config and sets environment variables if needed."""
    if exp_number is None:
        return None

    config_path = Path("experiments.toml")
    if not config_path.exists():
        logger.error(f"Experiment config file not found: {config_path}")
        return None

    logger.info(f"Applying configuration for experiment {exp_number} from {config_path}...")
    try:
        with open(config_path, "rb") as f:
            all_experiments_data = tomllib.load(f)

        if "experiment" not in all_experiments_data:
            logger.error(f"'[experiment]' table not found in {config_path}")
            return None

        experiment_table = all_experiments_data["experiment"]
        exp_number_str = str(exp_number)

        if exp_number_str not in experiment_table:
            logger.error(
                f"Experiment key '{exp_number_str}' not defined under [experiment] in {config_path}"
            )
            return None

        exp_config = experiment_table[exp_number_str]

        exp_task = exp_config.get("task")
        logger.info(f"Experiment Description: {exp_config.get('description', 'N/A')}")

        for key, value in exp_config.items():
            if key.upper().startswith("CHO_ANALYSIS"):
                value_str = ",".join(map(str, value)) if isinstance(value, list) else str(value)
                os.environ[key.upper()] = value_str
                logger.debug(f"  Override via experiment: {key.upper()}='{value_str}'")

        # Ensure the task is an integer if found
        if exp_task is not None:
            try:
                return int(exp_task)
            except (ValueError, TypeError):
                logger.exception(
                    f"Invalid 'task' value '{exp_task}' for experiment {exp_number}. Must be an integer."
                )
                return None
        else:
            logger.warning(
                f"'task' key not found for experiment {exp_number}. Task execution might rely on --task flag or default."
            )
            return None  # Return None if task is not explicitly defined

    except Exception as e:
        logger.exception(f"Error loading or applying experiment {exp_number} config: {e}")
        return None


def list_experiments() -> None:
    """Lists all available experiments from the experiments.toml file."""
    config_path = Path("experiments.toml")
    if not config_path.exists():
        logger.error(f"Experiment config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "rb") as f:
            all_experiments_data = tomllib.load(f)

        if "experiment" not in all_experiments_data:
            logger.error(f"'[experiment]' table not found in {config_path}")
            sys.exit(1)

        experiment_table = all_experiments_data["experiment"]
        if not experiment_table:
            console.print("[yellow]No experiments defined in experiments.toml[/]")
            sys.exit(0)

        # Create a Rich table for experiments
        table = Table(title="Available Experiments", show_header=True, header_style="bold")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Task", style="green", justify="center")
        table.add_column("Description", style="white")

        # Sort experiments by ID
        sorted_exp_ids = sorted(
            experiment_table.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
        )

        for exp_id in sorted_exp_ids:
            exp_config = experiment_table[exp_id]
            task = exp_config.get("task", "N/A")
            description = exp_config.get("description", "No description")
            table.add_row(exp_id, str(task), description)

        console.print(table)

    except Exception as e:
        logger.exception(f"Error loading or parsing experiments.toml: {e}")
        sys.exit(1)


# ===================================================
#  === Main Execution ===
# ===================================================
def main() -> None:
    logging.captureWarnings(True)
    parser = argparse.ArgumentParser(
        description="CHO cell line analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Task to run (0: Both, 1: Correlation, 2: Sequence Features). Overridden by --experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        metavar="N",
        default=None,
        help="Run predefined experiment N from experiments.toml. Overrides --task and sets env vars.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available predefined experiments from experiments.toml and exit.",
    )
    task1_group = parser.add_argument_group("Task 1 Manual Overrides")
    task1_group.add_argument(
        "--methods",
        type=str,
        nargs="+",
        metavar="M",
        default=None,
        help="Override correlation methods.",
    )
    task1_group.add_argument(
        "--top-n", type=int, metavar="N", default=None, help="Override number of top genes."
    )
    task1_group.add_argument(
        "--skip-batch-correction", action="store_true", help="Override batch correction setting."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG level logging.")
    args = parser.parse_args()

    if args.list_experiments:
        list_experiments()
        sys.exit(0)

    if args.experiment is not None:
        experiment_task = apply_experiment_config(args.experiment)
        if experiment_task is None:
            logger.error(f"Failed to load or apply experiment {args.experiment}. Exiting.")
            sys.exit(1)
        if args.task != experiment_task:
            logger.info(
                f"Overriding --task setting. Running Task {experiment_task} as defined by experiment {args.experiment}."
            )
            args.task = experiment_task

    if args.verbose:
        try:
            config_root_logger_name = get_logging_config().get("root_logger_name", "cho_analysis")
            logging.getLogger(config_root_logger_name).setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled.")
        except Exception as e:
            logger.exception(f"Failed to set verbose logging level: {e}")

    console.print(
        Panel(
            "[bold yellow]CHO Cell Line Analysis Pipeline Initializing...[/]", border_style="yellow"
        )
    )
    logger.info(
        ":file_folder: Output directories expected to be handled by entrypoint script or exist."
    )

    run_task_1_flag = args.task in {0, 1}
    run_task_2_flag = args.task in {0, 2}
    overall_start_time = time.time()
    logger.info(f"Pipeline starting (Effective Task: {args.task})")

    task1_status: bool | None = None
    task2_status: bool | None = None
    pipeline_critical_error = False

    try:
        if run_task_1_flag:
            task1_status = run_task1(args)

        if run_task_2_flag:
            if run_task_1_flag and task1_status is False:
                logger.warning(
                    ":warning: Skipping Task 2: Task 1 failed or did not complete successfully."
                )
                task2_status = False
            else:
                if run_task_1_flag and task1_status is True:
                    console.rule("[bold purple]--- Transitioning to Task 2 ---[/]", style="purple")
                elif not run_task_1_flag:
                    console.rule("[bold purple]--- Starting Task 2 ---[/]", style="purple")
                task2_status = run_task2(args)

    except Exception as e:
        logger.critical(f":skull: Pipeline orchestration failed critically: {e}", exc_info=True)
        pipeline_critical_error = True
    finally:
        overall_elapsed_time = time.time() - overall_start_time
        tasks_executed_desc = []
        overall_success = True

        if run_task_1_flag:
            tasks_executed_desc.append("Task 1")
            if task1_status is False:
                overall_success = False
        if run_task_2_flag:
            tasks_executed_desc.append("Task 2")
            if task2_status is False:
                overall_success = False

        if pipeline_critical_error:
            final_status_message = "[bold red]Pipeline terminated due to a critical error[/]"
            overall_success = False
        elif not tasks_executed_desc:
            final_status_message = "[yellow]Pipeline finished (no tasks selected)[/]"
        elif overall_success:
            final_status_message = (
                f"[bold green]Pipeline ({' & '.join(tasks_executed_desc)}) finished successfully[/]"
            )
        else:
            final_status_message = f"[bold yellow]Pipeline ({' & '.join(tasks_executed_desc)}) finished with errors or skipped steps[/]"

        exit_code = 0 if overall_success and not pipeline_critical_error else 1
        console.print(
            Panel(
                f"{final_status_message}\nTotal execution time: {overall_elapsed_time:.2f} seconds.",
                title="[bold]Pipeline Summary[/]",
                border_style="bold green" if overall_success else "bold red",
            )
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
