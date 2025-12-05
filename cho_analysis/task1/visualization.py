# cho_analysis/task1/visualization.py
"""Visualization utilities for Task 1 (gene correlation analysis)."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as sd
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

# Conditional imports
try:
    import scanpy as sc

    # Define AnnData as a proper type that can be used in annotations
    from anndata import AnnData

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    # Create a placeholder type for AnnData when not available
    AnnData = Any  # type: ignore
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from cho_analysis.core.config import get_correlation_config, get_visualization_config
from cho_analysis.core.visualization_utils import (
    BASE_RC_PARAMS,
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    DEFAULT_DPI,
    DEFAULT_FONT_FAMILY,
    DEFAULT_STYLE,
    EDGE_COLOR_NEG,
    EDGE_COLOR_POS,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_FIGURE_TITLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    NODE_COLOR_NEG,
    NODE_COLOR_POS,
    PALETTE_QUALITATIVE,
)
from cho_analysis.task1.advanced_batch_correction import AdvancedBatchCorrection

logger = logging.getLogger(__name__)


class CorrelationVisualization:
    """Generates visualizations for correlation results."""

    def __init__(self):
        self.figures: dict[str, Figure] = {}
        self.viz_config = get_visualization_config()
        self.corr_config = get_correlation_config()
        plt.style.use(self.viz_config.get("style", DEFAULT_STYLE))
        plt.rcParams.update(BASE_RC_PARAMS)
        logger.debug(f"Using plot style: {self.viz_config.get('style', DEFAULT_STYLE)}")
        logger.debug(f"Applied base rcParams including font: {DEFAULT_FONT_FAMILY}")

    # Helper Methods
    def _check_required_columns(
        self, df: pd.DataFrame | None, required_cols: list[str], plot_name: str
    ) -> bool:
        """Checks if a DataFrame exists and contains all required columns."""
        if df is None:
            logger.error(f"Cannot generate {plot_name}: Input DataFrame is None.")
            return False
        if df.empty:
            logger.error(f"Cannot generate {plot_name}: Input DataFrame is empty.")
            return False
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(
                f"Cannot generate {plot_name}: DataFrame missing required columns: {missing_cols}"
            )
            return False
        # Check for all-NaN columns which can cause issues
        for col in required_cols:
            if df[col].isnull().all():
                logger.warning(f"Required column '{col}' for {plot_name} contains only NaN values.")
        return True

    def _get_effective_method(self, method: str | None) -> str:
        """Gets the correlation method to use, handling None and defaults."""
        if method:
            return method.capitalize()
        default_method = self.corr_config.get("default_method", "spearman")
        return default_method.capitalize() if isinstance(default_method, str) else "Spearman"

    def _filter_and_sort_corr_data(
        self, correlation_df: pd.DataFrame, method: str, top_n: int
    ) -> pd.DataFrame:
        """Filters correlation data for a method, sorts, and returns top N."""
        if correlation_df is None or correlation_df.empty:
            msg = "Correlation DataFrame is empty or None."
            logger.error(msg)
            raise ValueError(msg)

        method_df = correlation_df[correlation_df["Correlation_Type"] == method]
        if method_df.empty:
            available = correlation_df["Correlation_Type"].unique()
            msg = f"No results found for method '{method}'. Available: {available}"
            logger.error(msg)
            raise ValueError(msg)

        sort_col = "Correlation_Coefficient_Abs"
        if sort_col not in method_df.columns:
            alt_sort_col = "Correlation_Coefficient"
            if alt_sort_col in method_df.columns:
                logger.warning(
                    f"Sorting column '{sort_col}' not found, using absolute of '{alt_sort_col}'."
                )
                method_df = method_df.copy()
                method_df[sort_col] = method_df[alt_sort_col].abs()
            else:
                msg = f"Cannot find suitable sorting column ({sort_col} or {alt_sort_col})."
                logger.error(msg)
                raise ValueError(msg)

        top_genes = method_df.sort_values(by=sort_col, ascending=False, na_position="last").head(
            top_n
        )
        if top_genes.empty:
            msg = f"No top genes found for method '{method}' after sorting."
            logger.error(msg)
            raise ValueError(msg)

        required_cols = ["ensembl_transcript_id", "Correlation_Coefficient", "Correlation_P_Value"]
        missing_req = [col for col in required_cols if col not in top_genes.columns]
        if missing_req:
            msg = f"Filtered top genes DataFrame missing required columns: {missing_req}"
            logger.error(msg)
            raise ValueError(msg)

        if "sym" not in top_genes.columns:
            logger.warning("Column 'sym' not found, using Ensembl IDs for labels.")

        return top_genes

    def _get_gene_labels(self, top_genes_df: pd.DataFrame) -> np.ndarray:
        """Gets gene labels, preferring 'sym' but falling back to ID."""
        gene_ids = top_genes_df["ensembl_transcript_id"].astype(str).values
        if "sym" in top_genes_df.columns and not top_genes_df["sym"].isnull().all():
            # Fill missing symbols with their corresponding gene ID
            labels = top_genes_df["sym"].fillna(pd.Series(gene_ids, index=top_genes_df.index))
            return labels.astype(str).values
        return gene_ids

    def _create_error_figure(self, error_message: str, figsize: tuple) -> Figure:
        """Creates a blank figure displaying an error message."""
        fig, ax = plt.subplots(figsize=figsize, dpi=self.viz_config.get("default_dpi", 100))
        ax.text(
            0.5,
            0.5,
            f"Plotting Error:\n{error_message}",
            ha="center",
            va="center",
            wrap=True,
            color="red",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(fig=fig, left=True, bottom=True)
        return fig

    # Plotting Methods
    # --------------------------------------------------------------------------
    def plot_correlation_heatmap(
        self,
        correlation_df: pd.DataFrame,
        top_n: int = 20,
        method: str | None = None,
        figsize: tuple[int, int] | None = None,
        cluster: bool = True,
    ) -> Figure:
        """Creates a heatmap of top correlated genes."""
        effective_method = self._get_effective_method(method)
        fig_size = figsize or self.viz_config.get("default_figsize", (12, 10))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        try:
            top_genes = self._filter_and_sort_corr_data(correlation_df, effective_method, top_n)
            gene_ids, labels = (
                top_genes["ensembl_transcript_id"].values,
                self._get_gene_labels(top_genes),
            )
            n_genes = len(gene_ids)
            if n_genes == 0:
                msg = "No genes available after filtering."
                raise ValueError(msg)

            corr_coeffs = pd.to_numeric(top_genes["Correlation_Coefficient"], errors="coerce")
            gene_corr_map = dict(zip(gene_ids, corr_coeffs, strict=False))

            # Calculate co-correlation matrix
            corr_matrix = np.full((n_genes, n_genes), np.nan, dtype=np.float64)
            for i in range(n_genes):
                corr1 = gene_corr_map.get(gene_ids[i], np.nan)
                for j in range(n_genes):
                    corr2 = gene_corr_map.get(gene_ids[j], np.nan)
                    corr_matrix[i, j] = (
                        1.0
                        if i == j
                        else (corr1 * corr2 if pd.notna(corr1) and pd.notna(corr2) else np.nan)
                    )

            if np.isnan(corr_matrix).all():
                msg = "Co-correlation matrix contains only NaNs."
                raise ValueError(msg)

        except (ValueError, KeyError) as e:
            logger.exception(f"Heatmap data preparation failed: {e}")
            return self._create_error_figure(str(e), fig_size)

        # Clustering
        clustered = False
        if cluster and n_genes > 1:
            try:
                # Use correlation distance (1 - abs(corr))
                dist_matrix = 1.0 - np.abs(corr_matrix)
                dist_matrix = np.nan_to_num(
                    dist_matrix, nan=1.0
                )  # Treat NaN correlation as max distance
                np.fill_diagonal(dist_matrix, 0)
                dist_matrix = np.maximum(dist_matrix, 0)  # Ensure non-negative

                # Check for valid condensed distance matrix
                if (
                    np.all(np.isfinite(dist_matrix))
                    and dist_matrix.shape[0] == dist_matrix.shape[1]
                ):
                    condensed_dist = sd.squareform(dist_matrix, checks=True)
                    if (
                        len(condensed_dist) > 0
                    ):  # Need at least 3 elements for squareform output if n>1
                        link = sch.linkage(condensed_dist.astype(np.float64), method="average")
                        idx = sch.leaves_list(link)
                        corr_matrix, labels = corr_matrix[idx, :][:, idx], labels[idx]
                        clustered = True
                        logger.info("Successfully clustered heatmap data.")
                    else:
                        logger.warning("Not enough data points for clustering distance matrix.")
                else:
                    logger.warning(
                        "Distance matrix invalid for clustering (non-finite values or wrong shape)."
                    )

            except Exception as e:
                logger.warning(f"Clustering failed: {e}. Plotting without clustering.")

        # Plotting
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=CMAP_DIVERGING,
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": max(4, FONT_SIZE_ANNOTATION - n_genes // 10)},
            linewidths=0.5,
            linecolor="lightgrey",
            cbar_kws={"shrink": 0.8},
        )
        cluster_status = ""
        if cluster:
            cluster_status = " (Clustered)" if clustered else " (Clustering Failed)"
        title = (
            f"Gene Co-correlation Matrix ({effective_method}, Top {len(labels)}){cluster_status}"
        )
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
        ax.tick_params(axis="both", which="major", length=0)
        sns.despine(fig=fig, ax=ax, left=True, bottom=True)
        fig.tight_layout()
        self.figures["correlation_heatmap"] = fig
        return fig

    def plot_scatter_matrix(
        self,
        adata: AnnData | None,
        correlation_df: pd.DataFrame,
        top_n: int = 5,
        method: str | None = None,
        target_col: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Creates a scatter matrix (pair plot) of top genes vs. target."""
        fig_size = figsize or self.viz_config.get("default_figsize", (10, 10))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        if not SCANPY_AVAILABLE:
            return self._create_error_figure("Scanpy not installed", fig_size)
        if adata is None:
            return self._create_error_figure("AnnData object is required", fig_size)
        effective_method = self._get_effective_method(method)
        if target_col is None:
            if not adata.obs.empty:
                target_col = adata.obs.columns[0]
                logger.warning(f"target_col defaulting: '{target_col}'")
            else:
                return self._create_error_figure("Cannot determine target_col", fig_size)
        elif target_col not in adata.obs.columns:
            return self._create_error_figure(f"Target column '{target_col}' not found", fig_size)
        try:
            top_genes = self._filter_and_sort_corr_data(correlation_df, effective_method, top_n)
            gene_ids, gene_symbols = (
                top_genes["ensembl_transcript_id"].values,
                self._get_gene_labels(top_genes),
            )
            if len(gene_ids) == 0:
                msg = "No top genes."
                raise ValueError(msg)
            scatter_df = pd.DataFrame({target_col: adata.obs[target_col]}, index=adata.obs_names)
            genes_found_in_adata = []
            for gid, gsymbol in zip(gene_ids, gene_symbols, strict=False):
                if gid in adata.var_names:
                    expr_vector = adata[:, gid].X
                    if hasattr(expr_vector, "toarray"):
                        expr_vector = expr_vector.toarray()
                    scatter_df[gsymbol] = np.asarray(expr_vector).flatten()
                    genes_found_in_adata.append(gsymbol)
                else:
                    logger.warning(f"Gene {gid} ({gsymbol}) not found in AnnData.")
            if not genes_found_in_adata:
                raise ValueError("None of the top genes found in AnnData.")
            if len(genes_found_in_adata) < top_n:
                logger.warning(f"Plotting for {len(genes_found_in_adata)} found genes.")
        except (ValueError, KeyError) as e:
            logger.error(f"Scatter matrix data prep failed: {e}")
            return self._create_error_figure(str(e), fig_size)

        # Plotting
        plot_vars = [target_col] + genes_found_in_adata
        try:
            grid = sns.pairplot(
                scatter_df[plot_vars].dropna(),
                kind="scatter",
                diag_kind="kde",
                corner=True,
                plot_kws={
                    "s": 25,
                    "alpha": 0.6,
                    "edgecolor": "w",
                    "linewidth": 0.5,
                },  # Added edge for points
                diag_kws={"fill": True, "alpha": 0.5, "linewidth": 1.5},
            )
        except Exception as plot_err:
            logger.error(f"Seaborn pairplot failed: {plot_err}")
            return self._create_error_figure(f"Pairplot failed: {plot_err}", fig_size)

        grid.figure.set_size_inches(fig_size)
        grid.figure.set_dpi(fig_dpi)

        # Add correlation annotations & Format Axes
        corr_map = dict(zip(gene_symbols, top_genes["Correlation_Coefficient"], strict=False))
        pval_map = dict(zip(gene_symbols, top_genes["Correlation_P_Value"], strict=False))

        # FIX: Apply thousands separators and adjust annotations
        formatter = ticker.StrMethodFormatter("{x:,.0f}")  # Formatter for integers with commas
        for i, row_var in enumerate(grid.y_vars):
            for j, col_var in enumerate(grid.x_vars):
                ax = grid.axes[i, j]
                if ax is None:
                    continue  # Skip empty upper triangle

                # Apply formatter to both axes
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK - 1)  # Slightly smaller ticks

                # Annotate only off-diagonal plots involving the target column
                if i > j and col_var == target_col and row_var in corr_map:
                    corr = corr_map.get(row_var, np.nan)
                    pval = pval_map.get(row_var, np.nan)
                    if pd.notna(corr) and pd.notna(pval):
                        pval_str = f"p={pval:.1e}" if pval < 0.001 else f"p={pval:.3f}"
                        ax.text(
                            0.05,
                            0.95,  # Position within axes
                            f"r = {corr:.3f}\n{pval_str}",
                            transform=ax.transAxes,
                            fontsize=FONT_SIZE_ANNOTATION - 2,  # Make annotation smaller
                            va="top",
                            ha="left",
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "fc": "white",
                                "alpha": 0.7,
                                "ec": "lightgrey",
                            },  # Smaller pad
                        )

        grid.figure.suptitle(
            f"Top {len(genes_found_in_adata)} Gene Expression vs. {target_col} ({effective_method})",
            y=1.02,  # Adjusted title position slightly
            fontsize=FONT_SIZE_FIGURE_TITLE,  # Use constant
        )
        sns.despine(fig=grid.figure)
        try:
            grid.figure.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust rect
        except ValueError:
            pass
        self.figures["scatter_matrix"] = grid.figure
        return grid.figure

    def plot_expression_by_platform(
        self,
        adata: AnnData | None,
        correlation_df: pd.DataFrame,
        top_n: int = 5,
        method: str | None = None,
        target_col: str | None = None,
        platform_col: str = "platform",
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Creates violin/swarm plots of top gene expression colored by target,
        faceted by platform.

        Args:
            adata: AnnData object with expression data in `.X` and metadata in `.obs`.
            correlation_df: DataFrame with correlation results.
            top_n: Number of top correlated genes to plot.
            method: Correlation method to use for selecting top genes.
            target_col: Column name in `adata.obs` for the target variable (used for coloring).
            platform_col: Column name in `adata.obs` indicating the platform/batch.
            figsize: Optional figure size tuple.

        Returns:
            Matplotlib Figure object.
        """
        # --- Initial Checks & Setup (remains mostly the same) ---
        if not SCANPY_AVAILABLE:
            return self._create_error_figure("Scanpy not installed", figsize or (6, 4))
        if adata is None:
            return self._create_error_figure("AnnData object is required", figsize or (6, 4))
        if correlation_df is None or correlation_df.empty:
            return self._create_error_figure("Correlation DataFrame is required", figsize or (6, 4))

        effective_method = self._get_effective_method(method)
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        # --- Validate Columns (remains the same) ---
        if target_col is None:
            if not adata.obs.empty:
                target_col = adata.obs.columns[0]
                logger.warning(f"target_col defaulting to: '{target_col}'")
            else:
                return self._create_error_figure("Cannot determine target_col", figsize or (6, 4))
        elif target_col not in adata.obs.columns:
            return self._create_error_figure(
                f"Target column '{target_col}' not found", figsize or (6, 4)
            )
        if platform_col not in adata.obs.columns:
            return self._create_error_figure(
                f"Platform column '{platform_col}' not found", figsize or (6, 4)
            )

        # --- Prepare Data (remains mostly the same) ---
        try:
            top_genes = self._filter_and_sort_corr_data(correlation_df, effective_method, top_n)
            gene_ids, gene_symbols = (
                top_genes["ensembl_transcript_id"].values,
                self._get_gene_labels(top_genes),
            )
            if len(gene_ids) == 0:
                raise ValueError("No top genes.")
            genes_found_in_adata = [gid for gid in gene_ids if gid in adata.var_names]
            symbols_to_plot = [
                sym
                for gid, sym in zip(gene_ids, gene_symbols, strict=False)
                if gid in genes_found_in_adata
            ]
            if not genes_found_in_adata:
                raise ValueError("None of the top genes found in AnnData.")
            actual_top_n = len(genes_found_in_adata)
            ncols = min(actual_top_n, 3)
            nrows = (actual_top_n + ncols - 1) // ncols
            fig_size = figsize or self.viz_config.get("default_figsize", (ncols * 4.5, nrows * 4.5))
            target_values = pd.to_numeric(adata.obs[target_col], errors="coerce").values
            target_min, target_max = np.nanmin(target_values), np.nanmax(target_values)
            norm = (
                Normalize(vmin=target_min, vmax=target_max)
                if pd.notna(target_min) and pd.notna(target_max) and target_max > target_min
                else None
            )
            if norm is None:
                logger.warning(f"Target '{target_col}' has no range for color mapping.")
        except (ValueError, KeyError) as e:
            logger.error(f"Platform plot data prep failed: {e}")
            return self._create_error_figure(str(e), figsize or (6, 4))

        # --- Plotting Setup (remains the same) ---
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=fig_size, dpi=fig_dpi, squeeze=False
        )
        axes = axes.flatten()
        platform_order = sorted(adata.obs[platform_col].dropna().unique())
        qual_palette = sns.color_palette(PALETTE_QUALITATIVE, n_colors=len(platform_order))
        platform_color_map = dict(zip(platform_order, qual_palette, strict=False))
        cmap_points = plt.get_cmap(CMAP_SEQUENTIAL)
        plot_count = 0

        # --- Plot Each Gene (Modified part) ---
        for gid, gsymbol in zip(genes_found_in_adata, symbols_to_plot, strict=False):
            if plot_count >= len(axes):
                break
            ax = axes[plot_count]
            try:
                expr_vector = adata[:, gid].X
                if hasattr(expr_vector, "toarray"):
                    expr_vector = expr_vector.toarray()
                expr_vector = np.asarray(expr_vector).flatten()
            except Exception as e:
                logger.error(f"Failed to extract expression for {gsymbol} ({gid}): {e}")
                ax.text(
                    0.5, 0.5, f"{gsymbol}\n(Expr. Error)", ha="center", va="center", color="red"
                )
                sns.despine(ax=ax, left=True, bottom=True)
                ax.set_xticks([])
                ax.set_yticks([])
                plot_count += 1
                continue

            # Create DataFrame with unfiltered indices initially
            plot_df_unfiltered = pd.DataFrame(
                {
                    "Expression": expr_vector,
                    platform_col: adata.obs[platform_col].values,
                    target_col: target_values,
                },
                index=adata.obs_names,  # Keep original index
            )

            # Filter rows with NaNs in essential columns AFTER creating the DF
            essential_cols = ["Expression", platform_col, target_col]
            valid_rows_mask = plot_df_unfiltered[essential_cols].notna().all(axis=1)
            plot_df = plot_df_unfiltered[valid_rows_mask]  # This is the DF used for plotting

            if plot_df.empty or plot_df[platform_col].nunique() == 0:
                logger.warning(f"No valid data points for {gsymbol} after dropping NaNs.")
                ax.text(0.5, 0.5, f"{gsymbol}\n(No valid data)", ha="center", va="center")
                sns.despine(ax=ax, left=True, bottom=True)
                ax.set_xticks([])
                ax.set_yticks([])
                plot_count += 1
                continue

            current_platform_order = [
                p for p in platform_order if p in plot_df[platform_col].unique()
            ]
            if not current_platform_order:
                logger.warning(f"No platforms left for {gsymbol} after filtering.")
                ax.text(0.5, 0.5, f"{gsymbol}\n(No platforms)", ha="center", va="center")
                sns.despine(ax=ax, left=True, bottom=True)
                ax.set_xticks([])
                ax.set_yticks([])
                plot_count += 1
                continue

            # Draw violin plot (remains the same)
            try:
                sns.violinplot(
                    x=platform_col,
                    y="Expression",
                    data=plot_df,
                    order=current_platform_order,
                    palette=platform_color_map,
                    ax=ax,
                    inner="quartile",
                    linewidth=1.5,
                    saturation=0.7,
                )
            except Exception as e:
                logger.error(f"Violinplot failed for {gsymbol}: {e}")
                ax.text(
                    0.5, 0.5, f"{gsymbol}\n(Violin Error)", ha="center", va="center", color="red"
                )
                sns.despine(ax=ax, left=True, bottom=True)
                ax.set_xticks([])
                ax.set_yticks([])
                plot_count += 1
                continue

            # FIX: Calculate swarmplot colors based *only* on the filtered plot_df's target values
            # 'plot_df[target_col]' now correctly corresponds to the data points being plotted
            point_colors = cmap_points(norm(plot_df[target_col])) if norm else "black"

            # Draw swarm plot
            try:
                # Use the filtered plot_df directly
                sns.swarmplot(
                    x=platform_col,
                    y="Expression",
                    data=plot_df,  # Use the filtered DataFrame
                    order=current_platform_order,
                    ax=ax,
                    size=max(1.5, 4 - ncols),
                    alpha=0.7,
                    c=point_colors,  # Pass the correctly sized color array
                    edgecolor="none",
                    legend=False,
                    warn_thresh=0.1,
                )
            except Exception as e:
                # Warning is sufficient as violin plot still shows data
                logger.warning(f"Swarmplot failed or had issues for {gsymbol}: {e}")

            # Add annotations (remains the same)
            try:
                gene_info = top_genes[top_genes["ensembl_transcript_id"] == gid].iloc[0]
                corr, pval = gene_info["Correlation_Coefficient"], gene_info["Correlation_P_Value"]
                pval_str = (
                    f"p={pval:.1e}"
                    if pd.notna(pval) and pval < 0.001
                    else f"p={pval:.3f}"
                    if pd.notna(pval)
                    else "p=N/A"
                )
                corr_str = f"r = {corr:.3f}" if pd.notna(corr) else "r=N/A"
                ax.text(
                    0.03,
                    0.97,
                    f"{corr_str}\n{pval_str}",
                    transform=ax.transAxes,
                    fontsize=max(6, FONT_SIZE_ANNOTATION - 1),
                    va="top",
                    ha="left",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "fc": "white",
                        "alpha": 0.7,
                        "ec": "lightgrey",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to add annotation for {gsymbol}: {e}")

            ax.set_title(gsymbol)
            ax.set_xlabel("")
            ax.set_ylabel("Expression")
            ax.tick_params(axis="x", rotation=30)
            sns.despine(ax=ax)
            plot_count += 1

        # --- Final Touches (remains the same) ---
        for k in range(plot_count, len(axes)):
            fig.delaxes(axes[k])
        if norm and plot_count > 0:
            try:
                fig.subplots_adjust(right=0.88)
                cax = fig.add_axes((0.91, 0.15, 0.02, 0.7))
                sm = plt.cm.ScalarMappable(cmap=cmap_points, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
                cbar.set_label(f"{target_col} Value", size=FONT_SIZE_LABEL)
                cbar.outline.set_visible(False)
            except Exception as e:
                logger.error(f"Failed to add colorbar: {e}")
        if plot_count > 0:
            fig.suptitle(
                f"Top {plot_count} Gene Expression by {platform_col} ({effective_method})", y=1.0
            )
        try:
            fig.tight_layout(rect=(0, 0, 0.9, 1))
        except ValueError:
            pass

        self.figures["expression_by_platform"] = fig
        return fig

    def plot_correlation_network(
        self,
        correlation_df: pd.DataFrame,
        top_n: int = 15,
        method: str | None = None,
        min_edge_weight_abs: float = 0.5,
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Creates a gene correlation network visualization."""
        fig_size = figsize or self.viz_config.get("default_figsize", (10, 10))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        if not NETWORKX_AVAILABLE:
            msg = "NetworkX is required for network plot."
            logger.error(msg)
            return self._create_error_figure("NetworkX not installed", fig_size)

        effective_method = self._get_effective_method(method)

        try:
            top_genes = self._filter_and_sort_corr_data(correlation_df, effective_method, top_n)
            gene_ids, gene_symbols = (
                top_genes["ensembl_transcript_id"].values,
                self._get_gene_labels(top_genes),
            )
            n_genes = len(gene_symbols)
            if n_genes == 0:
                raise ValueError("No genes available after filtering.")

            corr_coeffs = pd.to_numeric(top_genes["Correlation_Coefficient"], errors="coerce")
            gene_corr_map = dict(zip(gene_ids, corr_coeffs, strict=False))

            # Calculate co-correlation matrix
            corr_matrix = np.full((n_genes, n_genes), np.nan, dtype=np.float64)
            for i in range(n_genes):
                corr1 = gene_corr_map.get(gene_ids[i], np.nan)
                for j in range(n_genes):
                    corr2 = gene_corr_map.get(gene_ids[j], np.nan)
                    corr_matrix[i, j] = (
                        1.0
                        if i == j
                        else (corr1 * corr2 if pd.notna(corr1) and pd.notna(corr2) else np.nan)
                    )

        except (ValueError, KeyError) as e:
            logger.error(f"Network plot data preparation failed: {e}")
            return self._create_error_figure(str(e), fig_size)

        # Build Graph
        G = nx.Graph()
        for i, symbol in enumerate(gene_symbols):
            corr_target = gene_corr_map.get(gene_ids[i], 0.0)
            # Ensure corr_target is a float for size calculation
            corr_target_float = float(corr_target) if pd.notna(corr_target) else 0.0
            G.add_node(
                symbol,
                size=max(100, 1500 * abs(corr_target_float)),
                color=NODE_COLOR_POS if corr_target_float >= 0 else NODE_COLOR_NEG,
                label=symbol,
                corr_target=corr_target_float,
            )

        num_edges = 0
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                weight = corr_matrix[i, j]
                if pd.notna(weight) and abs(weight) >= min_edge_weight_abs:
                    G.add_edge(
                        gene_symbols[i],
                        gene_symbols[j],
                        weight=abs(weight),
                        color=EDGE_COLOR_POS if weight > 0 else EDGE_COLOR_NEG,
                        linestyle="solid" if weight > 0 else "dashed",  # Store linestyle
                    )
                    num_edges += 1
        if num_edges == 0:
            logger.warning(
                f"No edges met the minimum absolute weight threshold ({min_edge_weight_abs})."
            )

        # Plotting
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        if not G.nodes():
            return self._create_error_figure("No nodes to plot in network.", fig_size)

        # Position nodes using spring layout
        try:
            pos = nx.spring_layout(
                G, k=0.6 / np.sqrt(max(1, len(G.nodes()))), iterations=100, seed=42
            )
        except Exception as layout_err:
            logger.warning(f"Spring layout failed ({layout_err}), using random layout.")
            pos = nx.random_layout(G, seed=42)

        node_sizes = [G.nodes[n]["size"] for n in G.nodes()]
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]

        # NetworkX 2.x+ supports list inputs for these parameters contrary to type hints
        # The linter is incorrect here - these parameters can accept lists in NetworkX
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax
        )

        if G.edges():
            edge_colors = [G.edges[e]["color"] for e in G.edges()]
            edge_widths = [G.edges[e]["weight"] * 3 for e in G.edges()]
            edge_styles = [G.edges[e]["linestyle"] for e in G.edges()]

            # NetworkX 2.x+ supports list inputs for these parameters
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_widths,
                edge_color=edge_colors,
                style=edge_styles,
                alpha=0.6,
                ax=ax,
            )

        # Use label size from global config
        nx.draw_networkx_labels(G, pos, font_size=FONT_SIZE_ANNOTATION, font_weight="bold", ax=ax)
        ax.set_title(
            f"Gene Correlation Network ({effective_method}, Top {n_genes}, |Co-corr| ≥ {min_edge_weight_abs})"
        )
        plt.axis("off")

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Pos. Corr w/ Target",
                markerfacecolor=NODE_COLOR_POS,
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Neg. Corr w/ Target",
                markerfacecolor=NODE_COLOR_NEG,
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                color=EDGE_COLOR_POS,
                lw=2,
                linestyle="solid",
                label="Pos. Co-correlation",
            ),
            Line2D(
                [0],
                [0],
                color=EDGE_COLOR_NEG,
                lw=2,
                linestyle="dashed",
                label="Neg. Co-correlation",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.1, 1.0),
            fontsize=FONT_SIZE_LEGEND,
            frameon=False,
        )
        try:
            fig.tight_layout()
        except ValueError:
            pass
        self.figures["correlation_network"] = fig
        return fig

    def plot_bootstrap_confidence(
        self,
        correlation_df: pd.DataFrame,
        top_n: int = 20,
        method: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Plots correlation coefficients with bootstrap confidence intervals."""
        effective_method = self._get_effective_method(method)
        plot_name = f"Bootstrap Confidence ({effective_method})"
        fig_size = figsize or self.viz_config.get("default_figsize", (8, 10))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        ci_lower_col = f"Correlation_CI_Lower_{effective_method}"
        ci_upper_col = f"Correlation_CI_Upper_{effective_method}"
        point_estimate_col = f"Correlation_Coefficient_{effective_method}_MeanBoot"
        fallback_point_estimate_col = "Correlation_Coefficient"

        actual_point_estimate_col = point_estimate_col
        if point_estimate_col not in correlation_df.columns:
            if fallback_point_estimate_col in correlation_df.columns:
                actual_point_estimate_col = fallback_point_estimate_col
                logger.warning(
                    f"'{point_estimate_col}' not found, using '{fallback_point_estimate_col}' as point estimate."
                )
            else:
                actual_point_estimate_col = point_estimate_col

        required_cols = [
            actual_point_estimate_col,
            ci_lower_col,
            ci_upper_col,
            "ensembl_transcript_id",
        ]

        try:
            method_df = correlation_df[
                correlation_df["Correlation_Type"] == effective_method
            ].copy()
            if not self._check_required_columns(method_df, required_cols, plot_name):
                raise ValueError(
                    f"Missing required columns for bootstrap plot. Checked: {required_cols}"
                )

            method_df["abs_point_estimate"] = method_df[actual_point_estimate_col].abs()
            plot_data = method_df.sort_values(
                "abs_point_estimate", ascending=False, na_position="last"
            ).head(top_n)
            plot_data = plot_data.sort_values(actual_point_estimate_col, ascending=True)

            if plot_data.empty:
                raise ValueError("No data available after filtering/sorting for bootstrap plot.")

            gene_labels = self._get_gene_labels(plot_data)
            point_estimates = plot_data[actual_point_estimate_col]
            ci_lower = plot_data[ci_lower_col]
            ci_upper = plot_data[ci_upper_col]
            lower_error = (point_estimates - ci_lower).fillna(0)
            upper_error = (ci_upper - point_estimates).fillna(0)
            # Ensure errors is a 2xN array for individual plotting
            errors_array = np.array([lower_error.values, upper_error.values])

        except (ValueError, KeyError) as e:
            logger.error(f"{plot_name} data preparation failed: {e}")
            return self._create_error_figure(str(e), fig_size)

        # --- Plotting (FIXED) ---
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

        # Loop through each gene/point to plot error bars individually with correct color
        for i, (y_pos, estimate, label) in enumerate(
            zip(range(len(gene_labels)), point_estimates, gene_labels, strict=False)
        ):
            point_color = NODE_COLOR_POS if estimate >= 0 else NODE_COLOR_NEG
            # Extract error for this specific point
            # Need to handle potential shape issues if errors_array is not 2xN
            current_error = (
                [[errors_array[0, i]], [errors_array[1, i]]]
                if errors_array.shape[0] == 2
                else [[0], [0]]
            )

            ax.errorbar(
                estimate,  # Single x value
                y_pos,  # Single y value
                xerr=current_error,  # Error for this point (shape [[lower],[upper]])
                fmt="none",  # No marker here, just the error bar
                ecolor=point_color,  # Single color for this error bar
                elinewidth=1.5,
                capsize=4,
                zorder=2,
            )

        # Plot scatter points on top (can still do this vectorized)
        scatter_colors = [NODE_COLOR_POS if est >= 0 else NODE_COLOR_NEG for est in point_estimates]
        ax.scatter(
            point_estimates,
            range(len(gene_labels)),
            color=scatter_colors,
            s=40,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_yticks(range(len(gene_labels)))
        ax.set_yticklabels(gene_labels)
        ax.set_xlabel(f"{effective_method} Correlation Coefficient (Estimate & 95% CI)")
        ax.set_ylabel("Gene")
        ax.set_title(
            f"Top {len(gene_labels)} Correlated Genes ({effective_method}) with Bootstrap CI"
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.grid(axis="x", linestyle=":", linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()

        self.figures["bootstrap_confidence"] = fig
        return fig

    def plot_rank_stability_heatmap(
        self,
        correlation_df: pd.DataFrame,
        top_n_genes: int = 30,
        top_n_ranks: int = 50,
        method: str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Plots a heatmap showing the probability of genes appearing in top ranks."""
        effective_method = self._get_effective_method(method)
        plot_name = f"Rank Stability Heatmap ({effective_method})"
        # Adjusted default size for potentially many genes
        fig_size = figsize or self.viz_config.get("default_figsize", (8, max(6, top_n_genes * 0.3)))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        stability_col = f"Rank_Stability_{effective_method}_Perc"
        required_cols = [stability_col, "ensembl_transcript_id"]

        try:
            method_df = correlation_df[
                correlation_df["Correlation_Type"] == effective_method
            ].copy()
            if not self._check_required_columns(method_df, required_cols, plot_name):
                msg = f"Missing required columns: {required_cols}"
                raise ValueError(msg)
            plot_data = method_df.sort_values(
                stability_col, ascending=False, na_position="last"
            ).head(top_n_genes)
            if plot_data.empty:
                msg = "No data for rank stability plot."
                raise ValueError(msg)
            heatmap_values = plot_data[[stability_col]].copy()
            heatmap_values.columns = ["Stability (%)"]
            gene_labels = self._get_gene_labels(plot_data)
            heatmap_values.index = gene_labels

        except (ValueError, KeyError) as e:
            logger.error(f"{plot_name} data prep failed: {e}")
            return self._create_error_figure(str(e), fig_size)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
        sns.heatmap(
            heatmap_values,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            linecolor="lightgrey",
            cbar_kws={"label": f"Probability of being in Top {top_n_ranks} (%)"},
            ax=ax,
            vmin=0,
            vmax=100,
        )

        ax.set_title(f"Rank Stability for Top {len(gene_labels)} Genes ({effective_method})")
        ax.set_xlabel("Overall Bootstrap Stability")
        ax.set_ylabel("Gene")
        # FIX: Adjust y-tick label size based on number of genes
        ytick_fontsize = max(6, FONT_SIZE_TICK - int(len(gene_labels) / 10))
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        fig.tight_layout()
        self.figures["rank_stability_heatmap"] = fig
        return fig

    def plot_panel_comparison(
        self,
        panel_results: dict[str, dict[str, Any]],
        figsize: tuple[int, int] | None = None,
    ) -> Figure:
        """Compares performance metrics across different marker panels."""
        plot_name = "Marker Panel Comparison"
        fig_size = figsize or self.viz_config.get("default_figsize", (12, 7))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)
        plot_data = []
        for p_type, panels in panel_results.items():
            for panel_name, metrics in panels.items():
                if metrics:
                    plot_data.append(
                        {
                            "Panel Name": panel_name.replace(f"panel_{p_type}_", ""),
                            "Panel Type": p_type.replace("_", " ").title(),
                            "R2 (LOOCV)": metrics.get("r2_loocv", np.nan),
                            "RMSE (LOOCV)": metrics.get("rmse_loocv", np.nan),
                            "Panel Size": metrics.get("size", 0),
                        }
                    )
        if not plot_data:
            return self._create_error_figure("No panel data to plot", fig_size)
        df = pd.DataFrame(plot_data)

        # --- Plotting (FIXED Legend and Palette) ---
        fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=fig_dpi, sharey=True)
        # Use a consistent, nice palette (e.g., viridis or mako as used before)
        palette = sns.color_palette("viridis", n_colors=df["Panel Type"].nunique())

        # Plot R-squared
        sns.barplot(
            data=df,
            y="Panel Name",
            x="R2 (LOOCV)",
            hue="Panel Type",
            ax=axes[0],
            palette=palette,
            orient="h",
        )
        axes[0].set_title("Panel Predictive Performance (R²)")  # Fontsize from rcParams
        axes[0].set_xlabel("Leave-One-Out R²")  # Fontsize from rcParams
        axes[0].set_ylabel("Panel Configuration")  # Fontsize from rcParams
        axes[0].grid(axis="x", linestyle="--", alpha=0.6)
        # FIX: Move legend outside
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend_.remove()  # Remove the default legend inside

        # Plot RMSE
        sns.barplot(
            data=df,
            y="Panel Name",
            x="RMSE (LOOCV)",
            hue="Panel Type",
            ax=axes[1],
            palette=palette,
            orient="h",
            legend=False,  # Keep legend=False
        )
        axes[1].set_title("Panel Prediction Error (RMSE)")  # Fontsize from rcParams
        axes[1].set_xlabel("Leave-One-Out RMSE")  # Fontsize from rcParams
        axes[1].set_ylabel("")  # Remove y-label
        axes[1].grid(axis="x", linestyle="--", alpha=0.6)

        # Add the shared legend outside
        fig.legend(
            handles, labels, title="Objective", loc="center right", bbox_to_anchor=(1.08, 0.5)
        )

        fig.suptitle("Marker Panel Performance Comparison", y=1.02, fontsize=FONT_SIZE_FIGURE_TITLE)
        sns.despine(fig=fig)
        fig.tight_layout(rect=(0, 0, 0.9, 1))

        self.figures["panel_comparison"] = fig
        return fig

    def plot_information_gain(
        self,
        panel_results: dict[str, dict[str, Any]],
        figsize: tuple[int, int] | None = None,
        # NOTE: This plot requires calculating information gain during selection,
        # which the current _greedy_forward_selection doesn't explicitly return!!
        # This implementation will plot R^2 vs panel size as a proxy.
    ) -> Figure:
        """Plots predictive performance (R^2) as markers are added."""
        plot_name = "Information Gain (R² vs Panel Size)"
        fig_size = figsize or self.viz_config.get("default_figsize", (8, 5))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)

        plot_data = []
        for p_type, panels in panel_results.items():
            for panel_name, metrics in panels.items():
                if metrics and metrics.get("size", 0) > 0:
                    plot_data.append(
                        {
                            "Panel Type": p_type.replace("_", " ").title(),
                            "Panel Size": metrics["size"],
                            "R2 (LOOCV)": metrics.get("r2_loocv", np.nan),
                        }
                    )

        if not plot_data:
            logger.warning("No panel data available for information gain plot.")
            return self._create_error_figure("No panel data", fig_size)

        df = pd.DataFrame(plot_data).dropna(subset=["R2 (LOOCV)"])
        df = df.sort_values(by=["Panel Type", "Panel Size"])

        # Plotting
        fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

        sns.lineplot(
            data=df,
            x="Panel Size",
            y="R2 (LOOCV)",
            hue="Panel Type",
            marker="o",
            ax=ax,
            palette="tab10",
        )

        ax.set_title("Predictive Performance vs. Panel Size")
        ax.set_xlabel("Number of Markers in Panel")
        ax.set_ylabel("Leave-One-Out R²")
        ax.legend(title="Panel Objective")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.6)
        sns.despine(ax=ax)
        fig.tight_layout()

        self.figures["information_gain"] = fig
        return fig

    def plot_platform_effect_heatmap(
        self,
        expression_df: pd.DataFrame,  # Genes x Samples (can be symbol or ID indexed)
        manifest_df: pd.DataFrame,
        residual_effects_df: pd.DataFrame | None = None,  # Indexed by Gene ID
        top_n_genes: int = 50,
        significance_col: str = "kruskal_pvalue_adj",
        figsize: tuple[int, int] | None = None,
        cluster_genes: bool = True,
        gene_id_col_mapping: dict[str, str] | None = None,  # Optional: Precomputed ID->Symbol map
    ) -> Figure:
        """Plots a heatmap showing expression patterns across platforms for affected genes."""
        logger = logging.getLogger(__name__)

        plot_name = "Platform Effect Heatmap"
        logger.info(f"DEBUG: Starting {plot_name}")

        fig_size = figsize or self.viz_config.get("default_figsize", (10, 14))
        fig_dpi = self.viz_config.get("default_dpi", DEFAULT_DPI)
        default_color = "#CCCCCC"  # Default color for missing platform info (light grey)

        # --- Platform Mapping ---
        logger.info("DEBUG: Getting platform map")
        platform_map = AdvancedBatchCorrection()._get_platform_map(manifest_df)
        if not platform_map:
            logger.error("DEBUG: Cannot map platforms")
            return self._create_error_figure("Cannot map platforms", fig_size)
        platforms = sorted(set(platform_map.values()) - {"Unknown"})
        if len(platforms) < 2:
            logger.error("DEBUG: Need >= 2 platforms")
            return self._create_error_figure("Need >= 2 platforms", fig_size)
        logger.info(f"DEBUG: Found platforms: {platforms}")

        # --- Align Expression Data ---
        logger.info("DEBUG: Aligning expression data with platform info")
        common_samples = sorted(set(expression_df.columns) & set(platform_map.keys()))
        if not common_samples:
            logger.error("DEBUG: No common samples between expression and manifest")
            return self._create_error_figure(
                "No common samples between expression and manifest", fig_size
            )
        expr_aligned = expression_df[common_samples]
        # platforms_aligned is a Series: Index=SampleID, Value=PlatformName
        platforms_aligned = pd.Series(
            [platform_map.get(s) for s in common_samples], index=common_samples
        )
        logger.info(f"DEBUG: Aligned {len(common_samples)} samples with platform info")

        # --- Select Genes to Plot ---
        logger.info("DEBUG: Selecting genes to plot")
        genes_to_plot = []
        title_suffix = ""

        # Debug residual_effects_df details
        if residual_effects_df is not None:
            logger.info(f"DEBUG: residual_effects_df type: {type(residual_effects_df)}")
            logger.info(f"DEBUG: residual_effects_df columns: {residual_effects_df.columns}")
            logger.info(f"DEBUG: residual_effects_df index type: {type(residual_effects_df.index)}")
            logger.info(
                f"DEBUG: residual_effects_df index sample: {list(residual_effects_df.index[:5]) if not residual_effects_df.empty else []}"
            )
            if significance_col in residual_effects_df.columns:
                logger.info(
                    f"DEBUG: significance_col values sample: {residual_effects_df[significance_col].head()}"
                )
                logger.info(
                    f"DEBUG: significance_col null values: {residual_effects_df[significance_col].isnull().sum()}"
                )
        else:
            logger.info("DEBUG: residual_effects_df is None")

        # Debug expression_df details
        logger.info(f"DEBUG: expression_df index type: {type(expression_df.index)}")
        logger.info(
            f"DEBUG: expression_df index sample: {list(expression_df.index[:5]) if not expression_df.empty else []}"
        )

        if (
            residual_effects_df is not None
            and significance_col in residual_effects_df.columns
            and not residual_effects_df[significance_col].isnull().all()
        ):
            logger.info("DEBUG: Using significance filter to select genes")
            try:
                # Fix for ID format mismatch (ENSCGRG vs ENSCGRT)
                # Create mapping dictionaries to handle different ID formats
                expr_ids = set(expression_df.index)
                effect_ids = set(residual_effects_df.index)

                # Log the prefix patterns to diagnose the issue
                expr_prefixes = set(
                    str(idx).split("00")[0] for idx in expr_ids if str(idx).find("00") > 0
                )
                effect_prefixes = set(
                    str(idx).split("00")[0] for idx in effect_ids if str(idx).find("00") > 0
                )

                logger.info(f"DEBUG: Expression ID prefixes: {expr_prefixes}")
                logger.info(f"DEBUG: Effect ID prefixes: {effect_prefixes}")

                # Create normalized versions of IDs for matching (strip prefixes if needed)
                normalized_expr_map = {}  # Original ID -> Normalized
                normalized_effect_map = {}  # Original ID -> Normalized

                for idx in expr_ids:
                    # Extract numeric portion after prefix (e.g. 00001000001 from ENSCGRG00001000001)
                    norm_id = str(idx)
                    for prefix in expr_prefixes:
                        if norm_id.startswith(prefix):
                            norm_id = norm_id[len(prefix) :]
                            break
                    normalized_expr_map[norm_id] = idx

                for idx in effect_ids:
                    norm_id = str(idx)
                    for prefix in effect_prefixes:
                        if norm_id.startswith(prefix):
                            norm_id = norm_id[len(prefix) :]
                            break
                    normalized_effect_map[norm_id] = idx

                # Find common normalized IDs
                common_norm_ids = set(normalized_expr_map.keys()) & set(
                    normalized_effect_map.keys()
                )
                logger.info(
                    f"DEBUG: Found {len(common_norm_ids)} matching genes after normalization"
                )

                # Map these back to original IDs in both datasets
                expr_matched_ids = [normalized_expr_map[norm_id] for norm_id in common_norm_ids]
                effect_matched_ids = [normalized_effect_map[norm_id] for norm_id in common_norm_ids]

                # Create a mapping from effect IDs to expression IDs
                id_mapping = dict(zip(effect_matched_ids, expr_matched_ids, strict=False))
                logger.info(f"DEBUG: Created ID mapping with {len(id_mapping)} entries")

                # Now filter residual effects data and map IDs back to expression data format
                if common_norm_ids:
                    effect_filtered = residual_effects_df.loc[effect_matched_ids].sort_values(
                        significance_col, ascending=True
                    )
                    # Map IDs and select top N genes
                    top_effect_ids = effect_filtered.head(top_n_genes).index
                    genes_to_plot = [id_mapping[eid] for eid in top_effect_ids]
                    logger.info(
                        f"DEBUG: Selected {len(genes_to_plot)} genes using significance filter"
                    )
                    title_suffix = f" (Top {len(genes_to_plot)} by {significance_col})"
                else:
                    logger.warning("DEBUG: No matching genes after ID normalization")
            except Exception as e:
                logger.exception(f"DEBUG: Error in gene selection with significance: {e}")
                genes_to_plot = []
        else:
            logger.warning(
                f"DEBUG: Residual effects data/column '{significance_col}' invalid. Plotting top variable genes."
            )

        if not genes_to_plot:  # Fallback to top variable
            logger.info("DEBUG: Falling back to top variable genes")
            try:
                gene_vars = expr_aligned.var(axis=1).sort_values(ascending=False)
                genes_to_plot = gene_vars.head(top_n_genes).index.tolist()
                logger.info(f"DEBUG: Selected {len(genes_to_plot)} top variable genes")
                title_suffix = f" (Top {top_n_genes} Most Variable)"
            except Exception as e:
                logger.exception(f"DEBUG: Error in fallback gene selection: {e}")

        if not genes_to_plot:
            logger.error("DEBUG: No genes selected to plot")
            return self._create_error_figure("No genes selected to plot", fig_size)

        # --- Extract Data for Selected Genes ---
        logger.info("DEBUG: Extracting data for selected genes")
        try:
            plot_data_unscaled = expr_aligned.loc[genes_to_plot]
            logger.info(f"DEBUG: Unscaled data shape: {plot_data_unscaled.shape}")
        except Exception as e:
            logger.exception(f"DEBUG: Error extracting gene data: {e}")
            return self._create_error_figure(f"Error extracting gene data: {e}", fig_size)

        # --- Standardize Expression ---
        logger.info("DEBUG: Standardizing expression data")
        scaler = StandardScaler()
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                plot_data_scaled_np = scaler.fit_transform(plot_data_unscaled.T).T
            plot_data_scaled_np = np.nan_to_num(plot_data_scaled_np)
            logger.info("DEBUG: Standardization successful")
        except ValueError as scale_err:
            logger.exception(f"DEBUG: StandardScaler failed: {scale_err}")
            return self._create_error_figure(f"Scaling error: {scale_err}", fig_size)

        try:
            plot_data_scaled = pd.DataFrame(
                plot_data_scaled_np,
                index=plot_data_unscaled.index,
                columns=plot_data_unscaled.columns,
            )
            logger.info("DEBUG: Scaled DataFrame created successfully")
        except Exception as e:
            logger.error(f"DEBUG: Error creating scaled DataFrame: {e}")
            return self._create_error_figure(f"Error creating scaled DataFrame: {e}", fig_size)

        # Order Columns by Platform
        logger.info("DEBUG: Ordering columns by platform")
        try:
            ordered_samples = []
            for p in platforms:
                platform_samples = platforms_aligned[platforms_aligned == p].index.tolist()
                ordered_samples.extend(platform_samples)

            # Filter to samples in the data
            col_order = [c for c in ordered_samples if c in plot_data_scaled.columns]
            logger.info(f"DEBUG: Ordered {len(col_order)} columns by platform")
        except Exception as e:
            logger.exception(f"DEBUG: Error ordering columns: {e}")
            return self._create_error_figure(f"Error ordering columns: {e}", fig_size)

        if not col_order:
            logger.error("DEBUG: No valid columns after platform ordering")
            return self._create_error_figure("No valid columns after platform ordering", fig_size)

        try:
            plot_data_final = plot_data_scaled[col_order]
            logger.info(f"DEBUG: Final plot data shape: {plot_data_final.shape}")
        except Exception as e:
            logger.exception(f"DEBUG: Error creating final plot data: {e}")
            return self._create_error_figure(f"Error creating final plot data: {e}", fig_size)

        # --- Prepare Row Labels (Gene Symbols/IDs) ---
        logger.info("DEBUG: Preparing row labels")
        try:
            # Fix: Create row labels from index in a safer way
            if gene_id_col_mapping:
                logger.info("DEBUG: Using provided gene ID to symbol mapping")
                # Create a list of labels, with fallback to original ID if not in mapping
                row_labels_map = pd.Series(
                    [gene_id_col_mapping.get(idx, str(idx)) for idx in plot_data_final.index]
                )
                logger.info("DEBUG: Created mapped labels using dictionary lookup with fallback")
            else:
                logger.info("DEBUG: No mapping provided, using index values as labels")
                # Convert index directly to Series for consistency
                row_labels_map = pd.Series(plot_data_final.index.astype(str))

            logger.info(f"DEBUG: Created row labels, sample: {list(row_labels_map.head())}")
        except Exception as e:
            logger.error(f"DEBUG: Error creating row labels: {e}")
            return self._create_error_figure(f"Error creating row labels: {e}", fig_size)

        # --- Platform Color Bar ---
        logger.info("DEBUG: Creating platform color bar")
        try:
            # Create color mapping from platform names to colors
            color_palette = sns.color_palette("tab10", len(platforms))
            platform_colors_map = {
                p: color for p, color in zip(platforms, color_palette, strict=False)
            }

            # Convert default color from string to RGB tuple
            # Using matplotlib's to_rgb function to convert hex to RGB tuple
            default_color_rgb = to_rgb(default_color)

            # Create a list of colors for each column instead of an array
            column_colors = []
            for sample in plot_data_final.columns:
                platform = platforms_aligned.get(sample, "Unknown")
                if platform in platform_colors_map:
                    column_colors.append(platform_colors_map[platform])
                else:
                    column_colors.append(default_color_rgb)

            # Create a DataFrame with proper structure for seaborn
            column_colors_df = pd.DataFrame(
                {"Platform": column_colors}, index=plot_data_final.columns
            )

            logger.info(f"DEBUG: Created column colors DataFrame for {len(column_colors)} columns")
        except Exception as e:
            logger.error(f"DEBUG: Error creating color mapping: {e}")
            return self._create_error_figure(f"Error creating color mapping: {e}", fig_size)

        # --- Plotting ---
        logger.info("DEBUG: Creating clustermap")
        try:
            # Check if we have enough genes to do clustering
            can_cluster_rows = cluster_genes and plot_data_final.shape[0] > 1
            if not can_cluster_rows and cluster_genes:
                logger.info(
                    f"DEBUG: Disabling row clustering because there are only {plot_data_final.shape[0]} genes"
                )

            # If we have very few genes, adjust the figure size to be more appropriate
            if plot_data_final.shape[0] < 5:
                adjusted_height = max(5, plot_data_final.shape[0] * 1.0)  # At least 5 inches height
                fig_size = (fig_size[0], adjusted_height)
                logger.info(
                    f"DEBUG: Adjusted figure height to {adjusted_height} for better display of few genes"
                )

            g = sns.clustermap(
                plot_data_final,
                cmap=CMAP_DIVERGING,
                center=0,
                # Pass the color dataframe instead of an array
                col_colors=column_colors_df,
                col_cluster=False,
                row_cluster=can_cluster_rows,
                figsize=fig_size,
                linewidths=0.1,
                linecolor="lightgrey",
                dendrogram_ratio=(0.2, 0.05) if can_cluster_rows else (0.0, 0.05),
                cbar_pos=None,  # Hide the color bar for expression values
                yticklabels=True,  # Enable labels since we may have few genes
            )
            logger.info("DEBUG: Clustermap created successfully")

            g.fig.suptitle(f"Gene Expression Across Platforms{title_suffix}", y=1.02)
            g.ax_heatmap.set_xlabel("Samples (Ordered by Platform)")
            g.ax_heatmap.set_ylabel("Genes")
            g.ax_heatmap.set_xticks([])  # Remove sample ticks
            logger.info("DEBUG: Basic plot formatting complete")

            # Set yticklabels AFTER clustermap using the potentially mapped symbols/IDs
            # Only needed if yticklabels=False in the clustermap call, but we're now using yticklabels=True
            # This section is kept for posterity but modified to check first if we need to set labels
            if hasattr(g, "ax_heatmap") and len(row_labels_map) > 0:
                try:
                    logger.info("DEBUG: Checking if custom y-tick labels are needed")

                    # If labels aren't visible or are not what we want, set them
                    current_labels = g.ax_heatmap.get_yticklabels()
                    if not current_labels or len(current_labels) != len(row_labels_map):
                        logger.info(
                            f"DEBUG: Setting custom y-tick labels, length: {len(row_labels_map)}"
                        )

                        if (
                            can_cluster_rows
                            and hasattr(g, "dendrogram_row")
                            and g.dendrogram_row is not None
                        ):
                            logger.info("DEBUG: Using clustered indices for labels")
                            # Fix: Careful handling of reordered indices and Series indexing
                            clustered_indices = g.dendrogram_row.reordered_ind
                            logger.info(
                                f"DEBUG: Got clustered indices, length: {len(clustered_indices)}"
                            )

                            clustered_labels = [row_labels_map.iloc[i] for i in clustered_indices]
                            logger.info("DEBUG: Created clustered labels")
                            final_row_labels = clustered_labels
                        else:
                            logger.info("DEBUG: Using unclustered labels")
                            final_row_labels = row_labels_map.tolist()

                        logger.info(f"DEBUG: Setting yticks, length: {len(final_row_labels)}")
                        g.ax_heatmap.set_yticks(np.arange(len(final_row_labels)) + 0.5)
                        g.ax_heatmap.set_yticklabels(final_row_labels)
                        logger.info("DEBUG: Y-tick labels set successfully")
                except Exception as e:
                    logger.error(f"DEBUG: Error setting y-tick labels: {e}")
            elif not len(row_labels_map):
                logger.warning("DEBUG: No row labels available")
                # yticklabels were already set to False

            ytick_fontsize = max(4, FONT_SIZE_TICK - int(len(genes_to_plot) / 10))
            g.ax_heatmap.tick_params(axis="y", labelsize=ytick_fontsize, rotation=0)
            logger.info("DEBUG: Y-tick formatting complete")

            # Add custom legend for platforms
            logger.info("DEBUG: Adding platform legend")
            handles = [
                Patch(facecolor=color, label=platform)
                for platform, color in platform_colors_map.items()
            ]
            # Place legend relative to the *heatmap axis* to avoid overlap
            g.ax_heatmap.legend(
                handles=handles,
                title="Platform",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=FONT_SIZE_LEGEND,
            )
            logger.info("DEBUG: Platform legend added")

            fig = g.fig
            logger.info("DEBUG: Clustermap plot completed successfully")

        except Exception as e:
            logger.exception(f"DEBUG: Clustermap failed with exception: {e}")
            return self._create_error_figure(f"Heatmap failed: {e}", fig_size)

        self.figures["platform_effect_heatmap"] = fig
        return fig
