# cho_analysis/task2/visualization.py
"""Visualization utilities for Task 2 (sequence feature analysis)."""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from cho_analysis.core.config import get_visualization_config
from cho_analysis.core.visualization_utils import *

logger = logging.getLogger(__name__)


class SequenceFeatureVisualization:
    """Generates visualizations for sequence feature analysis results."""

    def __init__(self):
        self.figures: Dict[str, Figure] = {}
        self.viz_config = get_visualization_config()

        # Apply base style and rcParams, allowing overrides from config
        style = self.viz_config.get("style") or DEFAULT_STYLE
        font_family = self.viz_config.get("font_family") or DEFAULT_FONT_FAMILY
        base_params = BASE_RC_PARAMS.copy()
        base_params["font.family"] = font_family  # Apply override if present

        plt.style.use(style)
        plt.rcParams.update(base_params)
        logger.debug(f"Using plot style: {style}")
        logger.debug(f"Applied base rcParams with font: {font_family}")

    def _check_required_columns(
        self, df: pd.DataFrame, required_cols: list[str], plot_name: str
    ) -> bool:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Cannot generate {plot_name}: Missing columns: {missing_cols}")
            return False
        if df.empty:
            logger.error(f"Cannot generate {plot_name}: Input DataFrame is empty.")
            return False
        return True

    # --- Plotting Methods (Using Constants) ---
    def plot_utr_length_distribution(
        self, feature_df: pd.DataFrame, figsize: Tuple[int, int] | None = None
    ) -> Figure | None:
        """Plots the distribution of 5' and 3' UTR lengths."""
        plot_name = "UTR Length Distribution"
        required_cols = ["UTR5_length", "UTR3_length"]
        if not self._check_required_columns(feature_df, required_cols, plot_name):
            return None
        fig: Figure | None = None
        try:
            fig_size = figsize or self.viz_config.get("default_figsize", (8, 5))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            utr5_lengths = feature_df["UTR5_length"].dropna().loc[lambda x: x > 0]
            utr3_lengths = feature_df["UTR3_length"].dropna().loc[lambda x: x > 0]
            plot_kwargs = {"kde": True, "ax": ax, "stat": "count", "element": "step"}
            plotted_something = False
            if not utr5_lengths.empty:
                sns.histplot(
                    utr5_lengths, color=PALETTE_DISTRIBUTION[0], label="5'UTR", **plot_kwargs
                )
                plotted_something = True
            else:
                logger.warning("No positive 5'UTR lengths found.")
            if not utr3_lengths.empty:
                sns.histplot(
                    utr3_lengths, color=PALETTE_DISTRIBUTION[1], label="3'UTR", **plot_kwargs
                )
                plotted_something = True
            else:
                logger.warning("No positive 3'UTR lengths found.")
            ax.set_title("UTR Length Distribution of Expressed Genes")
            ax.set_xlabel("Length (bp)")
            ax.set_ylabel("Gene Count")
            if plotted_something:
                ax.legend(title="Region")
            sns.despine(ax=ax)
            fig.tight_layout()
            self.figures["utr_length_dist"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig

    def plot_gc_distribution(
        self, feature_df: pd.DataFrame, figsize: Tuple[int, int] | None = None
    ) -> Figure | None:
        """Plots the distribution of GC content for UTRs and CDS GC3."""
        plot_name = "GC Content Distribution"
        required_cols = ["UTR5_GC", "UTR3_GC", "GC3_content"]
        if not self._check_required_columns(feature_df, required_cols, plot_name):
            return None
        fig: Figure | None = None
        try:
            fig_size = figsize or self.viz_config.get("default_figsize", (8, 5))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            plot_kwargs = {"kde": True, "ax": ax, "stat": "count", "element": "step"}
            sns.histplot(
                feature_df["UTR5_GC"].dropna(),
                color=PALETTE_DISTRIBUTION[0],
                label="5'UTR GC",
                **plot_kwargs,
            )
            sns.histplot(
                feature_df["UTR3_GC"].dropna(),
                color=PALETTE_DISTRIBUTION[1],
                label="3'UTR GC",
                **plot_kwargs,
            )
            mean_gc3 = feature_df["GC3_content"].dropna().mean()
            if pd.notna(mean_gc3):
                ax.axvline(
                    mean_gc3,
                    color=PALETTE_DISTRIBUTION[2],
                    linestyle="--",
                    label=f"Avg CDS GC3 ({mean_gc3:.1f}%)",
                )
            else:
                logger.warning("Could not calculate average CDS GC3 content.")
            ax.set_title("GC Content Distribution")
            ax.set_xlabel("GC Content (%)")
            ax.set_ylabel("Gene Count")
            ax.legend(title="Region / Feature")
            sns.despine(ax=ax)
            fig.tight_layout()
            self.figures["gc_content_dist"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig

    def plot_cv_vs_feature(
        self,
        feature_df: pd.DataFrame,
        feature_col: str,
        feature_name: str | None = None,
        figsize: Tuple[int, int] | None = None,
        alpha: float = 0.5,
    ) -> Figure | None:
        """Creates a scatter plot of expression stability (CV) vs. a sequence feature."""
        plot_name = f"CV vs {feature_col}"
        required_cols = ["cv", feature_col]
        if not self._check_required_columns(feature_df, required_cols, plot_name):
            return None
        fig: Figure | None = None
        try:
            plot_data = feature_df[required_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if plot_data.empty:
                logger.error(f"No valid data for {plot_name}")
                return None
            fig_size = figsize or self.viz_config.get("default_figsize", (7, 5))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            sns.scatterplot(
                x="cv", y=feature_col, data=plot_data, alpha=alpha, ax=ax, s=20, edgecolor="none"
            )
            ylabel = feature_name or feature_col.replace("_", " ").title()
            ax.set_title(f"Expression Stability vs {ylabel}")
            ax.set_xlabel("Coefficient of Variation (Lower is More Stable)")
            ax.set_ylabel(ylabel)
            sns.despine(ax=ax)
            fig.tight_layout()
            self.figures[f"cv_vs_{feature_col}"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig

    def plot_regulatory_element_counts(
        self, feature_df: pd.DataFrame, figsize: Tuple[int, int] | None = None
    ) -> Figure | None:
        """Creates a bar plot showing counts of regulatory elements."""
        plot_name = "Regulatory Element Counts"
        element_cols = [
            "kozak_sequence_present",
            "TOP_motif_present",
            "G_quadruplex_present",
            "polyA_signal_present",
            "ARE_motifs_present",
        ]
        element_cols_present = [col for col in element_cols if col in feature_df.columns]
        if not element_cols_present:
            logger.error("No regulatory element boolean columns found.")
            return None
        if feature_df.empty:
            logger.error(f"Cannot generate {plot_name}: Input DataFrame empty.")
            return None
        fig: Figure | None = None
        try:
            counts = feature_df[element_cols_present].sum()
            labels = [
                col.replace("_present", "").replace("_sequence", "").replace("_", " ").title()
                for col in counts.index
            ]
            fig_size = figsize or self.viz_config.get("default_figsize", (8, 5))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            sns.barplot(
                x=labels, y=counts.values, ax=ax, palette=PALETTE_BAR, hue=labels, legend=False
            )
            ax.set_title("Presence of Regulatory Elements in Expressed Genes")
            ax.set_xlabel("Regulatory Element / Motif")
            ax.set_ylabel("Number of Genes Present")
            ax.tick_params(axis="x", rotation=45)
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
            sns.despine(ax=ax)
            fig.tight_layout()
            self.figures["regulatory_elements"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig

    def plot_codon_usage_heatmap(
        self,
        codon_usage_table: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        sort_by_cv: bool = True,
        top_n: int = 20,
        figsize: Tuple[int, int] | None = None,
    ) -> Figure | None:
        """Plots a heatmap of codon usage frequencies."""
        plot_name = "Codon Usage Heatmap"
        if codon_usage_table.empty:
            logger.error(f"Cannot generate {plot_name}: Table is empty.")
            return None
        fig: Figure | None = None
        try:
            if sort_by_cv:
                required_cols = ["cv", "ensembl_transcript_id"]
                if feature_df is None or not self._check_required_columns(
                    feature_df, required_cols, plot_name + " (sorting)"
                ):
                    logger.warning("Cannot sort by CV for heatmap. Plotting top N as given.")
                    transcripts_to_plot = codon_usage_table.index[:top_n].tolist()
                else:
                    stable_genes = feature_df.sort_values("cv", ascending=True, na_position="last")
                    common_transcripts = stable_genes["ensembl_transcript_id"].isin(
                        codon_usage_table.index
                    )
                    transcripts_to_plot = (
                        stable_genes.loc[common_transcripts, "ensembl_transcript_id"]
                        .head(top_n)
                        .tolist()
                    )
            else:
                transcripts_to_plot = codon_usage_table.index[:top_n].tolist()
            if not transcripts_to_plot:
                logger.error(f"No transcripts selected for {plot_name}.")
                return None

            codon_usage_subset = codon_usage_table.loc[transcripts_to_plot]
            n_transcripts = len(codon_usage_subset)
            fig_size = figsize or self.viz_config.get("default_figsize", (14, 8))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            sns.heatmap(
                codon_usage_subset,
                cmap=CMAP_SEQUENTIAL,
                xticklabels=True,
                yticklabels=True,
                ax=ax,
                linewidths=0.1,
                linecolor="lightgrey",
                cbar_kws={"shrink": 0.7, "label": "Codon Usage Frequency"},
            )
            title_suffix = (
                f"(Top {n_transcripts} Most Stable Genes)"
                if sort_by_cv and feature_df is not None
                else f"(Top {n_transcripts} Genes)"
            )
            ax.set_title(f"Codon Usage {title_suffix}")
            ax.set_xlabel("Codon")
            ax.set_ylabel("Transcript ID")
            ax.tick_params(axis="x", labelsize=FONT_SIZE_ANNOTATION - 1, rotation=90)
            ax.tick_params(axis="y", labelsize=FONT_SIZE_ANNOTATION - 1, rotation=0)
            fig.tight_layout()
            self.figures["codon_usage_heatmap"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig

    def plot_average_codon_usage(
        self,
        codon_usage_table: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] | None = None,
    ) -> Figure | None:
        """Plots a bar chart of the average usage frequency for the top N codons."""
        plot_name = "Average Codon Usage"
        if not codon_usage_table.columns.tolist():
            logger.error(f"Cannot generate {plot_name}: Table empty/no columns.")
            return None
        fig: Figure | None = None
        try:
            avg_usage = codon_usage_table.mean().sort_values(ascending=False)
            top_codons = avg_usage.head(top_n)
            fig_size = figsize or self.viz_config.get("default_figsize", (10, 5))
            fig_dpi = self.viz_config.get("default_dpi") or DEFAULT_DPI
            fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
            sns.barplot(
                x=top_codons.index,
                y=top_codons.values,
                ax=ax,
                palette=PALETTE_BAR,
                hue=top_codons.index,
                legend=False,
            )
            ax.set_title(f"Top {top_n} Most Frequently Used Codons (Average)")
            ax.set_xlabel("Codon")
            ax.set_ylabel("Average Usage Frequency")
            ax.tick_params(axis="x", rotation=90)
            sns.despine(ax=ax)
            fig.tight_layout()
            self.figures["average_codon_usage"] = fig
        except Exception as e:
            logger.exception(f"Error during {plot_name} generation: {e}")
            if fig is not None:
                plt.close(fig)
            fig = None
        return fig
