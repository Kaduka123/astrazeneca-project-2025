# cho_analysis/task1/ranking.py
"""Gene ranking for clone selection markers."""

import logging
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from cho_analysis.core.config import get_correlation_config, get_visualization_config
from cho_analysis.core.logging import setup_logging

logger = setup_logging(__name__)


class GeneRanking:
    """Ranks genes as potential selection markers based on correlation and expression."""

    def __init__(self, config=None):
        """Initialize gene ranking."""
        self.figures = {}

    # --- run_ranking_analysis function ---
    def run_ranking_analysis(
        self,
        correlation_df: pd.DataFrame,
        expression_df: pd.DataFrame,
        methods: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs full ranking analysis, calculating detailed scores including rank stability.

        Args:
            correlation_df: DataFrame containing correlation results, potentially
                            including bootstrap columns like 'Rank_Stability_{Method}_Perc'.
            expression_df: DataFrame with expression data (genes x samples or with metadata).
            methods: List of correlation methods used (to find stability columns).
            top_n: Number of top ranked genes to return in the 'ranked_markers' list.

        Returns:
            Dictionary containing:
                - 'rank_statistics': DataFrame with detailed ranking scores for all genes.
                - 'ranked_markers': List of gene symbols for the top N ranked genes.
        """
        # --- Parameter Handling ---
        config = get_correlation_config()
        # Use provided methods list if available, otherwise get from config
        if methods is None:
            methods = config.get("methods", ["spearman"])
        if top_n is None:
            top_n = config.get("top_n", 20)
        min_correlation = config.get("min_correlation", 0.5)  # Threshold for considering genes
        max_p_value = config.get("max_p_value", 0.05)  # Threshold for considering genes

        logger.info(
            f"Running gene ranking: methods={methods}, criteria: min_corr>={min_correlation}, max_p<={max_p_value}"
        )

        # --- Prepare Expression Data ---
        expr_data = None
        gene_id_col_expr = "ensembl_transcript_id"  # Assume this is the ID in expression_df
        sym_col = "sym"
        expression_df = expression_df.copy()
        is_index_id_expr = False

        # Determine structure of expression_df (IDs in column vs index)
        if gene_id_col_expr in expression_df.columns:
            if expression_df[gene_id_col_expr].isnull().any():
                logger.warning(
                    "Expression data has NaNs in the primary ID column. Rows with NaN IDs will be dropped."
                )
                expression_df = expression_df.dropna(subset=[gene_id_col_expr])
            if expression_df[gene_id_col_expr].duplicated().any():
                logger.warning(
                    f"Duplicate IDs found in '{gene_id_col_expr}'. Keeping first instance."
                )
                expression_df = expression_df.drop_duplicates(
                    subset=[gene_id_col_expr], keep="first"
                )

            meta_cols = [
                c
                for c in [gene_id_col_expr, sym_col, "ensembl_peptide_id"]
                if c in expression_df.columns
            ]
            sample_cols = [c for c in expression_df.columns if c not in meta_cols]
            expr_numeric = expression_df[sample_cols].apply(pd.to_numeric, errors="coerce")
            expr_data = expr_numeric.set_index(expression_df[gene_id_col_expr])
            logger.debug(f"Ranking: Using expression data indexed by '{gene_id_col_expr}'.")
        # Check if index looks like gene IDs (not just default RangeIndex)
        elif expression_df.index.name or not isinstance(expression_df.index, pd.RangeIndex):
            is_index_id_expr = True
            gene_id_col_expr = expression_df.index.name if expression_df.index.name else "index_id"
            if gene_id_col_expr == "index_id":
                logger.warning("Expression index unnamed, using 'index_id'.")
            # Assume all columns except potential 'sym' are samples
            sample_cols = [c for c in expression_df.columns if c != sym_col]
            expr_data = expression_df[sample_cols].apply(pd.to_numeric, errors="coerce")
            logger.debug(f"Ranking: Using expression data indexed by '{gene_id_col_expr}'.")
        else:
            logger.error("Cannot determine gene ID source in expression data for ranking.")
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}

        if expr_data is None or expr_data.empty:
            logger.error("Ranking: Failed to prepare numeric expression data.")
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}

        # --- Prepare Correlation Data ---
        correlation_df = correlation_df.copy()
        # Find the gene ID column in correlation_df (must match expression_df IDs)
        corr_id_col = None
        possible_id_cols = [gene_id_col_expr, "ensembl_transcript_id", correlation_df.index.name]
        for col in possible_id_cols:
            if col and col in correlation_df.columns or col and correlation_df.index.name == col:
                corr_id_col = col
                break
        if corr_id_col is None:
            logger.error(
                f"Cannot find a gene ID column in correlation_df matching expression ID '{gene_id_col_expr}'. Tried: {possible_id_cols}"
            )
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}
        is_corr_id_index = corr_id_col == correlation_df.index.name
        logger.info(f"Using '{corr_id_col}' as the gene ID column from correlation data.")

        # --- Get Gene Symbols Map ---
        gene_symbol_map = {}
        if sym_col in correlation_df.columns:
            try:
                if is_corr_id_index:
                    id_sym_df = correlation_df[[sym_col]].reset_index()  # Get index as column
                    id_sym_df = id_sym_df.dropna(subset=[corr_id_col, sym_col]).drop_duplicates(
                        subset=[corr_id_col]
                    )
                    gene_symbol_map = pd.Series(
                        id_sym_df[sym_col].values, index=id_sym_df[corr_id_col]
                    ).to_dict()
                else:
                    id_sym_df = (
                        correlation_df[[corr_id_col, sym_col]]
                        .dropna()
                        .drop_duplicates(subset=[corr_id_col])
                    )
                    gene_symbol_map = pd.Series(
                        id_sym_df[sym_col].values, index=id_sym_df[corr_id_col]
                    ).to_dict()
            except Exception as map_err:
                logger.warning(f"Could not create symbol map: {map_err}")

        # --- Filter Correlation & Aggregate Ranks/Scores ---
        gene_ranks_data: Dict[Any, Dict] = {}
        # Apply significance filters BEFORE aggregating over methods
        filtered_corr_df = correlation_df[
            (correlation_df["Correlation_Coefficient_Abs"] >= min_correlation)
            & (correlation_df["Correlation_P_Value"] <= max_p_value)
        ].copy()  # Use copy to avoid SettingWithCopyWarning

        if filtered_corr_df.empty:
            logger.warning(
                f"Ranking: No genes passed correlation/p-value filters (min_corr>={min_correlation}, max_p<={max_p_value}). Ranking cannot proceed."
            )
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}

        logger.info(
            f"Ranking based on {len(filtered_corr_df)} gene-method entries passing filters."
        )

        # Ensure Correlation_Rank_Score exists, calculate if missing
        if "Correlation_Rank_Score" not in filtered_corr_df.columns:
            logger.warning("Calculating 'Correlation_Rank_Score' as abs(corr)/(p_val + 1e-10).")
            # Handle potential division by zero or NaN p-values
            p_val_safe = filtered_corr_df["Correlation_P_Value"].fillna(1.0) + 1e-10
            filtered_corr_df["Correlation_Rank_Score"] = (
                filtered_corr_df["Correlation_Coefficient_Abs"] / p_val_safe
            )

        # Iterate through the methods requested by the user that are PRESENT in the filtered data
        methods_in_filtered = filtered_corr_df["Correlation_Type"].unique()
        # Capitalize methods for matching
        used_methods = [m.capitalize() for m in methods if m.capitalize() in methods_in_filtered]
        if not used_methods:
            logger.error(
                f"None of the requested methods ({methods}) were found in the filtered correlation data."
            )
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}
        logger.info(f"Aggregating ranks using methods: {used_methods}")

        # Aggregate stats per gene from the filtered data
        for gene_id, group in filtered_corr_df.groupby(corr_id_col):
            gene_stats = {
                "methods": [],
                "ranks": [],
                "scores": [],
                "correlations": [],
                "p_values": [],
                "stability_scores": [],
            }
            # Iterate through each method's result for this gene
            for method_cap in used_methods:
                method_row = group[group["Correlation_Type"] == method_cap]
                if not method_row.empty:
                    row_data = method_row.iloc[
                        0
                    ]  # Take the first (should be only one per gene-method)
                    gene_stats["methods"].append(method_cap.lower())  # Store lowercase method name
                    # Calculate rank within this method *after* filtering
                    # This requires re-ranking the filtered data for each method
                    # For simplicity here, we'll skip per-method rank calculation and focus on scores/stability
                    # gene_stats["ranks"].append(rank) # Rank calculation would need full method_df sorted
                    gene_stats["scores"].append(row_data.get("Correlation_Rank_Score", 0.0))
                    gene_stats["correlations"].append(row_data.get("Correlation_Coefficient", 0.0))
                    gene_stats["p_values"].append(row_data.get("Correlation_P_Value", 1.0))

                    stability_col = f"Rank_Stability_{method_cap}_Perc"
                    if stability_col in row_data:
                        stability = row_data.get(stability_col, 0.0)
                        gene_stats["stability_scores"].append(
                            stability if pd.notna(stability) else 0.0
                        )
                    else:
                        # If bootstrap wasn't run or failed, stability is unknown (or 0)
                        gene_stats["stability_scores"].append(0.0)

            # Only add gene if it had results for at least one used method
            if gene_stats["methods"]:
                gene_ranks_data[gene_id] = gene_stats

        if not gene_ranks_data:
            logger.error(
                "No gene data aggregated after filtering and method selection. Cannot rank."
            )
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}

        # --- Calculate Expression Stats & Final Scores ---
        logger.info("Calculating expression stats and final ranking scores...")
        rank_stats = []
        genes_in_expr = set(expr_data.index)

        # Normalize Scores (Correlation and Stability)
        all_corr_scores = [
            np.mean(s["scores"]) for s in gene_ranks_data.values() if s.get("scores")
        ]
        max_corr_score_norm = max(all_corr_scores) if all_corr_scores else 1.0
        if max_corr_score_norm == 0:
            max_corr_score_norm = 1.0  # Avoid division by zero if all scores are 0

        # Pre-calculate expression stats for normalization
        all_levels, all_vars = [], []
        gene_expr_stats_cache = {}
        for gene_id in gene_ranks_data:
            if gene_id in genes_in_expr:
                # Ensure index type matches if necessary (e.g., both strings)
                try:
                    expr_values = expr_data.loc[gene_id].values
                except KeyError:
                    logger.debug(
                        f"Gene ID '{gene_id}' not found in expression data index. Skipping expr stats."
                    )
                    continue

                if pd.api.types.is_numeric_dtype(expr_values) and not np.isnan(expr_values).all():
                    mean_expr = np.nanmean(expr_values)
                    std_expr = np.nanstd(expr_values)
                    # Calculate CV only if mean is reasonably far from zero
                    if abs(mean_expr) > 1e-9:
                        cv = std_expr / mean_expr
                    else:
                        cv = np.inf  # Assign Inf CV if mean is zero/tiny
                    gene_expr_stats_cache[gene_id] = {"level": mean_expr, "variability_cv": cv}
                    all_levels.append(mean_expr)
                    if np.isfinite(cv):
                        all_vars.append(cv)  # Only use finite CVs for normalization stats

        # Calculate normalization parameters (handle cases with no valid data)
        mean_level = np.mean(all_levels) if all_levels else 0
        std_level = np.std(all_levels) if all_levels else 1.0
        mean_var = np.mean(all_vars) if all_vars else 0  # Mean of finite CVs
        std_var = np.std(all_vars) if all_vars else 1.0  # Std of finite CVs
        # Avoid zero std dev for normalization
        if std_level < 1e-9:
            std_level = 1.0
        if std_var < 1e-9:
            std_var = 1.0

        # Calculate final scores for each gene
        for gene_id, stats in gene_ranks_data.items():
            # --- Normalized Correlation Score ---
            mean_corr_rank_score = np.mean(stats["scores"]) if stats.get("scores") else 0
            corr_score_norm = max(
                0.0, min(1.0, mean_corr_rank_score / max_corr_score_norm)
            )  # Clip 0-1

            # --- Normalized Expression Level & Stability Scores ---
            level, variability, level_score, stability_expr_score = np.nan, np.nan, 0.0, 0.0
            if gene_id in gene_expr_stats_cache:
                level = gene_expr_stats_cache[gene_id]["level"]
                variability = gene_expr_stats_cache[gene_id]["variability_cv"]

                # Level score (higher expression = better, using sigmoid on Z-score)
                if pd.notna(level):
                    z_level = (level - mean_level) / std_level
                    level_score = 1 / (1 + np.exp(-z_level))
                # Expression Stability score (lower CV = better stability = higher score)
                if pd.notna(variability) and np.isfinite(variability):
                    # Normalize CV: lower values should map closer to 1
                    z_var = (variability - mean_var) / std_var
                    # Use inverse sigmoid: high z_var (high CV) -> low score
                    stability_expr_score = 1.0 - (1 / (1 + np.exp(-z_var)))
                elif variability == np.inf:  # Handle infinite CV (mean near zero)
                    stability_expr_score = 0.0  # Penalize infinite CV

            # --- Normalized Rank Stability Score ---
            # Average the stability percentages across methods for this gene
            mean_stability_perc = (
                np.mean(stats.get("stability_scores", [0.0]))
                if stats.get("stability_scores")
                else 0.0
            )
            # Normalize stability score (0-1)
            rank_stability_score_norm = max(0.0, min(1.0, mean_stability_perc / 100.0))

            # --- Final Weighted Score ---
            # Define weights (ensure they sum to 1 if desired, adjust as needed)
            weights = {
                "correlation": 0.40,
                "level": 0.30,
                "expression_stability": 0.15,
                "rank_stability": 0.15,
            }
            final_score = (
                weights["correlation"] * corr_score_norm
                + weights["level"] * level_score
                + weights["expression_stability"] * stability_expr_score
                + weights["rank_stability"] * rank_stability_score_norm
            )

            # --- Compile Results ---
            rank_stats.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol_map.get(gene_id, str(gene_id)),
                    "num_methods": len(stats["methods"]),
                    "methods": ", ".join(stats["methods"]),
                    # "mean_rank": np.mean(stats["ranks"]) if stats["ranks"] else np.nan, # Rank wasn't easily calculable here
                    "mean_corr_rank_score": mean_corr_rank_score,  # Raw avg score over methods
                    "mean_correlation": np.mean(stats["correlations"])
                    if stats["correlations"]
                    else np.nan,
                    "mean_abs_correlation": np.mean(np.abs(stats["correlations"]))
                    if stats["correlations"]
                    else np.nan,
                    "geo_mean_p_value": np.exp(
                        np.mean(np.log(np.clip(stats["p_values"], 1e-300, 1.0)))
                    )
                    if stats["p_values"]
                    else np.nan,
                    "correlation_score": corr_score_norm,  # Normalized component score
                    "expression_level": level,
                    "expression_variability_cv": variability,  # Store raw CV
                    "expression_level_score": level_score,  # Normalized component score
                    "expression_stability_score": stability_expr_score,  # Normalized component score (higher is better)
                    "mean_rank_stability_perc": mean_stability_perc,  # Store avg stability %
                    "rank_stability_score": rank_stability_score_norm,  # Normalized component score
                    "final_score": final_score,  # Final weighted score
                }
            )

        # --- Final DataFrame and Return ---
        if not rank_stats:
            logger.warning("Ranking: No genes available after processing ranks and scores.")
            return {"rank_statistics": pd.DataFrame(), "ranked_markers": []}

        rank_df = (
            pd.DataFrame(rank_stats)
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        # Ensure gene_symbol is present and near the start
        if "gene_symbol" not in rank_df.columns:  # Add if missing
            rank_df["gene_symbol"] = (
                rank_df["gene_id"].map(gene_symbol_map).fillna(rank_df["gene_id"])
            )
        cols = ["gene_symbol", "gene_id"] + [
            c for c in rank_df.columns if c not in ["gene_symbol", "gene_id"]
        ]
        rank_df = rank_df[cols]

        # Get top N ranked marker *symbols*
        top_genes_df = rank_df.head(top_n)
        # Handle case where gene_symbol might still be missing if mapping failed entirely
        ranked_markers_list = top_genes_df["gene_symbol"].fillna(top_genes_df["gene_id"]).tolist()

        logger.info(
            f"Generated ranking for {len(rank_df)} genes, returning top {len(ranked_markers_list)} symbols."
        )

        return {"rank_statistics": rank_df, "ranked_markers": ranked_markers_list}

    # --- visualize_ranking function ---
    def visualize_ranking(
        self, ranked_genes: pd.DataFrame, figsize: tuple[int, int] = (12, 8)
    ) -> Figure:
        """Visualize gene ranking results including radar plot of scores."""
        required_cols = [
            "final_score",
            "gene_symbol",
            "correlation_score",
            "expression_level_score",
            "expression_stability_score",
            "rank_stability_score",
        ]
        # Check if all required columns exist
        if ranked_genes.empty or not all(col in ranked_genes.columns for col in required_cols):
            missing = [col for col in required_cols if col not in ranked_genes.columns]
            msg = f"Ranking data missing required columns for visualization: {missing}"
            logger.error(msg)  # Log the error
            # Create and return an error figure instead of crashing
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            return fig  # Return the error figure

        # --- Proceed with plotting if columns are present ---
        try:
            plt_style = get_visualization_config().get("style", "seaborn-v0_8-whitegrid")
            plt.style.use(plt_style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")

        fig = plt.figure(figsize=figsize, dpi=120)
        gs = fig.add_gridspec(1, 2)

        # --- Bar chart (Left subplot) ---
        ax1 = fig.add_subplot(gs[0, 0])
        top_genes_bar = ranked_genes.head(10).sort_values("final_score", ascending=True)
        labels_bar = top_genes_bar["gene_symbol"].values
        scores_bar = top_genes_bar["final_score"].values
        bars = ax1.barh(range(len(labels_bar)), scores_bar, color="skyblue")
        ax1.set_yticks(range(len(labels_bar)))
        ax1.set_yticklabels(labels_bar)
        ax1.set_xlabel("Overall Marker Score")
        ax1.set_title(f"Top {len(labels_bar)} Ranked Genes")
        ax1.tick_params(axis="y", labelsize=9)
        ax1.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars:
            width = bar.get_width()
            ax1.text(
                width * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=8,
            )
        ax1.set_xlim(right=max(scores_bar) * 1.15)

        # --- Radar chart (Right subplot) ---
        ax2 = fig.add_subplot(gs[0, 1], projection="polar")
        top_n_radar = min(5, len(ranked_genes))
        radar_genes = ranked_genes.head(top_n_radar)

        criteria = [
            "correlation_score",
            "expression_level_score",
            "expression_stability_score",
            "rank_stability_score",
        ]
        criteria_labels = [
            "Correlation",
            "Expression Level",
            "Expression Stability",
            "Rank Stability",
        ]
        # Select the correct columns
        data_radar = radar_genes[criteria].clip(0, 1).values

        n_criteria = len(criteria)
        angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
        angles += angles[:1]

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(criteria_labels)
        ax2.set_yticks(np.arange(0.2, 1.01, 0.2))
        ax2.set_ylim(0, 1.05)

        colors = cm.viridis(np.linspace(0, 1, top_n_radar))
        for i in range(top_n_radar):
            values = data_radar[i].flatten().tolist()
            values += values[:1]
            ax2.plot(
                angles,
                values,
                color=colors[i],
                linewidth=1.5,
                linestyle="solid",
                label=radar_genes.iloc[i]["gene_symbol"],
            )
            ax2.fill(angles, values, color=colors[i], alpha=0.2)

        ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        ax2.set_title(f"Top {top_n_radar} Criteria Profile", y=1.12)

        plt.tight_layout(w_pad=3)
        self.figures["ranking_results"] = fig
        return fig

    # --- binary_marker_identification function ---
    def binary_marker_identification(
        self,
        expression_df: pd.DataFrame,
        target_col: str = "PRODUCT-TG",
        threshold_percentile: float = 0.5,
        top_n: int = 20,
        min_auc: float = 0.7,
        prefilter_genes: int = 5000,
    ) -> pd.DataFrame:
        """Identifies genes as potential binary markers based on AUC score
        predicting a binarized target variable.

        Args:
            expression_df: DataFrame with expression data (genes x samples or
                           includes metadata cols like 'ensembl_transcript_id').
            target_col: Name of the target gene column/index to predict.
            threshold_percentile: Percentile to binarize the target variable.
            top_n: Number of top binary markers to return.
            min_auc: Minimum Cross-Validated AUC score required for a gene
                     to be considered a potential marker.
            prefilter_genes: Max number of genes to evaluate (selected by
                             variance) for performance. Set <= 0 to disable.

        Returns:
            DataFrame containing potential binary markers sorted by a combined
            score, including AUC, CV AUC, direction, and mean expression.
            Returns an empty DataFrame if errors occur or no markers are found.
        """
        start_time_binary_id = time.time()
        logger.info(f"Identifying binary markers (processing up to {prefilter_genes} genes)...")
        default_return = pd.DataFrame(
            columns=[
                "gene_id",
                "gene_symbol",
                "auc",
                "cv_auc",
                "direction",
                "mean_expr",
                "marker_score",
            ]
        )

        # --- Prepare Expression Data & Target ---
        target_values: Optional[np.ndarray] = None
        expr_values: Optional[np.ndarray] = None
        gene_ids: Optional[np.ndarray] = None
        gene_symbols: Optional[np.ndarray] = None
        expression_df = expression_df.copy()
        id_col = "ensembl_transcript_id"
        sym_col = "sym"

        try:
            if id_col in expression_df.columns:
                # Case 1: Metadata in columns, IDs in id_col
                target_row = (
                    expression_df[expression_df[sym_col] == target_col]
                    if sym_col in expression_df.columns
                    else pd.DataFrame()
                )
                if target_row.empty:
                    target_row = expression_df[expression_df[id_col] == target_col]
                if target_row.empty:
                    msg = f"Target '{target_col}' not found by symbol or ID."
                    raise ValueError(msg)

                meta_cols = [
                    c for c in [id_col, sym_col, "ensembl_peptide_id"] if c in expression_df.columns
                ]
                sample_cols = [c for c in expression_df.columns if c not in meta_cols]
                target_idx = target_row.index[0]  # Get index of target row
                # Extract target values using sample columns
                target_values = pd.to_numeric(
                    expression_df.loc[target_idx, sample_cols], errors="coerce"
                ).values

                # Prepare expression matrix (numeric, indexed by ID, target removed)
                expr_numeric = expression_df[sample_cols].apply(pd.to_numeric, errors="coerce")
                valid_id_rows = expression_df[id_col].notna()
                if not valid_id_rows.any():
                    msg = "No valid gene IDs found."
                    raise ValueError(msg)
                expr_data_filtered = expr_numeric[valid_id_rows].set_index(
                    expression_df.loc[valid_id_rows, id_col]
                )
                target_gene_id = target_row[id_col].iloc[0]  # Get the ID of the target gene itself
                if target_gene_id in expr_data_filtered.index:
                    expr_data_filtered = expr_data_filtered.drop(target_gene_id)

                expr_values = expr_data_filtered.values
                gene_ids = expr_data_filtered.index.values  # Get final gene IDs

                # Robust symbol mapping using the final gene_ids
                if sym_col in expression_df.columns:
                    # Map original IDs to symbols
                    sym_map_base = pd.Series(
                        expression_df[sym_col].values, index=expression_df[id_col]
                    )
                    # Reindex to match the filtered gene IDs
                    sym_series_reindexed = sym_map_base.reindex(gene_ids)
                    # Create Series of gene IDs indexed by themselves for filling NaNs
                    ids_as_series_for_fill = pd.Series(gene_ids, index=gene_ids)
                    gene_symbols = sym_series_reindexed.fillna(ids_as_series_for_fill).values
                else:
                    gene_symbols = gene_ids  # Fallback to IDs if no symbol column

            elif target_col in expression_df.index:
                # Case 2: Index contains gene IDs, target is an index entry
                expr_numeric = expression_df.apply(pd.to_numeric, errors="coerce")
                target_values = expr_numeric.loc[target_col].values
                expr_data_filtered = expr_numeric.drop(target_col)
                expr_values = expr_data_filtered.values
                gene_ids = expr_data_filtered.index.values
                # Try to get symbols if 'sym' column exists, even if ID is index
                if sym_col in expression_df.columns:
                    sym_map_base = pd.Series(
                        expression_df[sym_col].values, index=expression_df.index
                    )
                    sym_series_reindexed = sym_map_base.reindex(gene_ids)
                    ids_as_series_for_fill = pd.Series(gene_ids, index=gene_ids)
                    gene_symbols = sym_series_reindexed.fillna(ids_as_series_for_fill).values
                else:
                    gene_symbols = gene_ids

            else:  # Case 3: Cannot find target
                msg = f"Target '{target_col}' not found in columns or index."
                raise ValueError(msg)

            if (
                target_values is None
                or expr_values is None
                or gene_ids is None
                or gene_symbols is None
            ):
                msg = "Data preparation failed, essential components are None."
                raise ValueError(msg)

        except Exception as prep_err:
            logger.exception(f"Failed data preparation for binary markers: {prep_err}")
            return default_return

        # --- Create Binary Target & Filter Data ---
        valid_target_idx = ~np.isnan(target_values)
        if not np.any(valid_target_idx):
            logger.error("Target values are all NaN. Cannot create binary target.")
            return default_return
        target_values_valid = target_values[valid_target_idx]
        if len(target_values_valid) < 5:
            logger.error("Too few valid target values to determine threshold.")
            return default_return

        # Use try-except for percentile calculation as it can fail with weird data
        try:
            threshold = np.percentile(target_values_valid, threshold_percentile * 100)
        except IndexError:
            msg = "Failed to calculate percentile for target threshold."
            logger.exception(msg)
            return default_return

        binary_target = (target_values_valid > threshold).astype(int)
        # Filter expression columns based on valid target indices
        expr_values_filtered = expr_values[:, valid_target_idx]

        logger.info(
            f"Binary target: {np.sum(binary_target)} high vs {len(binary_target) - np.sum(binary_target)} low (threshold={threshold:.3f})"
        )
        if len(np.unique(binary_target)) < 2:
            msg = "Binary target variable has only one class. Cannot calculate AUC."
            logger.error(msg)
            return default_return

        # --- Pre-filter genes ---
        num_avail = expr_values_filtered.shape[0]
        indices_to_process = np.arange(num_avail)
        if 0 < prefilter_genes < num_avail:
            logger.info(f"Prefiltering {num_avail} genes to top {prefilter_genes} by variance...")
            with np.errstate(invalid="ignore"):  # Ignore warnings from variance on rows with NaNs
                variances = np.nanvar(expr_values_filtered, axis=1)
            variances[np.isnan(variances)] = -np.inf  # Ensure NaNs are last when sorting
            # Use argsort and take the last 'prefilter_genes' indices
            indices_to_process = np.argsort(variances)[-prefilter_genes:]

        # --- Evaluate Genes ---
        binary_markers = []
        total_to_process = len(indices_to_process)
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Stratified CV

        for processed_count, i in enumerate(indices_to_process):
            if (processed_count + 1) % 500 == 0:
                logger.info(
                    f"  Binary marker progress: {processed_count + 1}/{total_to_process}..."
                )

            gene_expr_all_samples = expr_values_filtered[
                i, :
            ]  # Get expression for this gene across filtered samples
            gene_id = gene_ids[i]
            gene_sym = gene_symbols[i]

            # Filter out NaNs specifically for this gene's expression values
            valid_expr_idx_gene = ~np.isnan(gene_expr_all_samples)
            if not np.any(valid_expr_idx_gene):
                continue  # Skip gene if all its expression values are NaN

            # Align this gene's valid expression with the corresponding binary target values
            gene_expr_valid = gene_expr_all_samples[valid_expr_idx_gene]
            binary_target_gene_valid = binary_target[valid_expr_idx_gene]

            # Check for sufficient data points and variance after filtering
            if (
                len(gene_expr_valid) < 5
                or np.std(gene_expr_valid) < 1e-6
                or len(np.unique(binary_target_gene_valid)) < 2
            ):
                continue  # Skip constant expression or single target class for this gene

            try:
                # Calculate AUC on the valid data for this gene
                auc = roc_auc_score(binary_target_gene_valid, gene_expr_valid)
                direction = "positive"
                if auc < 0.5:
                    auc = 1 - auc  # AUC should be >= 0.5, indicates prediction direction
                    direction = "negative"

                # Calculate Cross-Validated AUC
                cv_auc = 0.0  # Default CV AUC
                try:
                    x = gene_expr_valid.reshape(-1, 1)
                    scaler = StandardScaler()
                    x_scaled = scaler.fit_transform(x)
                    model = LogisticRegression(
                        random_state=42, class_weight="balanced", solver="liblinear"
                    )
                    # Suppress warnings from sklearn about folds with single class during CV
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        cv_scores = cross_val_score(
                            model,
                            x_scaled,
                            binary_target_gene_valid,
                            cv=cv_strategy,
                            scoring="roc_auc",
                            n_jobs=1,
                        )
                    # Only use mean if all scores are valid
                    if not np.isnan(cv_scores).any():
                        cv_auc = np.mean(cv_scores)

                except (
                    ValueError
                ) as cv_val_err:  # Catch error if CV fails (e.g., single class in a fold)
                    logger.debug(
                        f"CV AUC ValueError for {gene_sym} (likely single class in fold): {cv_val_err}"
                    )
                except Exception as cv_err:  # Catch other potential CV errors
                    logger.debug(f"CV AUC calculation error for {gene_sym}: {cv_err}")

                # Add marker if CV AUC meets threshold
                if cv_auc >= min_auc:
                    marker_score = 0.7 * cv_auc + 0.3 * auc  # Weighted score
                    mean_val = np.mean(
                        gene_expr_valid
                    )  # Mean of valid expression values for this gene
                    binary_markers.append(
                        {
                            "gene_id": gene_id,
                            "gene_symbol": gene_sym,
                            "auc": auc,
                            "cv_auc": cv_auc,
                            "direction": direction,
                            "mean_expr": mean_val,
                            "marker_score": marker_score,
                        }
                    )
            except ValueError as roc_err:  # Catch AUC calculation error (e.g. only one class left)
                logger.debug(f"ROC AUC ValueError for {gene_sym}: {roc_err}")
            except Exception as calc_err:  # Catch any other calculation errors
                logger.warning(f"AUC calculation failed for {gene_sym}: {calc_err}")

        logger.info(
            f"Binary marker identification loop took {time.time() - start_time_binary_id:.2f}s."
        )
        if not binary_markers:
            logger.info("No binary markers found meeting criteria.")
            return default_return

        markers_df = (
            pd.DataFrame(binary_markers)
            .sort_values("marker_score", ascending=False)
            .reset_index(drop=True)
        )
        return markers_df.head(top_n)

    # --- find_optimal_thresholds function ---
    def find_optimal_thresholds(
        self,
        expression_df: pd.DataFrame,
        target_col: str = "PRODUCT-TG",
        marker_genes: Optional[pd.DataFrame] = None,
        num_thresholds: int = 20,
    ) -> pd.DataFrame:
        """Finds optimal expression thresholds for binary markers using balanced accuracy."""
        if marker_genes is None or marker_genes.empty:
            logger.warning("No marker genes provided for threshold optimization.")
            return pd.DataFrame()
        logger.info(f"Finding optimal thresholds for {len(marker_genes)} marker genes...")

        # --- Prepare Data ---
        target_values, expr_data = None, None
        id_col = "ensembl_transcript_id"
        sym_col = "sym"
        expression_df = expression_df.copy()
        try:
            if id_col in expression_df.columns:
                target_row = (
                    expression_df[expression_df[sym_col] == target_col]
                    if sym_col in expression_df
                    else pd.DataFrame()
                )
                if target_row.empty:
                    target_row = expression_df[expression_df[id_col] == target_col]
                if target_row.empty:
                    msg = f"Target '{target_col}' not found."
                    raise ValueError(msg)
                meta_cols = [
                    c for c in [id_col, sym_col, "ensembl_peptide_id"] if c in expression_df.columns
                ]
                sample_cols = [c for c in expression_df.columns if c not in meta_cols]
                target_idx = target_row.index[0]
                target_values = pd.to_numeric(
                    expression_df.loc[target_idx, sample_cols], errors="coerce"
                ).values
                expr_numeric = expression_df[sample_cols].apply(pd.to_numeric, errors="coerce")
                valid_id_rows = expression_df[id_col].notna()
                expr_data = expr_numeric[valid_id_rows].set_index(
                    expression_df.loc[valid_id_rows, id_col]
                )
            else:  # Assume index is ID
                if target_col not in expression_df.index:
                    msg = f"Target '{target_col}' not found."
                    raise ValueError(msg)
                expr_numeric = expression_df.apply(pd.to_numeric, errors="coerce")
                target_values = expr_numeric.loc[target_col].values
                expr_data = expr_numeric  # Index is already gene IDs
            if target_values is None or expr_data is None:
                msg = "Data prep failed."
                raise ValueError(msg)
        except Exception as prep_err:
            logger.exception(f"Failed data preparation for thresholds: {prep_err}")
            return pd.DataFrame()

        # Filter based on valid target values
        valid_target_idx = ~np.isnan(target_values)
        if not np.any(valid_target_idx):
            msg = "Target values are all NaN."
            raise ValueError(msg)
        median_target = np.nanmedian(target_values)
        target_binary = (target_values[valid_target_idx] > median_target).astype(int)
        expr_data_filtered = expr_data.iloc[:, valid_target_idx]  # Filter expression columns

        # --- Iterate through markers ---
        result_data = []
        marker_info = marker_genes.set_index("gene_id")  # For quick lookup of symbol/direction

        for gene_id in marker_info.index:
            if gene_id not in expr_data_filtered.index:
                continue
            gene_expr = expr_data_filtered.loc[gene_id].values
            if np.isnan(gene_expr).all() or np.nanstd(gene_expr) < 1e-6:
                continue

            valid_expr_idx = ~np.isnan(gene_expr)
            if not np.any(valid_expr_idx):
                continue  # Skip if all expr are NaN after filtering

            gene_expr_valid = gene_expr[valid_expr_idx]
            target_binary_valid = target_binary[
                valid_expr_idx
            ]  # Target corresponding to valid expr
            if len(np.unique(target_binary_valid)) < 2:
                continue  # Skip if only one class left

            min_expr, max_expr = np.percentile(gene_expr_valid, [5, 95])
            if max_expr <= min_expr:
                continue

            thresholds = np.linspace(min_expr, max_expr, num=num_thresholds)
            best_threshold, best_score = None, -1
            direction = marker_info.loc[gene_id, "direction"]

            for threshold in thresholds:
                preds = (
                    (gene_expr_valid >= threshold).astype(int)
                    if direction == "positive"
                    else (gene_expr_valid < threshold).astype(int)
                )
                try:
                    bal_acc = balanced_accuracy_score(target_binary_valid, preds)
                    if bal_acc > best_score:
                        best_score, best_threshold = bal_acc, threshold
                except ValueError:
                    continue  # Skip if balanced_accuracy fails

            if best_threshold is not None:
                # Recalculate metrics at best threshold
                best_preds = (
                    (gene_expr_valid >= best_threshold).astype(int)
                    if direction == "positive"
                    else (gene_expr_valid < best_threshold).astype(int)
                )
                tn, fp, fn, tp = confusion_matrix(target_binary_valid, best_preds).ravel()
                result_data.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": marker_info.loc[gene_id, "gene_symbol"],
                        "optimal_threshold": best_threshold,
                        "balanced_accuracy": best_score,
                        "accuracy": (tp + tn) / (tp + tn + fp + fn),
                        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "direction": direction,
                    }
                )

        if not result_data:
            logger.info("Could not determine optimal thresholds.")
            return pd.DataFrame()
        result_df = pd.DataFrame(result_data).sort_values("balanced_accuracy", ascending=False)
        return result_df
