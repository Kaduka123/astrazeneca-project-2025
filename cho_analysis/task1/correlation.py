# cho_analysis/task1/correlation.py
"""Gene correlation analysis for identifying marker genes related to target expression."""

from cho_analysis.core.config import get_correlation_config, get_visualization_config
from cho_analysis.core.logging import setup_logging

# Initialize logger for this module
logger = setup_logging(__name__)

import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.api as sm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.multitest import multipletests

from cho_analysis.core.visualization_utils import (
    FONT_SIZE_ANNOTATION,
)

try:
    import scanpy as sc
    from anndata import AnnData

    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    AnnData = type(None)


class GeneCorrelationAnalysis:
    """Class for analyzing gene correlations with a target variable (LC) in expression data.

    This class provides methods to identify genes whose expression levels correlate
    with the target gene (LC) for marker-based clone selection.
    """

    def __init__(self):
        """Initialize correlation analysis."""
        self.supported_methods = [
            "spearman",
            "pearson",
            "kendall",
            "regression",
            "decision_tree",
            "random_forest",
        ]
        self.figures = {}
        self.correlation_results = None

    def prepare_data(
        self,
        expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame | None = None,
        target_col: str | None = None,
    ) -> tuple[sc.AnnData | None, pd.Series | None]:
        """Transforms DataFrame to AnnData for correlation analysis."""
        if target_col is None:
            target_col = get_correlation_config().get("target_gene", "PRODUCT-TG")

        df = expression_df.copy()
        gene_id_col = "ensembl_transcript_id"
        sym_col = "sym"
        adata = None
        target_variable = None
        expression_data = None

        try:
            if gene_id_col in df.columns:
                meta_cols = [
                    c for c in [gene_id_col, sym_col, "ensembl_peptide_id"] if c in df.columns
                ]
                self.gene_metadata = df[meta_cols].set_index(gene_id_col)  # Store metadata

                target_row = df[df[sym_col] == target_col] if sym_col in df else pd.DataFrame()
                if target_row.empty:
                    target_row = df[df[gene_id_col] == target_col]
                if target_row.empty:
                    msg = f"Target gene '{target_col}' not found by symbol or ID."
                    raise ValueError(msg)
                target_id = target_row[gene_id_col].iloc[0]

                sample_cols = [c for c in df.columns if c not in meta_cols]
                expression_matrix = df[sample_cols].apply(
                    pd.to_numeric, errors="coerce"
                )  # Ensure numeric
                expression_matrix_t = expression_matrix.T
                expression_matrix_t.columns = df[gene_id_col].values  # Set gene IDs as columns

                target_variable = expression_matrix_t[target_id]
                expression_data = expression_matrix_t.drop(columns=[target_id])
            else:  # Assume index is gene ID
                if target_col not in df.index:
                    msg = f"Target gene '{target_col}' not found in index."
                    raise ValueError(msg)
                gene_id_col = (
                    df.index.name if df.index.name else "index_id"
                )  # Use index name or default

                df_numeric = df.apply(pd.to_numeric, errors="coerce")  # Ensure numeric
                df_t = df_numeric.T  # Samples as rows

                target_variable = df_t[target_col]
                expression_data = df_t.drop(columns=[target_col])
                self.gene_metadata = pd.DataFrame(
                    index=expression_data.columns
                )  # Create empty metadata df

            if expression_data is None or target_variable is None:
                msg = "Failed to separate target variable or expression data."
                raise ValueError(msg)

            # Create AnnData object
            obs_df = pd.DataFrame(target_variable)
            if manifest_df is not None:
                manifest_df_c = manifest_df.copy()
                if "Sample" in manifest_df_c.columns:
                    manifest_df_c = manifest_df_c.set_index("Sample")
                if (
                    "platform" not in manifest_df_c.columns
                    and "description" in manifest_df_c.columns
                ):
                    manifest_df_c["platform"] = manifest_df_c["description"].str.extract(
                        r"(HiSeq|NovaSeq|NextSeq)", expand=False
                    )
                # Join common columns, prioritizing 'platform' if available
                join_cols = [c for c in ["platform"] if c in manifest_df_c.columns]
                obs_df = obs_df.join(manifest_df_c[join_cols], how="left")

            adata = sc.AnnData(X=expression_data.astype(float).values, obs=obs_df, dtype=np.float64)
            adata.var_names = expression_data.columns  # Gene IDs
            adata.obs_names = expression_data.index  # Sample IDs

            # Add target group based on median
            median_tg = target_variable.median()
            adata.obs["TG_group"] = [
                "High TG" if x >= median_tg else "Low TG" for x in target_variable
            ]

            logger.debug(f"AnnData object created: {adata.shape}")

        except ValueError as ve:
            msg = f"Error preparing data: {ve}"
            logger.exception(msg)
            return None, None  # Return None tuple on failure
        except Exception as e:
            msg = f"Unexpected error preparing data: {e}"
            logger.exception(msg)
            return None, None

        return adata, target_variable

    def calculate_correlations(
        self,
        adata: sc.AnnData,
        methods: list[str] | None = None,
        target_col: str | None = None,
        run_bootstrap: bool = True,  # Default controlled by config now
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[sc.AnnData, pd.DataFrame]:
        """Calculate gene correlations using specified methods, optionally including bootstrap CIs.

        Args:
            adata: AnnData object containing expression data (samples x genes).
            methods: List of correlation methods to use. Defaults to config.
            target_col: Column name in adata.obs representing the target variable. Defaults to config.
            run_bootstrap: Whether to perform bootstrap analysis for CIs and stability. Overrides config if set.
            bootstrap_iterations: Number of bootstrap iterations. Overrides config if set.
            confidence_level: Confidence level for bootstrap intervals. Overrides config if set.

        Returns:
            Tuple containing the AnnData object and a DataFrame with correlation results,
            potentially including bootstrap statistics.
        """
        # --- Parameter Handling & Config ---
        config = get_correlation_config()
        if target_col is None:
            target_col = config.get("target_gene", "PRODUCT-TG")
        if methods is None:
            methods = config.get("methods", ["spearman"])
        if isinstance(methods, str):
            methods = [methods]

        # Use function args to override config for bootstrap if they are explicitly passed (not None/default)
        # Otherwise, rely on config values fetched below
        run_bootstrap_final = (
            config.get("run_bootstrap", True) if run_bootstrap is True else run_bootstrap
        )
        n_iter_final = (
            config.get("bootstrap_iterations", 1000)
            if bootstrap_iterations == 1000
            else bootstrap_iterations
        )
        conf_level_final = (
            config.get("confidence_level", 0.95) if confidence_level == 0.95 else confidence_level
        )
        top_n_stability_final = config.get("top_n", 50)  # Use general top_n for stability rank

        # --- Validate Methods ---
        methods_list = [m for m in methods if m in self.supported_methods]
        if not methods_list:
            msg = "No valid correlation methods specified."
            raise ValueError(msg)
        invalid_methods = [m for m in methods if m not in self.supported_methods]
        if invalid_methods:
            logger.warning(f"Unsupported methods ignored: {invalid_methods}")

        # --- Validate Target ---
        if target_col not in adata.obs.columns:
            msg = f"Target column '{target_col}' not found in AnnData observation."
            raise ValueError(msg)
        target = adata.obs[target_col].values.astype(float).flatten()
        if np.isnan(target).all():
            msg = "Target variable contains only NaNs."
            raise ValueError(msg)

        valid_target_idx = ~np.isnan(target)
        target_valid = target[valid_target_idx]
        if len(target_valid) < 3:
            msg = f"Not enough non-NaN target values ({len(target_valid)}) for correlation."
            raise ValueError(msg)

        # --- Initialize Results ---
        all_results = []
        num_genes = adata.shape[1]

        # --- Standard Correlation Loop ---
        for method in methods_list:
            logger.info(f"Calculating {method} correlations...")
            start_method_time = time.time()

            # Handle ML methods separately
            if method == "regression":
                # Pass only adata and target_valid to sub-methods if they handle index alignment
                self._calculate_regression(adata[valid_target_idx, :], all_results, target_valid)
                continue
            if method == "decision_tree":
                self._calculate_decision_tree(adata[valid_target_idx, :], all_results, target_valid)
                continue
            if method == "random_forest":
                self._calculate_random_forest(adata[valid_target_idx, :], all_results, target_valid)
                continue

            # Standard correlation methods (batching for performance)
            gene_batch_size = 5000
            for i in range(0, num_genes, gene_batch_size):
                batch_end = min(i + gene_batch_size, num_genes)
                logger.debug(f"  Processing genes {i+1}-{batch_end}...")
                gene_indices_batch = range(i, batch_end)
                # Extract expression data for the batch AND valid target samples
                # Use np.ix_ for safe indexing
                expr_batch_valid = adata.X[np.ix_(valid_target_idx, gene_indices_batch)]

                for j, gene_idx in enumerate(gene_indices_batch):
                    gene_expr_valid = expr_batch_valid[:, j].flatten()
                    gene_id = adata.var_names[gene_idx]

                    # Skip constant genes
                    # Use nanstd which ignores NaNs present in the extracted slice
                    with warnings.catch_warnings():  # Suppress RuntimeWarning for all-NaN slices
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        gene_std = np.nanstd(gene_expr_valid)

                    if gene_std < 1e-9 or pd.isna(gene_std):
                        corr, p_val = 0.0, 1.0
                    else:
                        try:
                            if method == "spearman":
                                corr, p_val = stats.spearmanr(
                                    gene_expr_valid, target_valid, nan_policy="omit"
                                )
                            elif method == "pearson":
                                corr, p_val = stats.pearsonr(gene_expr_valid, target_valid)
                            elif method == "kendall":
                                corr, p_val = stats.kendalltau(
                                    gene_expr_valid, target_valid, nan_policy="omit"
                                )
                            else:
                                corr, p_val = np.nan, 1.0  # Should not happen

                            # Handle NaN results
                            if np.isnan(corr):
                                corr = 0.0
                            if np.isnan(p_val):
                                p_val = 1.0
                        except Exception as stat_err:
                            logger.warning(
                                f"Could not calculate {method} for gene {gene_id}: {stat_err}"
                            )
                            corr, p_val = 0.0, 1.0

                    all_results.append(
                        {
                            "ensembl_transcript_id": gene_id,
                            "Correlation_Type": method.capitalize(),
                            "Correlation_Coefficient": corr,
                            "Correlation_Coefficient_Abs": abs(corr),
                            "Correlation_P_Value": p_val,
                            "Correlation_Rank_Score": abs(corr) / (p_val + 1e-10),
                        }
                    )
            logger.info(
                f"  {method.capitalize()} calculation took {time.time() - start_method_time:.2f} seconds."
            )

        # --- Create Initial DataFrame ---
        correlation_df = pd.DataFrame(all_results)
        if correlation_df.empty:
            logger.warning("Initial correlation analysis produced an empty DataFrame.")
            return adata, correlation_df  # Return empty if no results

        # --- Add Gene Symbols ---
        if (
            hasattr(self, "gene_metadata")
            and self.gene_metadata is not None
            and "sym" in self.gene_metadata.columns
        ):
            try:
                # Correctly handle index name for mapping
                if self.gene_metadata.index.name == "ensembl_transcript_id":
                    meta_for_map = self.gene_metadata
                elif "ensembl_transcript_id" in self.gene_metadata.columns:
                    meta_for_map = self.gene_metadata.set_index("ensembl_transcript_id")
                elif self.gene_metadata.index.name is not None:
                    # Attempt to use index if named, assuming it's the correct ID
                    logger.warning(
                        f"Attempting to use gene_metadata index '{self.gene_metadata.index.name}' for symbol mapping."
                    )
                    meta_for_map = self.gene_metadata
                else:
                    msg = "Cannot determine index for gene metadata mapping."
                    raise ValueError(msg)

                sym_map = meta_for_map["sym"].to_dict()
                correlation_df["sym"] = correlation_df["ensembl_transcript_id"].map(sym_map)
                # Fill missing symbols with the ID itself for complete column
                correlation_df["sym"] = correlation_df["sym"].fillna(
                    correlation_df["ensembl_transcript_id"]
                )
            except Exception as map_err:
                msg = f"Failed to map gene symbols: {map_err}"
                logger.warning(msg)
                correlation_df["sym"] = correlation_df["ensembl_transcript_id"]  # Fallback
        else:
            correlation_df["sym"] = correlation_df[
                "ensembl_transcript_id"
            ]  # Fallback if no sym info

        # --- FDR Correction ---
        logger.info("Applying FDR correction...")
        fdr_col_name = "FDR_Adjusted_P_Value"
        if fdr_col_name not in correlation_df.columns:  # Initialize column if it doesn't exist
            correlation_df[fdr_col_name] = np.nan

        for method in methods_list:
            method_cap = method.capitalize()
            method_mask = correlation_df["Correlation_Type"] == method_cap
            if method_mask.any():  # Check if any rows exist for this method
                p_values = correlation_df.loc[method_mask, "Correlation_P_Value"].values
                # Ensure p-values are numeric and handle NaNs before correction
                if not pd.api.types.is_numeric_dtype(p_values):
                    p_values_clean = pd.to_numeric(p_values, errors="coerce")
                else:
                    p_values_clean = p_values.copy()
                p_values_clean = np.nan_to_num(p_values_clean, nan=1.0)
                p_values_clean = np.clip(
                    p_values_clean, 1e-300, 1.0
                )  # Clip extremely small/zero values

                if len(p_values_clean) > 0:
                    try:
                        reject, adj_p, _, _ = multipletests(p_values_clean, method="fdr_bh")
                        # Use .loc with the mask to assign back correctly
                        correlation_df.loc[method_mask, fdr_col_name] = adj_p
                    except Exception as fdr_err:
                        logger.warning(f"FDR correction failed for {method}: {fdr_err}")
                        # Ensure NaNs are assigned back if FDR fails
                        correlation_df.loc[method_mask, fdr_col_name] = np.nan
                else:
                    logger.debug(f"No valid p-values to correct for method {method_cap}.")
                    correlation_df.loc[method_mask, fdr_col_name] = np.nan

        # --- Bootstrap Analysis (Conditional) ---
        if run_bootstrap_final:  # Use the final decision flag
            try:
                # Filter methods for bootstrap
                bootstrap_methods = config.get("bootstrap_methods", ["pearson", "spearman", "kendall", "regression"])
                bootstrap_methods_list = [m for m in methods_list if m in bootstrap_methods]

                if not bootstrap_methods_list:
                    logger.warning("None of the correlation methods are supported for bootstrap. Skipping bootstrap analysis.")
                else:
                    # If some methods were filtered out, log them
                    excluded_methods = [m for m in methods_list if m not in bootstrap_methods_list]
                    if excluded_methods:
                        logger.info(f"Excluding methods from bootstrap: {excluded_methods}")

                    bootstrap_stats = self.bootstrap_correlation_analysis(
                        adata=adata,  # Pass original adata
                        target_col=target_col,
                        methods=bootstrap_methods_list,  # Use filtered methods list
                        n_iterations=n_iter_final,
                        confidence_level=conf_level_final,
                        top_n_for_stability=top_n_stability_final,
                    )

                    # Merge bootstrap results
                    if bootstrap_stats:
                        bootstrap_dfs = []
                        for method_cap, gene_stats in bootstrap_stats.items():
                            # Ensure the gene IDs from bootstrap keys match the correlation_df ID column
                            df = pd.DataFrame.from_dict(gene_stats, orient="index")
                            df["Correlation_Type"] = method_cap
                            df.index.name = (
                                "ensembl_transcript_id"  # Assume bootstrap keys are ensembl IDs
                            )
                            bootstrap_dfs.append(df.reset_index())

                        if bootstrap_dfs:
                            bootstrap_merged_df = pd.concat(bootstrap_dfs, ignore_index=True)
                            # Merge with the main correlation_df
                            correlation_df = pd.merge(
                                correlation_df,
                                bootstrap_merged_df,
                                on=["ensembl_transcript_id", "Correlation_Type"],
                                how="left",
                            )
                            logger.info(
                                "Successfully merged bootstrap confidence intervals and rank stability."
                            )
                        else:
                            logger.warning("Bootstrap analysis ran but produced no data to merge.")
                    else:
                        logger.warning("Bootstrap analysis did not return statistics.")

            except Exception as boot_err:
                msg = f"Bootstrap analysis failed: {boot_err}"
                logger.exception(msg)
                # Add empty columns if bootstrap fails (important for downstream steps like ranking)
                for method in methods_list:
                    method_cap = method.capitalize()
                    base_col_name = f"Correlation_Coefficient_{method_cap}"  # Base name part
                    if f"{base_col_name}_MeanBoot" not in correlation_df.columns:
                        correlation_df[f"{base_col_name}_MeanBoot"] = np.nan
                    if f"Correlation_CI_Lower_{method_cap}" not in correlation_df.columns:
                        correlation_df[f"Correlation_CI_Lower_{method_cap}"] = np.nan
                    if f"Correlation_CI_Upper_{method_cap}" not in correlation_df.columns:
                        correlation_df[f"Correlation_CI_Upper_{method_cap}"] = np.nan
                    if f"Rank_Stability_{method_cap}_Perc" not in correlation_df.columns:
                        correlation_df[f"Rank_Stability_{method_cap}_Perc"] = (
                            0.0  # Default stability to 0
                        )
        else:
            logger.info("Skipping bootstrap analysis as per configuration/parameter.")
            # Ensure columns exist even if bootstrap is skipped, filled with NaN/0
            for method in methods_list:
                method_cap = method.capitalize()
                base_col_name = f"Correlation_Coefficient_{method_cap}"
                if f"{base_col_name}_MeanBoot" not in correlation_df.columns:
                    correlation_df[f"{base_col_name}_MeanBoot"] = np.nan
                if f"Correlation_CI_Lower_{method_cap}" not in correlation_df.columns:
                    correlation_df[f"Correlation_CI_Lower_{method_cap}"] = np.nan
                if f"Correlation_CI_Upper_{method_cap}" not in correlation_df.columns:
                    correlation_df[f"Correlation_CI_Upper_{method_cap}"] = np.nan
                if f"Rank_Stability_{method_cap}_Perc" not in correlation_df.columns:
                    correlation_df[f"Rank_Stability_{method_cap}_Perc"] = 0.0

        # --- Store Final Results ---
        self.correlation_results = correlation_df
        return adata, correlation_df

    def _calculate_regression(
        self, adata: sc.AnnData, results_list: list[dict[str, Any]], target: np.ndarray
    ) -> None:
        """Calculate linear regression coefficients.

        Args:
            adata: AnnData object with expression data
            results_list: List to store results
            target: Target variable values
        """
        try:
            # Prepare data
            x = adata.X  # Use uppercase X instead of lowercase x

            # Check dimensions for debugging
            logger.debug(f"Regression data shape: X={x.shape}, target={target.shape}")

            # For high-dimensional data, process genes individually instead of all at once
            for i, gene in enumerate(adata.var_names):
                gene_expr = x[:, i].flatten().reshape(-1, 1)

                # Skip genes with zero/negligible variance
                if np.std(gene_expr) < 1e-9:
                    continue

                # Fit simple linear regression for this single gene
                try:
                    model = LinearRegression()
                    model.fit(gene_expr, target)
                    coef = model.coef_[0]  # Single coefficient for single feature

                    # Calculate p-value properly for single variable regression
                    # Add constant term to X for statsmodels
                    x_sm = sm.add_constant(gene_expr)
                    est = sm.OLS(target, x_sm).fit()
                    p_val = est.pvalues[1]  # Skip constant term

                    results_list.append({
                        "ensembl_transcript_id": gene,
                        "Correlation_Type": "Regression",
                        "Correlation_Coefficient": coef,
                        "Correlation_Coefficient_Abs": abs(coef),
                        "Correlation_P_Value": p_val,
                        "Correlation_Rank_Score": abs(coef) / (p_val + 1e-10),
                    })
                except Exception as e:
                    logger.debug(f"Regression for gene {gene} failed: {e}")
                    # Add with fallback values only if we want to show all genes
                    results_list.append({
                        "ensembl_transcript_id": gene,
                        "Correlation_Type": "Regression",
                        "Correlation_Coefficient": 0.0,
                        "Correlation_Coefficient_Abs": 0.0,
                        "Correlation_P_Value": 1.0,
                        "Correlation_Rank_Score": 0.0,
                    })

        except Exception as e:  # Global error handling
            logger.warning(f"Error in regression analysis: {e!s}")

    def _calculate_decision_tree(
        self, adata: sc.AnnData, results_list: list[dict[str, Any]], target: np.ndarray
    ) -> None:
        """Calculate nonlinear relationships using decision trees.

        Args:
            adata: AnnData object with expression data
            results_list: List to store results
            target: Target variable values
        """
        # Limit analysis to top genes by variance for performance
        max_genes = 1000
        logger.info(f"Limiting tree analysis to {max_genes} genes for performance...")

        # Calculate variance for each gene to select most informative genes
        gene_variances = np.var(adata.X, axis=0)
        top_indices = np.argsort(gene_variances)[-max_genes:]  # Top genes by variance

        # Process only the top genes
        for i, gene_idx in enumerate(top_indices):
            gene = adata.var_names[gene_idx]
            gene_expr = adata.X[:, gene_idx].flatten().reshape(-1, 1)

            # Skip constant genes
            if np.all(gene_expr == gene_expr[0]):
                continue

            # Create and train model with simplified cross-validation
            dt = DecisionTreeRegressor(max_depth=3, random_state=42)
            scores = cross_val_score(dt, gene_expr, target, cv=3, scoring="r2")  # Reduced CV folds
            score = max(0, np.mean(scores))  # Ensure non-negative

            # Use a simplified statistical significance estimation
            # Fit on full data to get feature importance instead of permutation tests
            dt.fit(gene_expr, target)
            importance = max(0.001, dt.feature_importances_[0])  # Avoid division by zero

            # Approximate p-value based on score and importance
            # Higher score and importance = lower p-value
            pseudo_p = 1.0 / (1.0 + 10 * score * importance)
            pseudo_p = max(0.001, min(0.99, pseudo_p))  # Keep within reasonable range

            # Store results
            results_list.append(
                {
                    "ensembl_transcript_id": gene,
                    "Correlation_Type": "DecisionTree",
                    "Correlation_Coefficient": score,
                    "Correlation_Coefficient_Abs": score,
                    "Correlation_P_Value": pseudo_p,
                    "Correlation_Rank_Score": score / pseudo_p,
                }
            )

    def _calculate_random_forest(
        self, adata: sc.AnnData, results_list: list[dict[str, Any]], target: np.ndarray
    ) -> None:
        """Calculate nonlinear relationships using random forests.

        Args:
            adata: AnnData object with expression data
            results_list: List to store results
            target: Target variable values
        """
        # Limit analysis to top genes by variance for performance
        max_genes = 1000
        logger.info(f"Limiting random forest analysis to {max_genes} genes for performance...")

        # Calculate variance for each gene to select most informative genes
        gene_variances = np.var(adata.X, axis=0)
        top_indices = np.argsort(gene_variances)[-max_genes:]  # Top genes by variance

        # Process only the top genes
        for i, gene_idx in enumerate(top_indices):
            gene = adata.var_names[gene_idx]
            gene_expr = adata.X[:, gene_idx].flatten().reshape(-1, 1)

            # Skip constant genes
            if np.all(gene_expr == gene_expr[0]):
                continue

            # Create and train model with fewer trees and simplified cross-validation
            rf = RandomForestRegressor(
                n_estimators=20, max_depth=3, random_state=42
            )  # Reduced trees
            scores = cross_val_score(rf, gene_expr, target, cv=3, scoring="r2")  # Reduced CV folds
            score = max(0, np.mean(scores))  # Ensure non-negative

            # Use a simplified statistical significance estimation
            # Fit on full data to get feature importance instead of permutation tests
            rf.fit(gene_expr, target)
            importance = max(0.001, rf.feature_importances_[0])  # Avoid division by zero

            # Approximate p-value based on score and importance
            # Higher score and importance = lower p-value
            pseudo_p = 1.0 / (1.0 + 10 * score * importance)
            pseudo_p = max(0.001, min(0.99, pseudo_p))  # Keep within reasonable range

            # Store results with CORRECT CAPITALIZATION - use Random_forest to match expected format
            results_list.append(
                {
                    "ensembl_transcript_id": gene,
                    "Correlation_Type": "Random_forest",  # Changed from RandomForest to Random_forest
                    "Correlation_Coefficient": score,
                    "Correlation_Coefficient_Abs": score,
                    "Correlation_P_Value": pseudo_p,
                    "Correlation_Rank_Score": score / pseudo_p,
                }
            )

    def bootstrap_correlation_analysis(
        self,
        adata: AnnData,
        target_col: str,
        methods: list[str],
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        top_n_for_stability: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Performs bootstrap resampling to estimate confidence intervals for correlations."""
        logger.info(f"Starting bootstrap analysis ({n_iterations} iterations)...")
        if target_col not in adata.obs.columns:
            msg = f"Target column '{target_col}' not found in AnnData observation."
            raise ValueError(msg)

        n_samples = adata.shape[0]
        n_genes = adata.shape[1]
        lower_percentile = (1.0 - confidence_level) / 2.0 * 100
        upper_percentile = (1.0 + confidence_level) / 2.0 * 100

        # Store results: shape (n_genes, n_methods, n_iterations)
        bootstrap_results = np.full((n_genes, len(methods), n_iterations), np.nan)
        # Store ranks: Dictionary to track how often a gene appears in top N PER ITERATION
        # FIX: Initialize overall count, will be incremented correctly below
        rank_stability_counts = {gene_id: 0 for gene_id in adata.var_names}

        target_all = adata.obs[target_col].values.astype(float).flatten()
        valid_target_idx_all = ~np.isnan(target_all)
        target_all_valid = target_all[valid_target_idx_all]
        n_valid_samples = len(target_all_valid)

        if n_valid_samples < 5:
            logger.warning("Too few valid samples for meaningful bootstrapping.")
            return {}  # Return empty if not enough data

        original_indices = np.arange(n_valid_samples)

        # --- Setup Rich Progress Bar ---
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Items: [cyan]{task.completed}/{task.total}"),
            TextColumn("• Elapsed:"),
            TimeElapsedColumn(),
            TextColumn("• Remaining:"),
            TimeRemainingColumn(),
        ]

        # --- Bootstrap Loop with Progress Bar ---
        with Progress(*progress_columns, transient=False) as progress:
            task_id = progress.add_task("[cyan]Bootstrap Iterations", total=n_iterations)

            for iteration in range(n_iterations):
                # --- Resampling ---
                resampled_indices = np.random.choice(
                    original_indices, n_valid_samples, replace=True
                )
                target_resampled = target_all_valid[resampled_indices]
                original_valid_indices = np.where(valid_target_idx_all)[0]
                adata_resampled_indices = original_valid_indices[resampled_indices]
                if np.std(target_resampled) < 1e-9:
                    progress.update(task_id, advance=1)  # Advance progress even if skipping
                    continue
                expr_resampled = adata.X[adata_resampled_indices, :]

                # --- Calculate Correlations for this iteration ---
                iter_corrs = {}  # Store correlations for this iteration: {gene_idx: {method: corr}}
                for gene_idx in range(n_genes):
                    gene_expr_resampled = expr_resampled[:, gene_idx].flatten()
                    with warnings.catch_warnings():  # Suppress RuntimeWarning for all-NaN slices
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        gene_std = np.nanstd(gene_expr_resampled)

                    if gene_std < 1e-9 or pd.isna(gene_std):
                        for method_idx in range(len(methods)):
                            bootstrap_results[gene_idx, method_idx, iteration] = np.nan
                        continue

                    iter_corrs[gene_idx] = {}  # Initialize dict for this gene's results
                    for method_idx, method in enumerate(methods):
                        corr = np.nan
                        try:
                            if method.lower() == "spearman":
                                corr, _ = stats.spearmanr(
                                    gene_expr_resampled, target_resampled, nan_policy="omit"
                                )
                            elif method.lower() == "pearson":
                                corr, _ = stats.pearsonr(gene_expr_resampled, target_resampled)
                            elif method.lower() == "kendall":
                                corr, _ = stats.kendalltau(
                                    gene_expr_resampled, target_resampled, nan_policy="omit"
                                )
                            elif method.lower() in ["regression", "random_forest"]:
                                # For ML methods, use simple linear regression or R² measure
                                # as proxy for correlation strength during bootstrapping
                                model = LinearRegression()
                                model.fit(gene_expr_resampled.reshape(-1, 1), target_resampled)
                                corr = model.coef_[0] if method.lower() == "regression" else np.corrcoef(gene_expr_resampled, target_resampled)[0, 1]
                            else:
                                logger.warning(
                                    f"Unsupported method '{method}' encountered during bootstrap."
                                )
                                corr = np.nan

                            if np.isnan(corr):
                                corr = 0.0

                        except Exception:
                            corr = np.nan

                        bootstrap_results[gene_idx, method_idx, iteration] = corr
                        iter_corrs[gene_idx][method] = corr

                # --- Rank Stability Calculation ---
                # Use a set to track genes that hit the top N in *this iteration*
                genes_in_top_n_this_iteration = set()
                for method_idx, method in enumerate(methods):
                    # Get correlations calculated *for this iteration*
                    method_corrs_iter = [
                        (gene_idx, iter_corrs.get(gene_idx, {}).get(method, np.nan))
                        for gene_idx in range(n_genes)
                    ]
                    # Filter out genes with NaN correlation for this method in this iteration
                    valid_corrs_iter = [(idx, c) for idx, c in method_corrs_iter if pd.notna(c)]
                    if not valid_corrs_iter:
                        continue  # Skip method if no valid correlations

                    # Sort genes by absolute correlation magnitude for this method
                    sorted_genes_iter = sorted(
                        valid_corrs_iter, key=lambda item: abs(item[1]), reverse=True
                    )

                    # Add genes in the top N for this method to the set for this iteration
                    for rank, (gene_idx, _) in enumerate(sorted_genes_iter[:top_n_for_stability]):
                        genes_in_top_n_this_iteration.add(gene_idx)

                # Increment the overall count *once* for each gene that appeared in the top N
                # for *any* method during this iteration.
                for gene_idx in genes_in_top_n_this_iteration:
                    gene_id = adata.var_names[gene_idx]
                    rank_stability_counts[gene_id] += 1  # Increment overall count

                # --- Update Progress Bar ---
                progress.update(task_id, advance=1)

        # --- Calculate CIs and Stability ---
        bootstrap_stats = {}
        logger.info("Calculating final bootstrap statistics...")
        with Progress(*progress_columns, transient=False) as final_progress:
            final_task = final_progress.add_task(
                "[cyan]Final Stats Calculation", total=len(methods) * n_genes
            )
            for method_idx, method in enumerate(methods):
                method_cap = method.capitalize()
                method_results = {}
                for gene_idx in range(n_genes):
                    gene_id = adata.var_names[gene_idx]
                    gene_bootstrap_values = bootstrap_results[gene_idx, method_idx, :]
                    valid_bootstrap_values = gene_bootstrap_values[~np.isnan(gene_bootstrap_values)]

                    if (
                        len(valid_bootstrap_values) > n_iterations * 0.1
                    ):  # Min valid iterations check
                        ci_lower = np.percentile(valid_bootstrap_values, lower_percentile)
                        ci_upper = np.percentile(valid_bootstrap_values, upper_percentile)
                        mean_corr = np.mean(valid_bootstrap_values)
                        # FIX: Calculate percentage correctly and cap at 100
                        stability_pct = min(
                            100.0, (rank_stability_counts.get(gene_id, 0) / n_iterations) * 100
                        )

                        method_results[gene_id] = {
                            f"Correlation_Coefficient_{method_cap}_MeanBoot": mean_corr,
                            f"Correlation_CI_Lower_{method_cap}": ci_lower,
                            f"Correlation_CI_Upper_{method_cap}": ci_upper,
                            f"Rank_Stability_{method_cap}_Perc": stability_pct,  # Use corrected value
                        }
                    else:
                        method_results[gene_id] = {
                            f"Correlation_Coefficient_{method_cap}_MeanBoot": np.nan,
                            f"Correlation_CI_Lower_{method_cap}": np.nan,
                            f"Correlation_CI_Upper_{method_cap}": np.nan,
                            f"Rank_Stability_{method_cap}_Perc": 0.0,
                        }
                    final_progress.update(final_task, advance=1)  # Update inner progress

                bootstrap_stats[method_cap] = method_results

        logger.info("Bootstrap analysis finished.")
        return bootstrap_stats

    def visualize_correlation_results(
        self, top_n: int = 10, method: str | None = None, figsize: tuple[int, int] = (12, 10)
    ) -> Figure:
        """Visualize correlation results.

        Args:
            top_n: Number of top genes to visualize
            method: Correlation method to visualize (if None, uses the best method)
            figsize: Figure size

        Returns:
            matplotlib figure

        Raises:
            ValueError: If correlation results are not available.
        """
        if self.correlation_results is None:
            msg = "Correlation results are not available"
            raise ValueError(msg)

        # Set plot style
        plt.style.use(get_visualization_config().get("style", "seaborn-v0_8-whitegrid"))

        # If method is not specified, use the default method from config
        if method is None:
            method = get_correlation_config().get("default_method", "spearman") or "spearman"

        # Always ensure method is a string before capitalizing
        method = str(method).capitalize()

        # Filter results by method
        method_results = self.correlation_results[
            self.correlation_results["Correlation_Type"] == method
        ]

        if method_results.empty:
            msg = f"No results found for method '{method}'. Available methods: {self.correlation_results['Correlation_Type'].unique()}"
            raise ValueError(msg)

        # Get top genes by rank score
        top_genes = method_results.sort_values(by="Correlation_Rank_Score", ascending=False).head(
            top_n
        )

        # Create a combined visualization
        fig, axes = plt.subplots(
            2, 1, figsize=figsize, dpi=120, gridspec_kw={"height_ratios": [1, 2]}
        )

        # 1. Bar plot of top genes
        ax1 = axes[0]

        # Use gene symbols if available, otherwise use IDs
        if "sym" in top_genes.columns:
            labels = top_genes["sym"].values
        else:
            labels = top_genes["ensembl_transcript_id"].values

        # Create bar plot with custom coloring based on correlation coefficient
        colors = ["red" if val < 0 else "blue" for val in top_genes["Correlation_Coefficient"]]

        # Plot sorted by absolute correlation value
        sorted_indices = np.argsort(np.abs(top_genes["Correlation_Coefficient"].values))
        bars = ax1.barh(
            range(len(labels)),
            np.abs(top_genes["Correlation_Coefficient"].values)[sorted_indices],
            color=[colors[i] for i in sorted_indices],
        )

        # Add labels and titles
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels([labels[i] for i in sorted_indices])
        ax1.set_xlabel(f"{method} Correlation Coefficient (Absolute Value)")
        ax1.set_title(f"Top {top_n} Genes Correlated with Target Gene (by {method})")

        legend_elements = [
            Patch(facecolor="blue", label="Positive Correlation"),
            Patch(facecolor="red", label="Negative Correlation"),
        ]
        ax1.legend(handles=legend_elements, loc="lower right")

        # Add p-values as text
        for i, (idx, bar) in enumerate(zip(sorted_indices, bars, strict=False)):
            p_val = top_genes["Correlation_P_Value"].values[idx]
            p_val_text = f"p={p_val:.2e}" if p_val < 0.05 else f"p={p_val:.2f}"
            ax1.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                p_val_text,
                va="center",
                fontsize=8,
            )

        # 2. Heatmap of top genes vs all genes
        ax2 = axes[1]

        # Get correlation matrix of top genes with all other genes
        top_gene_ids = top_genes["ensembl_transcript_id"].values

        # Convert adata back to dataframe for correlation matrix
        if hasattr(self, "gene_metadata"):
            # Use gene symbols for the heatmap if available
            gene_id_to_symbol = dict(zip(top_gene_ids, top_genes["sym"].values, strict=False))
            labels = [gene_id_to_symbol.get(gene_id, gene_id) for gene_id in top_gene_ids]
        else:
            labels = top_gene_ids

        # Generate a correlation matrix for the top genes
        corr_matrix = np.zeros((len(top_gene_ids), len(top_gene_ids)))

        # Get all possible pairs of the top genes
        for i, gene1 in enumerate(top_gene_ids):
            for j, gene2 in enumerate(top_gene_ids):
                # Set diagonal to 1
                if i == j:
                    corr_matrix[i, j] = 1
                else:
                    # Get gene-gene correlation
                    gene1_data = method_results[method_results["ensembl_transcript_id"] == gene1]
                    gene2_data = method_results[method_results["ensembl_transcript_id"] == gene2]

                    if not gene1_data.empty and not gene2_data.empty:
                        # Use correlation coefficients
                        corr_matrix[i, j] = (
                            gene1_data["Correlation_Coefficient"].values[0]
                            * gene2_data["Correlation_Coefficient"].values[0]
                        )

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax2,
        )

        ax2.set_title(f"Co-correlation Matrix of Top {top_n} Genes")

        plt.tight_layout()

        # Store figure for later reference
        self.figures["correlation_results"] = fig

        return fig

    def run_analysis(
        self,
        expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame | None = None,
        methods: list[str] | None = None,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """Run the full correlation analysis pipeline."""
        # Parameter handling (ensure defaults are applied if None)
        config = get_correlation_config()
        if methods is None:
            methods = config.get("methods", ["spearman"])
        if top_n is None:
            top_n = config.get("top_n", 20)
        target_col = config.get("target_gene", "PRODUCT-TG")
        if isinstance(methods, str):
            methods = [methods]

        logger.info(
            f"Running correlation analysis: target='{target_col}', methods={methods}, top_n={top_n}"
        )

        # Initialize results dictionary with None/empty defaults
        results: dict[str, Any] = {
            "target_gene": target_col,
            "correlation_results": pd.DataFrame(),
            "adata": None,
            "method_performance": None,
            "top_genes_by_method": {},
            "consensus_genes": None,
            "best_method": None,
            "visualization": None,
        }

        try:
            # Prepare data - this creates adata
            adata, _ = self.prepare_data(expression_df, manifest_df, target_col)
            if adata is None:
                msg = "Data preparation failed, AnnData object is None."
                raise ValueError(msg)
            results["adata"] = adata  # Store adata immediately

            # Calculate correlations
            _, correlation_df = self.calculate_correlations(adata, methods, target_col)
            results["correlation_results"] = correlation_df  # Store correlation df

            # Only proceed with downstream analysis if correlations were successful
            if not correlation_df.empty:
                results["method_performance"] = self.compare_methods(
                    adata, correlation_df, top_n=int(top_n / 2)
                )

                for method in methods:
                    method_cap = method.capitalize()
                    method_df_subset = correlation_df[
                        correlation_df["Correlation_Type"] == method_cap
                    ]
                    if not method_df_subset.empty:
                        sort_key = (
                            "Correlation_Rank_Score"
                            if "Correlation_Rank_Score" in method_df_subset.columns
                            else "Correlation_Coefficient_Abs"
                        )
                        if sort_key not in method_df_subset.columns:
                            sort_key = "Correlation_Coefficient_Abs"
                        results["top_genes_by_method"][method] = method_df_subset.sort_values(
                            sort_key, ascending=False
                        ).head(top_n)

                if len(methods) > 1:
                    results["consensus_genes"] = self.find_consensus_markers(
                        correlation_df, methods, top_n=top_n
                    )

                # Determine best method
                if (
                    results["method_performance"] is not None
                    and not results["method_performance"].empty
                ):
                    results["best_method"] = str(results["method_performance"].iloc[0]["Method"])
                else:
                    results["best_method"] = methods[0].capitalize() if methods else None

        except ValueError as ve:  # Catch ValueErrors from prepare/calculate
            msg = f"Correlation analysis failed: {ve}"
            logger.exception(msg)
            # Results dict already has defaults, including empty df and None adata
        except Exception as e:
            msg = f"Unexpected error during correlation analysis: {e}"
            logger.exception(msg)
            # Ensure critical results are reset on unexpected error
            results["correlation_results"] = pd.DataFrame()
            results["adata"] = None

        return results

    def compare_methods(
        self,
        adata: sc.AnnData,
        correlation_df: pd.DataFrame,
        top_n: int = 20,
        # fig_size: tuple[int, int] = (12, 8),
    ) -> pd.DataFrame | None:  # Updated return type to include None
        """Compare correlation methods and identify the best performers.

        Args:
            adata: AnnData object with expression data
            correlation_df: DataFrame with correlation results
            top_n: Number of top genes to use for evaluation
            fig_size: Size of the figure

        Returns:
            DataFrame with performance metrics or None if fewer than 2 methods.
        """
        # Get unique methods
        methods = correlation_df["Correlation_Type"].unique()

        if len(methods) < 2:
            logger.info("Need at least 2 methods to compare")
            return None

        # Collect top genes and performance metrics for each method
        performance = []
        top_genes_by_method = {}

        for method in methods:
            # Get top genes
            method_df = correlation_df[correlation_df["Correlation_Type"] == method]
            top_genes = method_df.sort_values("Correlation_Rank_Score", ascending=False).head(top_n)
            top_genes_by_method[method] = top_genes

            # Calculate metrics
            # 1. Average correlation strength
            mean_corr = top_genes["Correlation_Coefficient_Abs"].mean()

            # 2. Statistical significance
            mean_sig = np.nan
            if (
                "Correlation_P_Value" in top_genes.columns
                and not top_genes["Correlation_P_Value"].isna().all()
            ):
                # Use -log10(p) for better scale
                epsilon = 1e-300
                mean_sig = -np.log10(top_genes["Correlation_P_Value"].fillna(1.0) + epsilon).mean()

            # 3. Predictive power (ensure it's positive)
            pred_power = 0.0
            try:
                # Get expression data for top genes
                gene_ids = top_genes["ensembl_transcript_id"].tolist()
                gene_indices = [
                    list(adata.var_names).index(gene)
                    for gene in gene_ids
                    if gene in adata.var_names
                ]

                if gene_indices:
                    x = adata.X[:, gene_indices]
                    y = adata.obs[adata.obs.columns[0]].values

                    # Calculate R² through cross-validation
                    model = LinearRegression()
                    scores = cross_val_score(model, x, y, cv=5, scoring="r2")
                    pred_power = max(0.0, np.mean(scores))
            except Exception as e:
                logger.info(f"Error calculating predictive power for {method}: {e!s}")

            # Store results
            performance.append(
                {
                    "Method": method,
                    "Mean_Correlation": mean_corr,
                    "Mean_Significance": mean_sig,
                    "Predictive_Power": pred_power,
                }
            )

        # Create performance DataFrame
        perf_df = pd.DataFrame(performance)

        # If DataFrame is empty, return None
        if perf_df.empty:
            return None

        # Calculate overall score (normalize and average metrics)
        score_cols = ["Mean_Correlation", "Mean_Significance", "Predictive_Power"]
        for col in score_cols:
            if col in perf_df.columns and not perf_df[col].isna().all():
                # Min-max normalization
                min_val = perf_df[col].min()
                max_val = perf_df[col].max()
                if max_val > min_val:
                    perf_df[f"{col}_Norm"] = (perf_df[col] - min_val) / (max_val - min_val)
                else:
                    perf_df[f"{col}_Norm"] = 0.5
            else:
                perf_df[f"{col}_Norm"] = np.nan

        # Calculate overall score (weighted average of normalized metrics)
        norm_cols = [c for c in perf_df.columns if c.endswith("_Norm")]
        valid_cols = [c for c in norm_cols if not perf_df[c].isna().all()]

        if valid_cols:
            perf_df["Overall_Score"] = perf_df[valid_cols].mean(axis=1)
            perf_df = perf_df.sort_values("Overall_Score", ascending=False)

        # Clean up for display
        display_cols = ["Method", "Mean_Correlation", "Mean_Significance", "Predictive_Power"]
        if "Overall_Score" in perf_df.columns:
            display_cols.append("Overall_Score")

        return perf_df[display_cols].round(3)

    def find_consensus_markers(
        self,
        correlation_df: pd.DataFrame,
        methods: list[str] | None = None,
        min_correlation: float | None = None,
        max_p_value: float | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Find genes consistently identified as markers across multiple methods."""
        empty_df_columns = [
            "gene_id",
            "symbol",
            "methods_count",
            "avg_correlation",
            "avg_abs_correlation",
            "max_abs_correlation",
            "same_direction",
            "consensus_score",
        ]

        if correlation_df is None or correlation_df.empty:
            logger.warning("Consensus: Input correlation_df is empty.")
            return pd.DataFrame(columns=empty_df_columns)

        # Default parameters from config if not specified
        config = get_correlation_config()
        if min_correlation is None:
            min_correlation = config.get("min_correlation", 0.5)
        if max_p_value is None:
            max_p_value = config.get("max_p_value", 0.05)

        # Check methods
        available_methods_in_df = correlation_df["Correlation_Type"].unique().tolist()
        if not available_methods_in_df:
            logger.error("Consensus: No 'Correlation_Type' found in correlation_df.")
            return pd.DataFrame(columns=empty_df_columns)

        if methods is None:
            methods_to_use = available_methods_in_df
        else:
            # Filter requested methods against available ones (case-insensitive)
            methods_lower = {m.lower() for m in methods}
            methods_to_use = [
                df_method
                for df_method in available_methods_in_df
                if df_method.lower() in methods_lower
            ]

        if not methods_to_use:
            msg = (
                "Consensus: None of the requested methods found in correlation data. "
                f"Requested: {methods}, Available: {available_methods_in_df}"
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(methods_to_use) < len(methods or []):
            ignored = set(methods or []) - {m.lower() for m in methods_to_use}
            logger.warning(
                f"Consensus: Using subset of requested methods found in data: {methods_to_use}. "
                f"Ignoring: {ignored}"
            )

        # Determine gene ID source and symbol mapping
        gene_id_col = None
        if "ensembl_transcript_id" in correlation_df.columns:
            gene_id_col = "ensembl_transcript_id"
        elif correlation_df.index.name:
            gene_id_col = correlation_df.index.name
            logger.info(f"Consensus: Using index '{gene_id_col}' as gene ID.")
        else:
            logger.error("Consensus: Cannot determine gene ID column or index in correlation_df.")
            return pd.DataFrame(columns=empty_df_columns)
        is_id_index = gene_id_col == correlation_df.index.name

        gene_symbol_map = {}
        if "sym" in correlation_df.columns:
            map_df = (
                correlation_df[[gene_id_col, "sym"]].dropna().drop_duplicates(subset=[gene_id_col])
            )
            if is_id_index:
                gene_symbol_map = pd.Series(map_df["sym"].values, index=map_df.index).to_dict()
            else:
                gene_symbol_map = pd.Series(
                    map_df["sym"].values, index=map_df[gene_id_col]
                ).to_dict()

        # Collect significant genes for each method
        method_genes = {}
        for method_name in methods_to_use:
            method_df = correlation_df[correlation_df["Correlation_Type"] == method_name]
            sig_genes = method_df[
                (method_df["Correlation_Coefficient_Abs"] >= min_correlation)
                & (method_df["Correlation_P_Value"] <= max_p_value)
            ]

            if not sig_genes.empty:
                if is_id_index:
                    method_genes[method_name] = dict(
                        zip(sig_genes.index, sig_genes["Correlation_Coefficient"], strict=False)
                    )
                else:
                    method_genes[method_name] = dict(
                        zip(
                            sig_genes[gene_id_col],
                            sig_genes["Correlation_Coefficient"],
                            strict=False,
                        )
                    )
            else:
                method_genes[method_name] = {}

        # Find unique genes across all methods
        all_genes = set().union(*[genes.keys() for genes in method_genes.values()])
        if not all_genes:
            logger.warning("Consensus: No significant genes found across any used method.")
            return pd.DataFrame(columns=empty_df_columns)

        # Create consensus scoring
        consensus_data = []
        min_methods_required = max(1, len(methods_to_use) // 2)  # Need at least 1 method, or half

        for gene in all_genes:
            methods_found = [method for method, genes in method_genes.items() if gene in genes]
            methods_count = len(methods_found)

            if methods_count >= min_methods_required:
                correlations = [method_genes[method][gene] for method in methods_found]
                # Handle potential NaN correlation values if any slipped through
                correlations_clean = [c for c in correlations if pd.notna(c)]
                if not correlations_clean:
                    continue

                avg_correlation = np.mean(correlations_clean)
                avg_abs_correlation = np.mean(np.abs(correlations_clean))
                max_abs_correlation = np.max(np.abs(correlations_clean))
                same_dir = all(c > 0 for c in correlations_clean) or all(
                    c < 0 for c in correlations_clean
                )

                # Robust consensus score calculation
                consensus_score = methods_count * avg_abs_correlation
                if same_dir:
                    consensus_score *= 1.2  # Bonus for same direction
                else:
                    consensus_score *= 0.8  # Penalty for mixed direction

                gene_symbol = gene_symbol_map.get(gene, str(gene))

                consensus_data.append(
                    {
                        "gene_id": gene,
                        "symbol": gene_symbol,
                        "methods_count": methods_count,
                        "avg_correlation": avg_correlation,
                        "avg_abs_correlation": avg_abs_correlation,
                        "max_abs_correlation": max_abs_correlation,
                        "same_direction": same_dir,
                        "consensus_score": consensus_score,
                    }
                )

        if not consensus_data:
            logger.warning("No genes met the cross-method consensus criteria.")
            return pd.DataFrame(columns=empty_df_columns)

        consensus_df = pd.DataFrame(consensus_data).sort_values("consensus_score", ascending=False)
        return consensus_df.head(top_n)

    def analyze_platform_consensus(
        self,
        expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        target_col: str = "PRODUCT-TG",
        top_n: int = 100,
        min_correlation: float = 0.5,
        max_p_value: float = 0.05,
    ) -> pd.DataFrame:
        """Analyze correlation consistency across different sequencing platforms."""
        logger.info("Analyzing platform consensus for marker genes...")
        manifest_df = manifest_df.copy()
        platform_col = "platform"
        # Ensure platform column exists or derive it
        if platform_col not in manifest_df.columns and "description" in manifest_df.columns:
            manifest_df[platform_col] = manifest_df["description"].str.extract(
                r"(HiSeq|NovaSeq|NextSeq)", expand=False
            )
        elif platform_col not in manifest_df.columns:
            logger.error(
                "Cannot determine platform: 'platform' or 'description' column missing in manifest."
            )
            return pd.DataFrame()
        # Ensure manifest index is Sample if needed for lookup
        if "Sample" in manifest_df.columns and manifest_df.index.name != "Sample":
            manifest_df = manifest_df.set_index(
                "Sample", drop=False
            )  # Keep Sample column if needed elsewhere

        platforms = manifest_df[platform_col].dropna().unique()
        if len(platforms) < 2:
            logger.warning("Need at least 2 platforms for cross-platform analysis. Skipping.")
            return pd.DataFrame()
        logger.info(f"Found {len(platforms)} platforms: {', '.join(platforms)}")

        platform_results_dfs: dict[str, pd.DataFrame] = {}  # Store resulting DataFrames

        # Determine expression data structure
        gene_id_col = (
            "ensembl_transcript_id"
            if "ensembl_transcript_id" in expression_df.columns
            else (expression_df.index.name or "index_id")
        )
        is_index_id = gene_id_col == (expression_df.index.name or "index_id")
        meta_cols = [
            c
            for c in ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
            if c in expression_df.columns and not is_index_id
        ]
        sample_cols = [c for c in expression_df.columns if c not in meta_cols]

        for platform in platforms:
            logger.info(f"Analyzing platform: {platform}")
            # Get sample IDs for this platform from manifest index
            platform_sample_ids = manifest_df[manifest_df[platform_col] == platform].index.tolist()
            # Find which of these sample IDs actually exist as columns in the expression data
            valid_platform_samples = [s for s in platform_sample_ids if s in sample_cols]

            if len(valid_platform_samples) < 5:
                logger.warning(
                    f"Skipping platform {platform}: Too few valid samples ({len(valid_platform_samples)})."
                )
                continue

            # Create subset DataFrame including metadata and valid samples
            if is_index_id:
                platform_df_subset = expression_df.loc[
                    :, valid_platform_samples
                ].copy()  # Select columns
            else:
                platform_df_subset = expression_df[meta_cols + valid_platform_samples].copy()

            # Create manifest subset for this platform's valid samples
            manifest_subset = manifest_df.loc[valid_platform_samples]

            try:
                # Use prepare_data for the subset
                adata_platform, _ = self.prepare_data(
                    expression_df=platform_df_subset,
                    manifest_df=manifest_subset,
                    target_col=target_col,
                )
                if adata_platform is None:
                    msg = "Failed to prepare AnnData for platform subset."
                    raise ValueError(msg)

                # Calculate correlations for the subset
                _, platform_corr_df = self.calculate_correlations(
                    adata=adata_platform,
                    methods="spearman",  # Use consistent method
                    target_col=target_col,
                )
                if platform_corr_df is not None and not platform_corr_df.empty:
                    platform_results_dfs[platform] = platform_corr_df
                else:
                    msg = f"Correlation calculation returned no results for platform {platform}."
                    logger.warning(msg)

            except ValueError as ve:
                msg = f"Error analyzing platform {platform}: {ve}"
                logger.exception(msg)
            except Exception as e:
                msg = f"Unexpected error analyzing platform {platform}: {e}"
                logger.exception(msg)

        # --- Consensus calculation ---
        if len(platform_results_dfs) < 2:
            logger.warning(
                "Not enough platforms with valid correlation results for consensus analysis."
            )
            return pd.DataFrame()

        # Collect significant genes per platform
        platform_genes = {}
        for platform, corr_df in platform_results_dfs.items():
            corr_df_filtered = corr_df[corr_df["Correlation_Type"] == "Spearman"]
            sig_genes = corr_df_filtered[
                (corr_df_filtered["Correlation_Coefficient_Abs"] >= min_correlation)
                & (corr_df_filtered["Correlation_P_Value"] <= max_p_value)
            ]
            # Use the gene ID column identified earlier or index
            id_col_runtime = (
                "ensembl_transcript_id"
                if "ensembl_transcript_id" in sig_genes.columns
                else sig_genes.index.name
            )
            if id_col_runtime:
                platform_genes[platform] = dict(
                    zip(
                        sig_genes[id_col_runtime],
                        sig_genes["Correlation_Coefficient"],
                        strict=False,
                    )
                )
            else:  # Fallback if index is unnamed but contains IDs
                platform_genes[platform] = dict(
                    zip(sig_genes.index, sig_genes["Correlation_Coefficient"], strict=False)
                )

        # Find consensus
        all_genes = set().union(*[genes.keys() for genes in platform_genes.values()])
        if not all_genes:
            logger.warning("No significant genes found across platforms to calculate consensus.")
            return pd.DataFrame()

        consensus_data = []
        min_platforms = max(2, len(platform_genes) // 2)

        # Build symbol map from original data if possible
        gene_symbol_map = {}
        if not is_index_id and "sym" in expression_df.columns:
            gene_symbol_map = pd.Series(
                expression_df["sym"].values, index=expression_df[gene_id_col]
            ).to_dict()

        for gene_id in all_genes:
            platforms_found = [p for p, genes in platform_genes.items() if gene_id in genes]
            if len(platforms_found) >= min_platforms:
                correlations = [platform_genes[p][gene_id] for p in platforms_found]
                avg_corr = np.mean(correlations)
                avg_abs_corr = np.mean(np.abs(correlations))
                same_dir = all(c > 0 for c in correlations) or all(c < 0 for c in correlations)
                score = len(platforms_found) * avg_abs_corr * (1.2 if same_dir else 0.8)
                symbol = gene_symbol_map.get(gene_id, str(gene_id))  # Use map or fallback

                consensus_data.append(
                    {
                        "gene_id": gene_id,
                        "symbol": symbol,
                        "platforms_count": len(platforms_found),
                        "avg_correlation": avg_corr,
                        "avg_abs_correlation": avg_abs_corr,
                        "same_direction": same_dir,
                        "consensus_score": score,
                    }
                )

        if not consensus_data:
            logger.warning("No genes met the cross-platform consensus criteria.")
            return pd.DataFrame()

        consensus_df = pd.DataFrame(consensus_data).sort_values("consensus_score", ascending=False)
        logger.info(f"Generated consensus ranking for {len(consensus_df)} genes.")
        return consensus_df.head(top_n)

    def analyze_method_consensus_overlap(
        self,
        correlation_df: pd.DataFrame,
        consensus_df: pd.DataFrame,
        top_n: int = 50,
    ) -> dict:
        """Analyze overlap between different correlation methods and consensus ranking.

        Args:
            correlation_df: Correlation results
            consensus_df: Consensus ranking results
            top_n: Number of top genes to compare

        Returns:
            Dictionary with overlap statistics and visualizations
        """
        # Get unique methods
        methods = correlation_df["Correlation_Type"].unique().tolist()

        # Extract top genes for each method
        method_top_genes = {}
        for method in methods:
            method_df = correlation_df[correlation_df["Correlation_Type"] == method]
            method_df = method_df.sort_values("Correlation_Coefficient_Abs", ascending=False)

            # Get top gene IDs
            if "ensembl_transcript_id" in method_df.columns:
                method_top_genes[method] = method_df["ensembl_transcript_id"].head(top_n).tolist()
            else:
                method_top_genes[method] = method_df.index[:top_n].tolist()

        # Get consensus top genes
        if "gene_id" in consensus_df.columns:
            consensus_top_genes = consensus_df["gene_id"].head(top_n).tolist()
        else:
            consensus_top_genes = consensus_df.index[:top_n].tolist()

        # Calculate overlap statistics
        overlap_stats = {}
        for method in methods:
            method_genes = set(method_top_genes[method])
            consensus_genes = set(consensus_top_genes)

            # Find overlap
            overlap_genes = method_genes.intersection(consensus_genes)
            method_only_genes = method_genes - consensus_genes
            consensus_only_genes = consensus_genes - method_genes

            # Calculate statistics
            overlap_count = len(overlap_genes)
            overlap_percentage = (overlap_count / len(method_genes)) * 100

            # Jaccard similarity
            jaccard_similarity = len(overlap_genes) / len(method_genes.union(consensus_genes))

            # Store results
            overlap_stats[method] = {
                "overlap_genes": list(overlap_genes),
                "method_only_genes": list(method_only_genes),
                "consensus_only_genes": list(consensus_only_genes),
                "overlap_count": overlap_count,
                "method_only_count": len(method_only_genes),
                "consensus_only_count": len(consensus_only_genes),
                "overlap_percentage": overlap_percentage,
                "jaccard_similarity": jaccard_similarity,
            }

        # Return all results
        return {
            "overlap_stats": overlap_stats,
            "method_top_genes": method_top_genes,
            "consensus_top_genes": consensus_top_genes,
        }

    def visualize_consensus_ranking(
        self,
        consensus_df: pd.DataFrame,
        top_n: int = 15,
        figsize: tuple[int, int] = (12, 9),  # Adjusted size
    ) -> Figure:
        """Visualize consensus gene ranking."""
        if consensus_df.empty:
            # ... (error handling remains the same) ...
            logger.warning("No consensus data to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No consensus data available", ha="center", va="center")
            return fig

        plot_df = consensus_df.head(top_n).copy()
        # Sort for plotting (highest score at top)
        plot_df = plot_df.sort_values("consensus_score", ascending=True)

        fig, ax = plt.subplots(figsize=figsize)

        if "symbol" in plot_df.columns and not plot_df["symbol"].isna().any():
            labels = plot_df["symbol"]
        else:
            labels = plot_df["gene_id"]

        # FIX: Use a better sequential color palette
        colors = sns.color_palette(
            "viridis_r", n_colors=len(plot_df)
        )  # Use seaborn palette (viridis reversed)

        bars = ax.barh(
            range(len(plot_df)),
            plot_df["consensus_score"],
            color=colors,  # Apply palette
            edgecolor="grey",  # Add subtle edge
            linewidth=0.5,
        )

        # Add platform count and direction text
        for i, (_, row) in enumerate(plot_df.iterrows()):
            platforms = int(row["platforms_count"])
            # Use unicode check/cross marks
            direction_symbol = "✔" if row["same_direction"] else "✘"
            bar_width = bars[i].get_width()
            # FIX: Adjust text position slightly based on bar width
            text_x = bar_width + 0.02  # Small offset from bar end
            ax.text(
                text_x,
                i,  # y position (center of the bar)
                f"{platforms} platforms ({direction_symbol})",
                va="center",
                ha="left",  # Align text left
                fontsize=FONT_SIZE_ANNOTATION - 1,  # Use constant, slightly smaller
            )

        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Consensus Score")  # Fontsize from rcParams
        ax.set_ylabel("Gene")  # Fontsize from rcParams
        ax.set_title(f"Top {len(plot_df)} Cross-Platform Consensus Genes")  # Fontsize from rcParams
        ax.grid(axis="x", linestyle="--", alpha=0.6)  # Add grid
        # Adjust xlim for text annotations
        ax.set_xlim(right=plot_df["consensus_score"].max() * 1.15)

        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Same Direction (✔)"),
            Patch(facecolor="red", alpha=0.7, label="Mixed Direction (✘)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            title="Correlation Direction",
            frameon=True,
            facecolor="white",
            framealpha=0.7,
        )

        sns.despine(ax=ax)
        plt.tight_layout()
        return fig

    def visualize_method_overlap(self, overlap_results: dict) -> tuple[Figure, Figure | None]:
        """Visualize the overlap between different correlation methods.

        Args:
            overlap_results: Results from analyze_method_consensus_overlap

        Returns:
            Tuple of (barplot figure, venn diagram figure)
        """
        if not overlap_results or "overlap_stats" not in overlap_results:
            logger.warning("No overlap results to visualize")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No overlap data available", ha="center", va="center")
            return fig, None

        # Create bar plot of overlap percentages
        methods = list(overlap_results["overlap_stats"].keys())
        overlap_percentages = [
            overlap_results["overlap_stats"][method]["overlap_percentage"] for method in methods
        ]

        # Bar plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.bar(methods, overlap_percentages, color="skyblue")

        # Add percentage labels on bars
        for bar, percentage in zip(bars, overlap_percentages, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )

        # Set labels and title
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Overlap Percentage")
        ax1.set_title("Overlap Between Correlation Methods and Consensus Ranking")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Create Venn diagrams for methods with highest overlap
        # Get top 3 methods by overlap percentage
        top_methods = sorted(
            methods,
            key=lambda m: overlap_results["overlap_stats"][m]["overlap_percentage"],
            reverse=True,
        )[: min(3, len(methods))]

        # Try to create Venn diagram if we have matplotlib_venn
        venn_fig = None
        try:
            from matplotlib_venn import venn2, venn3

            fig2, axes = plt.subplots(1, len(top_methods), figsize=(5 * len(top_methods), 5))
            if len(top_methods) == 1:
                axes = [axes]

            for i, method in enumerate(top_methods):
                stats = overlap_results["overlap_stats"][method]

                # Create sets for Venn diagram
                method_set = set(overlap_results["method_top_genes"][method])
                consensus_set = set(overlap_results["consensus_top_genes"])

                # Plot Venn diagram
                if len(method_set) > 0 and len(consensus_set) > 0:
                    venn = venn2(
                        [method_set, consensus_set], set_labels=(method, "Consensus"), ax=axes[i]
                    )

            plt.tight_layout()
            venn_fig = fig2
        except ImportError:
            logger.warning("matplotlib_venn not available, skipping Venn diagrams")

        return fig1, venn_fig
