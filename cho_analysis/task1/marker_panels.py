# cho_analysis/task1/marker_panels.py
"""Tools for designing optimal marker gene panels."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

from cho_analysis.core.logging import setup_logging

logger = setup_logging(__name__)


class MarkerPanelOptimization:
    """Designs and evaluates marker gene panels."""

    def __init__(self):
        """Initialize the panel optimization class."""
        logger.info("MarkerPanelOptimization initialized.")

    def _calculate_mutual_information(self, expr_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates pairwise mutual information between genes."""
        logger.debug("Calculating pairwise mutual information...")
        n_genes = expr_df.shape[1]
        mi_matrix = np.zeros((n_genes, n_genes))
        gene_names = expr_df.columns

        # Ensure data is suitable for MI calculation (no NaNs, numeric)
        expr_df_clean = expr_df.dropna(axis=1, how="any")  # Drop columns (genes) with NaNs
        expr_df_clean = expr_df_clean.select_dtypes(include=np.number)
        if expr_df_clean.shape[1] < n_genes:
            logger.warning(
                f"Dropped {n_genes - expr_df_clean.shape[1]} genes with NaNs for MI calculation."
            )
            gene_names = expr_df_clean.columns
            n_genes = expr_df_clean.shape[1]
            mi_matrix = np.zeros((n_genes, n_genes))  # Resize matrix

        if n_genes < 2:
            logger.warning("Not enough genes for pairwise MI calculation.")
            return pd.DataFrame(index=gene_names, columns=gene_names)

        for i in range(n_genes):
            for j in range(i, n_genes):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Self-information is high (or handle as needed)
                else:
                    try:
                        # mutual_info_regression expects y to be 1D, X can be multiple features
                        # Here we use it pairwise: X=gene_i, y=gene_j
                        mi = mutual_info_regression(
                            expr_df_clean.iloc[:, i].values.reshape(-1, 1),
                            expr_df_clean.iloc[:, j].values,
                            discrete_features=False,  # Assume continuous expression
                            random_state=42,
                        )[0]
                        mi_matrix[i, j] = mi_matrix[j, i] = mi
                    except Exception as e:
                        logger.warning(
                            f"MI calculation failed for pair ({gene_names[i]}, {gene_names[j]}): {e}"
                        )
                        mi_matrix[i, j] = mi_matrix[j, i] = np.nan  # Indicate failure

        mi_df = pd.DataFrame(mi_matrix, index=gene_names, columns=gene_names)
        logger.debug(f"MI matrix calculated shape: {mi_df.shape}")
        return mi_df

    def _greedy_forward_selection(
        self,
        candidates_df: pd.DataFrame,  # DF with 'gene_symbol', 'final_score', etc.
        expr_df: pd.DataFrame,  # Expression data (samples x GENE SYMBOLS)
        mi_df: pd.DataFrame,  # Mutual information matrix (indexed/columned by GENE SYMBOLS)
        panel_size: int,
        objective: str = "minimal_redundancy",
    ) -> List[str]:
        """Selects markers using a greedy forward approach."""
        if candidates_df.empty or expr_df.empty:
            msg = "Cannot perform greedy selection: Input data missing or empty."
            logger.warning(msg)
            return []

        # --- Candidate Filtering and Index Preparation (As previously corrected) ---
        candidate_symbols = candidates_df["gene_symbol"].unique().tolist()
        valid_expr_cols = expr_df.columns.tolist()
        valid_mi_cols = mi_df.columns.tolist() if not mi_df.empty else valid_expr_cols
        valid_candidates_symbols = [
            sym for sym in candidate_symbols if sym in valid_expr_cols and sym in valid_mi_cols
        ]
        if not valid_candidates_symbols:
            logger.warning(
                "No valid candidates found present in both expression and MI data columns."
            )
            return []
        candidates_filtered = candidates_df[
            candidates_df["gene_symbol"].isin(valid_candidates_symbols)
        ].copy()
        sort_score_col = "final_score"
        if (
            sort_score_col not in candidates_filtered.columns
            or candidates_filtered[sort_score_col].isnull().all()
        ):
            sort_score_col = "mean_abs_correlation"
            if (
                sort_score_col not in candidates_filtered.columns
                or candidates_filtered[sort_score_col].isnull().all()
            ):
                logger.warning(
                    "Cannot find reliable score column to resolve duplicate symbols. Using first occurrence."
                )
                candidates_unique = candidates_filtered.drop_duplicates(
                    subset=["gene_symbol"], keep="first"
                )
            else:
                candidates_filtered = candidates_filtered.sort_values(
                    sort_score_col, ascending=False, na_position="last"
                )
                candidates_unique = candidates_filtered.drop_duplicates(
                    subset=["gene_symbol"], keep="first"
                )
        else:
            candidates_filtered = candidates_filtered.sort_values(
                sort_score_col, ascending=False, na_position="last"
            )
            candidates_unique = candidates_filtered.drop_duplicates(
                subset=["gene_symbol"], keep="first"
            )
        try:
            candidates_indexed_unique = candidates_unique.set_index("gene_symbol")
            if not candidates_indexed_unique.index.is_unique:
                logger.error("Failed to create unique gene_symbol index.")
                return []
        except KeyError:
            logger.exception("Failed to set 'gene_symbol' as index.")
            return []
        candidate_pool = candidates_indexed_unique.index.tolist()
        selected_markers = []

        # --- Greedy Selection Loop ---
        while len(selected_markers) < panel_size and candidate_pool:
            best_candidate = None
            best_score = -np.inf

            for candidate_symbol in candidate_pool:
                try:
                    candidate_data = candidates_indexed_unique.loc[candidate_symbol]
                    if not isinstance(candidate_data, pd.Series):
                        continue
                except KeyError:
                    continue

                current_score = 0
                base_score_val = candidate_data.get(
                    "final_score", candidate_data.get("mean_abs_correlation", 0)
                )
                if pd.isna(base_score_val):
                    base_score_val = 0.0

                # --- Calculate Objective Score ---
                if objective == "minimal_redundancy":
                    redundancy = 0
                    if selected_markers and not mi_df.empty and candidate_symbol in mi_df.index:
                        valid_selected_in_mi = [m for m in selected_markers if m in mi_df.columns]
                        if valid_selected_in_mi:
                            mi_values = (
                                mi_df.loc[candidate_symbol, valid_selected_in_mi].fillna(0).values
                            )
                            try:
                                # Ensure mi_values is treated as a numeric array (it should be)
                                mi_values_numeric = np.asarray(mi_values, dtype=float)
                                # Calculate mean ignoring potential NaNs that might remain if fillna(0) wasn't enough
                                redundancy = (
                                    np.nanmean(mi_values_numeric)
                                    if mi_values_numeric.size > 0
                                    else 0.0
                                )
                            except (TypeError, ValueError) as type_err:
                                logger.warning(
                                    f"Could not compute mean redundancy for {candidate_symbol} due to non-numeric MI values: {type_err}"
                                )
                                redundancy = 0.0  # Assign default redundancy penalty
                    current_score = base_score_val - redundancy

                elif objective == "max_correlation":
                    mc_val = candidate_data.get("mean_abs_correlation", 0)
                    current_score = 0.0 if pd.isna(mc_val) else mc_val

                elif objective == "max_score":
                    fs_val = candidate_data.get("final_score", 0)
                    current_score = 0.0 if pd.isna(fs_val) else fs_val

                else:  # Default to minimal redundancy
                    redundancy = 0
                    if selected_markers and not mi_df.empty and candidate_symbol in mi_df.index:
                        valid_selected_in_mi = [m for m in selected_markers if m in mi_df.columns]
                        if valid_selected_in_mi:
                            mi_values = (
                                mi_df.loc[candidate_symbol, valid_selected_in_mi].fillna(0).values
                            )
                            try:
                                mi_values_numeric = np.asarray(mi_values, dtype=float)
                                redundancy = (
                                    np.nanmean(mi_values_numeric)
                                    if mi_values_numeric.size > 0
                                    else 0.0
                                )
                            except (TypeError, ValueError) as type_err:
                                logger.warning(
                                    f"Could not compute mean redundancy for {candidate_symbol} due to non-numeric MI values: {type_err}"
                                )
                                redundancy = 0.0
                    current_score = base_score_val - redundancy

                # Update best candidate if current score is higher
                if current_score > best_score:
                    best_score = current_score
                    best_candidate = candidate_symbol

            if best_candidate:
                selected_markers.append(best_candidate)
                candidate_pool.remove(best_candidate)
            else:
                break

        logger.info(f"Selected {len(selected_markers)} markers for panel (objective: {objective}).")
        return selected_markers

    def _evaluate_panel_loocv(
        self, panel_genes: List[str], expr_df: pd.DataFrame, target_values: pd.Series
    ) -> Dict[str, float]:
        """Evaluates panel performance using Leave-One-Out Cross-Validation."""
        results = {"r2_loocv": np.nan, "rmse_loocv": np.nan}
        if not panel_genes or expr_df.empty or target_values.empty:
            return results

        # Align expression data and target, drop NaNs
        data = expr_df[panel_genes].join(target_values).dropna()
        if data.shape[0] < 2:  # Need at least 2 samples for LOOCV
            logger.warning(
                f"Not enough non-NaN samples ({data.shape[0]}) for LOOCV evaluation of panel: {panel_genes}"
            )
            return results

        X = data[panel_genes].values
        y = data[target_values.name].values

        loo = LeaveOneOut()
        y_preds = np.zeros_like(y, dtype=float)
        model = LinearRegression()

        try:
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, _ = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_preds[test_index] = y_pred[0]

            results["r2_loocv"] = r2_score(y, y_preds)
            results["rmse_loocv"] = np.sqrt(np.mean((y - y_preds) ** 2))

        except Exception as e:
            msg = f"LOOCV evaluation failed for panel {panel_genes}: {e}"
            logger.exception(msg)
            # Keep results as NaN

        return results

    def design_marker_panels(
        self,
        ranked_genes_df: pd.DataFrame,  # Expects 'gene_symbol', 'final_score', etc.
        expression_df: pd.DataFrame,  # Samples x GENE SYMBOLS (columns are symbols)
        target_series: pd.Series,  # Target variable (indexed by sample)
        panel_sizes: List[int] = None,
        panel_types: List[str] = None,  # Objectives
        top_n_candidates: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Designs and evaluates multiple marker panels."""
        if panel_types is None:
            panel_types = ["minimal_redundancy", "max_score"]
        if panel_sizes is None:
            panel_sizes = [3, 5, 10]
        logger.info(f"Designing marker panels (Sizes: {panel_sizes}, Types: {panel_types})...")
        panel_results = {}

        # --- Prepare Data ---
        # Select top N candidates from ranking data
        candidates_df = ranked_genes_df.head(top_n_candidates)
        if "gene_symbol" not in candidates_df.columns:
            msg = "Ranked genes DataFrame must contain 'gene_symbol' column."
            logger.error(msg)
            return {}
        candidate_symbols = candidates_df["gene_symbol"].unique().tolist()

        # Filter expression data to include only candidate symbols that are actually columns
        expr_candidates = expression_df[
            [sym for sym in candidate_symbols if sym in expression_df.columns]
        ]

        if expr_candidates.empty:
            logger.error(
                "No candidate gene symbols found as columns in the provided expression data."
            )
            return {}

        # Calculate Mutual Information once on the filtered expression data
        mi_df = self._calculate_mutual_information(expr_candidates)

        # --- Generate Panels ---
        for p_type in panel_types:
            panel_results[p_type] = {}
            logger.info(f"--- Designing panels with objective: {p_type} ---")
            for p_size in panel_sizes:
                panel_name = f"panel_{p_type}_size{p_size}"
                logger.debug(f"Designing {panel_name}...")

                selected_genes = self._greedy_forward_selection(
                    candidates_df=candidates_df,  # Pass the ranked candidates df
                    expr_df=expr_candidates,  # Pass SYMBOL-columned expression
                    mi_df=mi_df,  # Pass SYMBOL-columned MI matrix
                    panel_size=p_size,
                    objective=p_type,
                )

                if selected_genes:
                    # Evaluate the selected panel
                    evaluation_metrics = self._evaluate_panel_loocv(
                        panel_genes=selected_genes,
                        expr_df=expr_candidates,  # Evaluate using SYMBOL-columned expression
                        target_values=target_series,
                    )
                    panel_results[p_type][panel_name] = {
                        "genes": selected_genes,
                        "size": len(selected_genes),
                        "objective": p_type,
                        **evaluation_metrics,  # Add R2, RMSE etc.
                    }
                    logger.debug(
                        f"  Panel {panel_name}: Genes={selected_genes}, Metrics={evaluation_metrics}"
                    )
                else:
                    logger.warning(f"Could not generate panel {panel_name}.")

        logger.info("Marker panel design and evaluation complete.")
        return panel_results
