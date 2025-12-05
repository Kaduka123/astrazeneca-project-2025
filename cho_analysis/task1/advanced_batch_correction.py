# cho_analysis/task1/advanced_batch_correction.py
"""Advanced methods for detecting and correcting platform-specific batch effects."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.sandbox.stats.multicomp import multipletests

from cho_analysis.core.logging import setup_logging

logger = setup_logging(__name__)


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size, handling NaNs and zero variance.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d statistic, or NaN/Inf if calculation is not possible.
    """
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    n1, n2 = len(group1_clean), len(group2_clean)

    if n1 < 2 or n2 < 2:
        return np.nan

    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    var1, var2 = np.var(group1_clean, ddof=1), np.var(group2_clean, ddof=1)

    # Handle near-zero variance robustly
    is_var1_zero = var1 < 1e-9
    is_var2_zero = var2 < 1e-9

    if is_var1_zero and is_var2_zero:
        return 0.0  # Both constant, no difference
    if is_var1_zero:
        var1 = 0.0
    if is_var2_zero:
        var2 = 0.0

    # Ensure pooled standard deviation calculation is valid
    denominator = n1 + n2 - 2
    if denominator <= 0:
        return np.nan  # Cannot calculate pooled std dev

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / denominator)

    if pooled_std < 1e-9:
        # Means differ, std is zero OR both groups constant with same mean (already handled)
        return np.inf * np.sign(mean1 - mean2) if mean1 != mean2 else 0.0
    return (mean1 - mean2) / pooled_std


class AdvancedBatchCorrection:
    """Implements advanced detection and correction for platform effects."""

    def __init__(self):
        """Initializes the AdvancedBatchCorrection class."""
        logger.info("AdvancedBatchCorrection initialized.")
        self.platform_stats: Dict[str, Any] = {}

    def _get_platform_map(self, manifest_df: pd.DataFrame) -> Dict[str, str]:
        """Extracts platform information robustly from manifest.

        Attempts to derive platform from 'description' if 'platform' column is missing.
        Ensures the index is 'Sample' if that column exists.

        Args:
            manifest_df: DataFrame containing sample metadata. Must have a sample identifier
                         as index or in a 'Sample' column, and either a 'platform' column
                         or a 'description' column containing platform names (HiSeq, NovaSeq, NextSeq).

        Returns:
            A dictionary mapping Sample ID to Platform name. Returns empty if platform
            information cannot be reliably determined.
        """
        manifest_copy = manifest_df.copy()
        platform_col = "platform"

        # Try to derive platform if not present
        if platform_col not in manifest_copy.columns and "description" in manifest_copy.columns:
            logger.debug("Deriving platform from 'description' column.")
            manifest_copy[platform_col] = (
                manifest_copy["description"]
                .str.extract(r"(HiSeq|NovaSeq|NextSeq)", expand=False)
                .fillna("Unknown")
            )
        elif platform_col not in manifest_copy.columns:
            logger.error("Cannot determine platform: 'platform' or 'description' column missing.")
            return {}

        # Ensure index is Sample ID if it exists as a column
        if "Sample" in manifest_copy.columns and manifest_copy.index.name != "Sample":
            try:
                # Check if index is already Sample ID before trying to set it
                if not pd.Index(manifest_copy["Sample"]).equals(manifest_copy.index):
                    manifest_copy = manifest_copy.set_index("Sample", drop=False)
            except KeyError:
                # This should ideally not happen if 'Sample' is in columns, but protects
                logger.warning("Manifest 'Sample' column present but failed during set_index.")
                # Proceed cautiously, map might be based on existing index
                pass

        # Return mapping, excluding samples explicitly marked as 'Unknown' platform
        platform_map = manifest_copy[platform_col].dropna().to_dict()
        return {k: v for k, v in platform_map.items() if v != "Unknown"}

    def detect_residual_platform_effects(
        self,
        corrected_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        alpha: float = 0.01,
        correction_method: str = "fdr_bh",
    ) -> pd.DataFrame | None:
        """Detects genes still significantly affected by platform after correction.

        Uses ANOVA and Kruskal-Wallis tests with multiple testing correction.

        Args:
            corrected_df: Corrected expression data (genes x samples, numeric).
            manifest_df: Sample metadata DataFrame with platform information.
            alpha: Significance level for adjusted p-values.
            correction_method: Method for multiple testing correction (see statsmodels.sandbox.stats.multicomp.multipletests).

        Returns:
            DataFrame with p-values and adjusted p-values for each gene, indexed by gene_id.
            Returns None if detection cannot be performed (e.g., < 2 platforms).
            Returns an empty DataFrame if no genes yield valid test results.
        """
        logger.info(
            f"Detecting residual platform effects (alpha={alpha}, correction={correction_method})..."
        )
        platform_map = self._get_platform_map(manifest_df)
        if not platform_map:
            logger.error("Cannot detect effects: Failed to get platform map.")
            return None

        platforms = sorted(set(platform_map.values()))
        if len(platforms) < 2:
            logger.info("Not enough platforms (>= 2) for residual effect detection.")
            return None

        # Align expression data columns with manifest samples that have platform info
        valid_manifest_samples = list(platform_map.keys())
        common_samples = sorted(set(corrected_df.columns) & set(valid_manifest_samples))
        min_required_samples = max(5, len(platforms) * 2)
        if len(common_samples) < min_required_samples:
            logger.warning(
                f"Too few common samples ({len(common_samples)}) with platform info for robust "
                f"residual effect detection (need at least {min_required_samples})."
            )
            return None

        expr_aligned = corrected_df[common_samples]
        platforms_aligned = np.array([platform_map[s] for s in common_samples])

        results = []
        n_genes = expr_aligned.shape[0]
        if n_genes == 0:
            logger.warning("No genes found in the input dataframe for residual effect detection.")
            return None

        logger.debug(f"Testing {n_genes} genes across {len(platforms)} platforms...")
        for i, gene_id in enumerate(expr_aligned.index):
            if (i + 1) % 5000 == 0:
                logger.debug(f"  Testing gene {i + 1}/{n_genes}...")

            gene_expr = expr_aligned.loc[gene_id].values
            valid_idx = ~np.isnan(gene_expr)  # Index for non-NaN expression for this gene

            # Need at least one valid value per platform for a meaningful comparison
            if sum(valid_idx) < len(platforms):
                continue

            gene_expr_valid = gene_expr[valid_idx]
            platforms_valid = platforms_aligned[valid_idx]

            # Group data by platform for this gene
            groups = [gene_expr_valid[platforms_valid == p] for p in platforms]
            # Filter out groups with less than 2 samples (needed for variance/tests)
            groups_filtered = [g for g in groups if len(g) >= 2]

            # Need at least 2 valid groups for comparison
            if len(groups_filtered) < 2:
                continue

            # Perform tests only on valid groups
            anova_p, kruskal_p = np.nan, np.nan
            try:
                _, anova_p = stats.f_oneway(*groups_filtered)
            except ValueError as e:
                # Can happen if a group has zero variance after filtering NaNs
                logger.debug(f"ANOVA failed for gene {gene_id}: {e}")
            try:
                # Kruskal-Wallis requires variation within groups
                if any(np.nanstd(g) < 1e-9 for g in groups_filtered if len(g) > 0):
                    kruskal_p = 1.0  # Assign non-significant if any group is constant
                else:
                    _, kruskal_p = stats.kruskal(*groups_filtered)
            except ValueError as e:
                logger.debug(f"Kruskal-Wallis failed for gene {gene_id}: {e}")

            results.append(
                {
                    "gene_id": gene_id,
                    "anova_pvalue": anova_p,
                    "kruskal_pvalue": kruskal_p,
                }
            )

        if not results:
            logger.info("No valid results from residual effect tests (check group sizes/variance).")
            return pd.DataFrame()  # Return empty DF if no tests were valid

        results_df = pd.DataFrame(results).set_index("gene_id")

        # Multiple testing correction
        significant_counts = {}
        for p_col in ["anova_pvalue", "kruskal_pvalue"]:
            pvals = results_df[p_col].dropna()
            adj_col_name = f"{p_col}_adj"
            results_df[adj_col_name] = np.nan
            if not pvals.empty:
                # Ensure p-values are within the valid range [0, 1] before correction
                pvals_clipped = np.clip(pvals.values, 0.0, 1.0)
                try:
                    reject, pvals_adj, _, _ = multipletests(
                        pvals_clipped, alpha=alpha, method=correction_method
                    )
                    # Map adjusted p-values back using the original index of non-NaN p-values
                    results_df.loc[pvals.index, adj_col_name] = pvals_adj
                    significant_counts[p_col] = reject.sum()
                except Exception as e:
                    logger.exception(f"Failed multiple testing correction for {p_col}: {e}")
                    significant_counts[p_col] = 0
            else:
                significant_counts[p_col] = 0

        logger.info(f"Residual effects detected (adj p < {alpha}):")
        count_a = significant_counts.get("anova_pvalue", 0)
        count_k = significant_counts.get("kruskal_pvalue", 0)
        total_tested = len(results_df)
        if total_tested > 0:
            logger.info(
                f"  ANOVA: {count_a} / {total_tested} genes ({count_a / total_tested * 100:.1f}%)"
            )
            logger.info(
                f"  Kruskal-Wallis: {count_k} / {total_tested} genes ({count_k / total_tested * 100:.1f}%)"
            )
        else:
            logger.info("  No genes were tested.")

        self.platform_stats["residual_effects"] = results_df
        return results_df

    def quantify_platform_bias(
        self, corrected_df: pd.DataFrame, manifest_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Quantifies bias for each platform relative to all other platforms combined.

        Calculates mean difference, variance ratio, and Cohen's d for each gene.

        Args:
            corrected_df: Corrected expression data (genes x samples, numeric).
            manifest_df: Sample metadata DataFrame with platform information.

        Returns:
            Dictionary where keys are platform names and values are DataFrames
            containing bias metrics (mean_difference, variance_ratio, cohen_d)
            for each gene, indexed by gene_id. Returns empty dict if bias cannot be quantified.
        """
        logger.info("Quantifying platform-specific bias...")
        platform_map = self._get_platform_map(manifest_df)
        if not platform_map:
            logger.error("Cannot quantify bias: Failed to get platform map.")
            return {}

        platforms = sorted(set(platform_map.values()))
        if len(platforms) < 2:
            logger.info("Not enough platforms (>= 2) for bias quantification.")
            return {}

        common_samples = sorted(set(corrected_df.columns) & set(platform_map.keys()))
        if len(common_samples) < 5:  # Arbitrary minimum for meaningful comparison
            logger.warning(
                f"Too few common samples ({len(common_samples)}) for robust bias quantification."
            )
            return {}

        expr_aligned = corrected_df[common_samples]
        platforms_aligned = np.array([platform_map[s] for s in common_samples])

        bias_metrics: Dict[str, pd.DataFrame] = {}
        for platform in platforms:
            logger.debug(f"  Quantifying bias for platform: {platform}")
            platform_idx = platforms_aligned == platform
            # Compare against all OTHER known platforms combined
            other_idx = (platforms_aligned != platform) & (platforms_aligned != "Unknown")

            # Check if we have samples in both groups
            if not np.any(platform_idx) or not np.any(other_idx):
                logger.warning(
                    f"Skipping bias for {platform}: Not enough samples in platform "
                    f"({np.sum(platform_idx)}) or 'other known' platforms ({np.sum(other_idx)})."
                )
                continue

            platform_expr = expr_aligned.loc[:, platform_idx]
            other_expr = expr_aligned.loc[:, other_idx]

            platform_bias_list = []
            for gene_id in expr_aligned.index:
                group1 = platform_expr.loc[gene_id].values
                group2 = other_expr.loc[gene_id].values

                # Check if sufficient non-NaN data exists in both groups for comparison
                if np.sum(~np.isnan(group1)) < 2 or np.sum(~np.isnan(group2)) < 2:
                    continue

                mean_diff = np.nanmean(group1) - np.nanmean(group2)
                var1 = np.nanvar(group1, ddof=1)  # Use sample variance
                var2 = np.nanvar(group2, ddof=1)

                # Calculate variance ratio, handle division by zero or near-zero
                if var2 > 1e-9:
                    var_ratio = var1 / var2
                elif var1 > 1e-9:
                    var_ratio = np.inf
                else:  # Both variances are near zero
                    var_ratio = 1.0  # Or NaN, depending on desired interpretation

                d = cohen_d(group1, group2)

                platform_bias_list.append(
                    {
                        "gene_id": gene_id,
                        "mean_difference": mean_diff,
                        "variance_ratio": var_ratio,
                        "cohen_d": d,
                    }
                )

            if platform_bias_list:
                bias_df = pd.DataFrame(platform_bias_list).set_index("gene_id")
                bias_metrics[platform] = bias_df
                # Calculate mean absolute Cohen's d, ignoring NaNs
                avg_cohen_d = np.nanmean(bias_df["cohen_d"].abs())
                logger.info(f"    Platform {platform}: Avg|Cohen's d| = {avg_cohen_d:.3f}")
            else:
                logger.warning(
                    f"    Platform {platform}: No bias metrics calculated (check per-gene data validity)."
                )

        self.platform_stats["platform_bias"] = bias_metrics
        return bias_metrics

    def hierarchical_batch_correction(
        self,
        standard_corrected_df: pd.DataFrame,
        original_expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        target_gene_id: str,
        problem_platform: str,
        reference_platforms: List[str] | None = None,
        epsilon: float = 1e-6,
    ) -> pd.DataFrame | None:
        """Applies a second-stage location-scale adjustment to a specific platform.

        Adjusts all genes *except* the target_gene_id on the problem_platform
        to match the location (mean) and scale (std dev) of the reference_platforms,
        based on the standard_corrected_df. The target gene's values are preserved
        from the original_expression_df.

        Args:
            standard_corrected_df: Data already corrected by a standard method (e.g., ComBat).
                                   Must be numeric genes x samples (genes as index).
            original_expression_df: Original uncorrected numeric expression data (genes x samples).
                                    Needed to preserve target gene's original values. Must have same gene index.
            manifest_df: Sample metadata containing platform information.
            target_gene_id: The identifier (index) of the target gene whose original values should be preserved.
            problem_platform: The label of the platform whose non-target genes need adjustment.
            reference_platforms: Optional list of platforms to use as reference. If None,
                                 all platforms *except* problem_platform are used.
            epsilon: Small value added to standard deviation before division to avoid zero division.

        Returns:
            DataFrame with hierarchical correction applied, preserving original target gene values,
            or None if an error occurs or adjustment isn't possible. The returned DataFrame
            has the same structure (genes x samples) as the input standard_corrected_df.
        """
        logger.info(
            f"Applying hierarchical location-scale adjustment to platform: {problem_platform}"
        )

        # --- Input Validation ---
        if problem_platform is None:
            logger.error("Problem platform must be specified for hierarchical correction.")
            return None
        if not isinstance(standard_corrected_df, pd.DataFrame) or not isinstance(
            original_expression_df, pd.DataFrame
        ):
            logger.error("Input expression data must be pandas DataFrames.")
            return None
        if not standard_corrected_df.index.equals(original_expression_df.index):
            logger.error(
                "Gene indices of standard_corrected_df and original_expression_df do not match."
            )
            return None
        if target_gene_id not in standard_corrected_df.index:
            logger.error(
                f"Target gene ID '{target_gene_id}' not found in standard corrected data index."
            )
            return None

        # Check target gene in original data is implicitly covered by index equality check
        platform_map = self._get_platform_map(manifest_df)
        if not platform_map:
            logger.error("Cannot correct: Failed to get platform map.")
            return None
        all_platforms = sorted(set(platform_map.values()))
        if problem_platform not in all_platforms:
            logger.error(f"Specified problem platform '{problem_platform}' not found in manifest.")
            return None

        # Determine reference platforms
        if reference_platforms is None:
            reference_platforms = [p for p in all_platforms if p != problem_platform]
        else:
            # Validate provided reference platforms are present and not the problem platform
            valid_refs = [
                p for p in reference_platforms if p in all_platforms and p != problem_platform
            ]
            if len(valid_refs) != len(reference_platforms):
                logger.warning(
                    "Some specified reference platforms were invalid or matched the problem platform. Using only valid ones."
                )
            reference_platforms = valid_refs
        if not reference_platforms:
            logger.error(
                "No valid reference platforms found or specified. Cannot perform adjustment."
            )
            return None
        logger.info(f"Using reference platforms: {reference_platforms}")

        # --- Align Data Columns ---
        # Find samples present in both dataframes and the platform map
        common_samples = sorted(
            set(standard_corrected_df.columns)
            & set(original_expression_df.columns)
            & set(platform_map.keys())
        )
        if len(common_samples) < 5:  # Arbitrary minimum for stability
            logger.warning(
                f"Too few common samples ({len(common_samples)}) across inputs for robust hierarchical correction."
            )
            return None

        # Work on a copy of the standard corrected data for adjustment
        data_to_adjust = standard_corrected_df[common_samples].copy()
        # Align original data as well to fetch target gene values
        original_aligned = original_expression_df[common_samples]
        platforms_aligned = np.array([platform_map[s] for s in common_samples])
        original_sample_order = data_to_adjust.columns

        # --- Preserve Target Gene ---
        # Store original values of the target gene for the aligned samples
        original_target_values = original_aligned.loc[target_gene_id].copy()
        # Identify genes to adjust (all except the target gene)
        gene_ids_to_adjust = data_to_adjust.index.drop(target_gene_id)
        logger.debug(f"Excluding target gene '{target_gene_id}' from location-scale adjustment.")

        # --- Identify Sample Indices for Adjustment ---
        problem_idx_bool = platforms_aligned == problem_platform
        reference_idx_bool = np.isin(platforms_aligned, reference_platforms)

        n_problem_samples = np.sum(problem_idx_bool)
        n_reference_samples = np.sum(reference_idx_bool)

        if n_problem_samples == 0:
            logger.error(
                f"No samples found for problem platform '{problem_platform}' among common samples."
            )
            return None
        if n_reference_samples == 0:
            logger.error("No samples found for reference platforms among common samples.")
            return None
        if n_problem_samples < 2 or n_reference_samples < 2:
            logger.warning(
                f"Low sample count for problem ({n_problem_samples}) or reference ({n_reference_samples}) "
                f"platforms. Adjustment stability might be affected."
            )

        logger.info(
            f"Adjusting {n_problem_samples} samples from '{problem_platform}' based on {n_reference_samples} reference samples."
        )

        # --- Apply Location-Scale Adjustment Per Gene ---
        n_genes_adjusted = 0
        # Iterate through genes that are NOT the target gene
        for gene_id in gene_ids_to_adjust:
            # Get expression for this gene across relevant samples using boolean masks
            ref_values = data_to_adjust.loc[gene_id, reference_idx_bool].values
            prob_values = data_to_adjust.loc[gene_id, problem_idx_bool].values

            # Calculate stats (ignore NaNs)
            mean_ref = np.nanmean(ref_values)
            std_ref = np.nanstd(ref_values)
            mean_prob = np.nanmean(prob_values)
            std_prob = np.nanstd(prob_values)

            # Check if stats are valid for adjustment (require mean and std for both groups)
            if pd.isna(mean_ref) or pd.isna(std_ref) or pd.isna(mean_prob) or pd.isna(std_prob):
                logger.debug(
                    f"Skipping gene {gene_id}: Insufficient data for stats in reference or problem group."
                )
                continue

            # Apply adjustment: shift location, then scale
            # If std dev is near zero, only apply location shift to avoid division by zero/instability
            if std_prob < epsilon:
                adjusted_values = prob_values - mean_prob + mean_ref
                logger.debug(
                    f"Gene {gene_id}: Applying location shift only (low problem variance)."
                )
            else:
                adjusted_values = ((prob_values - mean_prob) / std_prob * std_ref) + mean_ref
                n_genes_adjusted += 1

            # Put adjusted values back into the DataFrame using boolean mask for columns
            # Ensure NaNs in original problem values remain NaNs
            nan_mask_prob = np.isnan(prob_values)
            adjusted_values[nan_mask_prob] = np.nan
            data_to_adjust.loc[gene_id, problem_idx_bool] = adjusted_values

        logger.info(
            f"Applied location-scale adjustment to {n_genes_adjusted} non-target genes for platform '{problem_platform}'."
        )

        # --- Re-insert Original Target Gene Values ---
        # Overwrite the target gene row in the adjusted data with its original values
        data_to_adjust.loc[target_gene_id] = original_target_values
        logger.info(f"Re-inserted original values for target gene '{target_gene_id}'.")

        # Ensure original row order and column order
        # Reindex rows based on the initial standard_corrected_df index order
        # Select columns based on the original sample order we stored
        final_adjusted_df = data_to_adjust.reindex(standard_corrected_df.index).loc[
            :, original_sample_order
        ]

        return final_adjusted_df

    def validate_correction(
        self,
        original_df: pd.DataFrame,
        corrected_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        n_neighbors: int = 15,
        n_pcs: int = 10,
        ks_alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Validates batch correction effectiveness using multiple metrics.

        Metrics include:
        1. PCA Silhouette Score: Measures cluster separation by batch label in PCA space. Lower is better after correction.
        2. k-NN Mixing Rate: Measures the proportion of nearest neighbors belonging to the same batch. Lower is better after correction.
        3. Kolmogorov-Smirnov Test: Measures the proportion of genes with significantly different distributions between platform pairs. Lower is better after correction.

        Args:
            original_df: Original uncorrected data (numeric genes x samples).
            corrected_df: Batch-corrected data (numeric genes x samples).
            manifest_df: Sample metadata with platform info.
            n_neighbors: Number of neighbors for kNN mixing test.
            n_pcs: Number of principal components to use for PCA/kNN space.
            ks_alpha: Significance level for reporting KS test results.

        Returns:
            Dictionary containing validation metrics before and after correction.
            Keys include 'silhouette_before/after', 'knn_mixing_rate_before/after',
            'ks_prop_diff_before/after'. Values might be NaN if metrics cannot be calculated.
            Includes 'error' or 'warning' keys if fundamental issues arise.
        """
        logger.info("Validating batch correction effectiveness...")
        validation_results: Dict[str, Any] = {}

        platform_map = self._get_platform_map(manifest_df)
        if not platform_map:
            logger.error("Validation failed: Could not get platform map.")
            return {"error": "Platform mapping failed"}

        # --- Align dataframes and get batch labels ---
        common_genes = sorted(set(original_df.index) & set(corrected_df.index))
        common_samples = sorted(
            set(original_df.columns) & set(corrected_df.columns) & set(platform_map.keys())
        )

        # Determine minimum sample requirement based on tests
        unique_platforms = {
            platform_map[s] for s in common_samples if s in platform_map
        }  # Get platforms actually present in common samples
        min_required_samples = max(5, n_neighbors + 1, len(unique_platforms) * 2)

        if len(common_genes) < 2 or len(common_samples) < min_required_samples:
            logger.error(
                f"Validation failed: Insufficient common genes ({len(common_genes)} < 2) or "
                f"samples ({len(common_samples)} < {min_required_samples}) for all tests."
            )
            return {"error": "Insufficient common data"}

        original_aligned = original_df.loc[common_genes, common_samples]
        corrected_aligned = corrected_df.loc[common_genes, common_samples]
        batch_labels = np.array([platform_map[s] for s in common_samples])
        unique_batches, batch_counts = np.unique(batch_labels, return_counts=True)

        # Check for sufficient samples per batch for specific tests
        if len(unique_batches) < 2:
            logger.warning(
                "Skipping validation: Less than 2 unique batches found in common samples."
            )
            return {"warning": "Less than 2 unique batches"}
        if any(count < 2 for count in batch_counts):
            low_sample_batches = {
                batch: count
                for batch, count in zip(unique_batches, batch_counts, strict=False)
                if count < 2
            }
            logger.warning(
                f"Some batches have < 2 samples: {low_sample_batches}. Silhouette/kNN may be unreliable or skipped."
            )

        # --- Preprocessing for PCA/kNN ---
        # Ensure data is numeric and handle NaNs by imputation (gene mean)
        # Transpose to Samples x Genes for PCA/kNN input format
        logger.debug("Preprocessing data for PCA/kNN (imputing NaNs)...")
        original_aligned_filled = original_aligned.apply(lambda x: x.fillna(x.mean()), axis=1).T
        corrected_aligned_filled = corrected_aligned.apply(lambda x: x.fillna(x.mean()), axis=1).T

        # Check for remaining NaNs after imputation (if a whole gene was NaN)
        if (
            original_aligned_filled.isnull().any().any()
            or corrected_aligned_filled.isnull().any().any()
        ):
            logger.warning(
                "NaNs remain after gene-mean imputation (likely entire gene was NaN). These genes will be dropped for PCA/kNN."
            )
            original_aligned_filled = original_aligned_filled.dropna(axis=1)
            corrected_aligned_filled = corrected_aligned_filled.dropna(axis=1)  # Drop same columns
            if original_aligned_filled.shape[1] == 0:
                logger.error(
                    "Validation failed: No valid genes remaining after NaN removal for PCA/kNN."
                )
                return {"error": "No valid genes for PCA/kNN"}

        pca_original, pca_corrected = None, None  # Initialize PCA results

        # --- 1. PCA and Silhouette Score ---
        logger.debug("Calculating PCA and Silhouette scores...")
        # Adjust n_pcs based on the minimum of (samples-1) and (features-1)
        n_pcs_actual = min(
            n_pcs,
            original_aligned_filled.shape[0] - 1,
            original_aligned_filled.shape[1] - 1,
        )

        if n_pcs_actual < 2:
            logger.warning(
                f"Skipping PCA/Silhouette validation: Not enough dimensions/samples to compute at least 2 PCs (max possible: {n_pcs_actual})."
            )
            validation_results.update({"silhouette_before": np.nan, "silhouette_after": np.nan})
        else:
            try:
                # Standardize data before PCA
                scaler = StandardScaler()
                scaled_original = scaler.fit_transform(original_aligned_filled)
                scaled_corrected = scaler.transform(corrected_aligned_filled)  # Use same scaler

                # Fit PCA on original data, transform both
                pca = PCA(n_components=n_pcs_actual)
                pca_original = pca.fit_transform(scaled_original)
                pca_corrected = pca.transform(scaled_corrected)

                # Calculate silhouette score (only if >1 batch label with >1 member)
                valid_batches_for_silhouette = [
                    batch
                    for batch, count in zip(unique_batches, batch_counts, strict=False)
                    if count > 1
                ]
                if len(valid_batches_for_silhouette) > 1:
                    score_before = silhouette_score(pca_original, batch_labels)
                    score_after = silhouette_score(pca_corrected, batch_labels)
                    validation_results["silhouette_before"] = score_before
                    validation_results["silhouette_after"] = score_after
                    logger.info(
                        f"  Silhouette Score (PCA space, {n_pcs_actual} PCs): Before={score_before:.4f}, After={score_after:.4f}"
                    )
                else:
                    logger.warning("Skipping silhouette score: Less than 2 batches with >1 member.")
                    validation_results.update(
                        {"silhouette_before": np.nan, "silhouette_after": np.nan}
                    )

            except Exception as e:
                logger.exception(f"PCA/Silhouette calculation failed: {e}")
                validation_results.update({"silhouette_before": np.nan, "silhouette_after": np.nan})
                pca_original, pca_corrected = (
                    None,
                    None,
                )  # Ensure PCA results are nullified on error

        # --- 2. k-NN Batch Effect Test ---
        logger.debug(f"Calculating k-NN mixing (k={n_neighbors})...")
        # Use PCA space if available and valid
        if pca_original is not None and pca_corrected is not None:
            # Check if n_samples > n_neighbors, adjust k if necessary
            current_n_samples = pca_original.shape[0]
            if current_n_samples <= n_neighbors:
                knn_k = max(1, current_n_samples - 1)
                logger.warning(
                    f"Number of samples ({current_n_samples}) <= n_neighbors ({n_neighbors}). Adjusting kNN k to {knn_k}."
                )
            else:
                knn_k = n_neighbors

            if knn_k > 0:
                try:
                    mixing_before = self._calculate_knn_mixing(pca_original, batch_labels, knn_k)
                    mixing_after = self._calculate_knn_mixing(pca_corrected, batch_labels, knn_k)
                    validation_results["knn_mixing_rate_before"] = mixing_before
                    validation_results["knn_mixing_rate_after"] = mixing_after
                    logger.info(
                        f"  kNN Mixing Rate (% same batch, k={knn_k}): Before={mixing_before*100:.1f}%, After={mixing_after*100:.1f}%"
                    )
                except Exception as e:
                    logger.exception(f"kNN mixing calculation failed: {e}")
                    validation_results.update(
                        {"knn_mixing_rate_before": np.nan, "knn_mixing_rate_after": np.nan}
                    )
            else:
                logger.warning(
                    "Skipping kNN validation: Not enough samples to find neighbors (k=0)."
                )
                validation_results.update(
                    {"knn_mixing_rate_before": np.nan, "knn_mixing_rate_after": np.nan}
                )
        else:
            logger.warning(
                "Skipping kNN validation: PCA results not available (likely due to previous error or insufficient dimensions)."
            )
            validation_results.update(
                {"knn_mixing_rate_before": np.nan, "knn_mixing_rate_after": np.nan}
            )

        # --- 3. Distribution Similarity (KS Test per gene) ---
        logger.debug("Calculating inter-platform distribution similarity (KS test)...")
        ks_results_before: List[bool] = []
        ks_results_after: List[bool] = []
        total_gene_platform_pairs_tested = 0
        platforms_with_samples = unique_batches  # Use batches actually present

        if len(platforms_with_samples) >= 2:
            num_valid_pairs = 0
            for i in range(len(platforms_with_samples)):
                for j in range(i + 1, len(platforms_with_samples)):
                    p1, p2 = platforms_with_samples[i], platforms_with_samples[j]
                    p1_idx = batch_labels == p1
                    p2_idx = batch_labels == p2

                    # Ensure both platforms in the pair have at least 2 samples for KS test
                    if np.sum(p1_idx) < 2 or np.sum(p2_idx) < 2:
                        logger.debug(
                            f"Skipping KS test for pair ({p1}, {p2}): Insufficient samples."
                        )
                        continue
                    num_valid_pairs += 1

                    genes_tested_this_pair = 0
                    for gene_id in common_genes:
                        try:
                            # Original Data
                            vals1_orig = original_aligned.loc[gene_id, p1_idx].dropna().values
                            vals2_orig = original_aligned.loc[gene_id, p2_idx].dropna().values
                            # KS test requires at least 2 values per group and variation
                            if (
                                len(vals1_orig) > 1
                                and len(vals2_orig) > 1
                                and np.ptp(vals1_orig) > 0
                                and np.ptp(vals2_orig) > 0
                            ):
                                _, p_val_orig = ks_2samp(vals1_orig, vals2_orig)
                                ks_results_before.append(p_val_orig < ks_alpha)
                                genes_tested_this_pair += 1  # Count only if test is performed

                            # Corrected Data
                            vals1_corr = corrected_aligned.loc[gene_id, p1_idx].dropna().values
                            vals2_corr = corrected_aligned.loc[gene_id, p2_idx].dropna().values
                            if (
                                len(vals1_corr) > 1
                                and len(vals2_corr) > 1
                                and np.ptp(vals1_corr) > 0
                                and np.ptp(vals2_corr) > 0
                            ):
                                _, p_val_corr = ks_2samp(vals1_corr, vals2_corr)
                                ks_results_after.append(p_val_corr < ks_alpha)
                            # Note: We count a gene-pair test if original OR corrected works, assuming alignment
                        except ValueError as e:
                            # Can occur if inputs are identical after dropping NaNs
                            logger.debug(
                                f"KS test failed for gene {gene_id}, pair ({p1}, {p2}): {e}"
                            )
                    total_gene_platform_pairs_tested += genes_tested_this_pair

            prop_diff_before = np.mean(ks_results_before) if ks_results_before else np.nan
            prop_diff_after = np.mean(ks_results_after) if ks_results_after else np.nan
            validation_results["ks_prop_diff_before"] = prop_diff_before
            validation_results["ks_prop_diff_after"] = prop_diff_after
            if total_gene_platform_pairs_tested > 0:
                logger.info(
                    f"  Avg Proportion Sig. Diff Distributions (KS p<{ks_alpha} across {total_gene_platform_pairs_tested} gene-platform pairs): Before={prop_diff_before*100:.1f}%, After={prop_diff_after*100:.1f}%"
                )
            else:
                logger.warning("  KS test: No valid gene-platform pairs were tested.")
                validation_results.update(
                    {"ks_prop_diff_before": np.nan, "ks_prop_diff_after": np.nan}
                )

        else:
            logger.warning(
                "Skipping KS test validation: Less than 2 platforms with sufficient samples in common data."
            )
            validation_results.update({"ks_prop_diff_before": np.nan, "ks_prop_diff_after": np.nan})

        logger.info("Batch correction validation complete.")
        self.platform_stats["validation_results"] = validation_results
        return validation_results

    def _calculate_knn_mixing(
        self, data_matrix: np.ndarray, batch_labels: np.ndarray, n_neighbors: int
    ) -> float:
        """Helper to calculate the k-NN mixing rate.

        Calculates the average proportion of the k nearest neighbors for each sample
        that belong to the same batch as the sample itself. Lower values indicate
        better mixing (less batch separation).

        Args:
            data_matrix: Data matrix (samples x features). Should be preprocessed (e.g., scaled, PCA).
            batch_labels: Array of batch labels corresponding to the rows of data_matrix.
            n_neighbors: Number of neighbors (k) to consider.

        Returns:
            The average mixing rate (proportion of same-batch neighbors), or NaN if calculation fails.
        """
        n_samples = data_matrix.shape[0]

        # n_neighbors should have been adjusted before calling, but double-check
        if n_samples <= n_neighbors or n_neighbors <= 0:
            logger.error(
                f"kNN mixing called with invalid n_neighbors ({n_neighbors}) for n_samples ({n_samples})."
            )
            return np.nan

        # Check for constant features which can cause issues with distance metrics
        non_constant_features_mask = np.std(data_matrix, axis=0) > 1e-9
        if not np.all(non_constant_features_mask):
            n_const = np.sum(~non_constant_features_mask)
            logger.warning(f"Removing {n_const} constant features before kNN.")
            data_matrix = data_matrix[:, non_constant_features_mask]
            if data_matrix.shape[1] == 0:
                logger.error("No non-constant features left for kNN.")
                return np.nan

        try:
            # Find k+1 neighbors because the sample itself is included
            nn = NearestNeighbors(
                n_neighbors=n_neighbors + 1, algorithm="auto", metric="euclidean"
            ).fit(data_matrix)
            distances, indices = nn.kneighbors(data_matrix)
        except ValueError as e:
            logger.exception(f"NearestNeighbors failed: {e}. Check input data (e.g., NaNs, Infs).")
            return np.nan

        mixing_rates = []
        for i in range(n_samples):
            # Exclude self (index 0) which is always the first neighbor
            neighbor_indices = indices[i, 1:]

            # Ensure neighbor indices are within the bounds of batch_labels (sanity check)
            valid_neighbor_indices = neighbor_indices[
                (neighbor_indices >= 0) & (neighbor_indices < len(batch_labels))
            ]
            if len(valid_neighbor_indices) == 0:
                # This shouldn't happen with valid k and data, but handle defensively
                logger.warning(f"Sample {i} had no valid neighbors found.")
                continue

            neighbor_batches = batch_labels[valid_neighbor_indices]
            same_batch_proportion = np.mean(neighbor_batches == batch_labels[i])
            mixing_rates.append(same_batch_proportion)

        return np.mean(mixing_rates) if mixing_rates else np.nan
