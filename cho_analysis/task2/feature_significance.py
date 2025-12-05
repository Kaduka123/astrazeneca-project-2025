# cho_analysis/task2/feature_significance.py
"""Statistical feature significance analysis for Task 2.

This module provides functionality to analyze the statistical significance of sequence features,
identify feature interactions, and perform motif enrichment analysis in UTR regions.
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)

# Suppress specific SciPy and sklearn user warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", message="The p_value parameter", module="sklearn")


class FeatureSignificanceAnalysis:
    """Analyzes the statistical significance of sequence features."""

    def __init__(self):
        """Initialize the feature significance analysis."""
        self.results = {}

    def analyze_feature_significance(
        self, feature_df: pd.DataFrame, alpha: float = 0.05, correction_method: str = "fdr_bh"
    ) -> pd.DataFrame:
        """Calculate correlations between sequence features and expression metrics.

        Args:
            feature_df: DataFrame containing sequence features and expression metrics
            alpha: Significance level for hypothesis testing
            correction_method: Method for multiple testing correction ('fdr_bh', 'bonferroni')

        Returns:
            DataFrame with correlation statistics for each feature
        """
        logger.info("Analyzing feature significance...")

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return pd.DataFrame()

        # Check if any required expression metrics exist
        available_metrics = [col for col in ["cv", "mean"] if col in feature_df.columns]
        if not available_metrics:
            logger.error("Neither 'cv' nor 'mean' columns found in feature DataFrame")
            return pd.DataFrame()

        # Select numeric features
        feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ["cv", "mean"]]

        if not feature_cols:
            logger.error("No numeric feature columns found")
            return pd.DataFrame()

        # Create result structure
        result_data = []

        # Analyze each available target metric
        for target in available_metrics:
            target_values = feature_df[target].values

            for feature in feature_cols:
                feature_values = feature_df[feature].replace([np.inf, -np.inf], np.nan).dropna()

                # Skip if too many missing values
                if len(feature_values) < 10:
                    logger.warning(f"Skipping {feature} - insufficient data")
                    continue

                # Create mask for valid indices in both feature and target
                valid_mask = feature_df[feature].notna() & feature_df[target].notna()
                if valid_mask.sum() < 10:
                    logger.warning(f"Skipping {feature} vs {target} - insufficient valid pairs")
                    continue

                feature_data = feature_df.loc[valid_mask, feature]
                target_data = feature_df.loc[valid_mask, target]

                # Calculate correlations
                try:
                    # Pearson correlation (linear)
                    pearson_corr, pearson_p = stats.pearsonr(feature_data, target_data)

                    # Ensure p-values are within reasonable bounds for correction
                    # Set a minimum p-value threshold based on sample size
                    # Formula: 10^(-2*log10(n)) where n is sample size
                    min_possible_p = 10**(-2 * np.log10(len(feature_data)))

                    # Clip p-values to be at least the minimum possible
                    pearson_p = max(pearson_p, min_possible_p)

                    # Spearman correlation (monotonic)
                    spearman_corr, spearman_p = stats.spearmanr(feature_data, target_data)
                    spearman_p = max(spearman_p, min_possible_p)

                    # Mutual information (non-linear)
                    mi_score = mutual_info_regression(
                        feature_data.values.reshape(-1, 1), target_data.values, random_state=42
                    )[0]

                    # Add to results
                    result_data.append(
                        {
                            "feature": feature,
                            "target": target,
                            "pearson_r": pearson_corr,
                            "pearson_p": float(pearson_p),  # Ensure float type
                            "spearman_rho": spearman_corr,
                            "spearman_p": float(spearman_p),  # Ensure float type
                            "mutual_info": mi_score,
                            "n_samples": len(feature_data),
                            "min_possible_p": min_possible_p
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error analyzing {feature} vs {target}: {e}")

        # Create results DataFrame
        if not result_data:
            logger.warning("No valid feature correlations could be calculated")
            return pd.DataFrame()

        result_df = pd.DataFrame(result_data)

        # Multiple testing correction
        for p_col in ["pearson_p", "spearman_p"]:
            try:
                # Clean p-values to ensure they're in valid range [0,1]
                valid_mask = np.isfinite(result_df[p_col]) & (result_df[p_col] >= 0) & (result_df[p_col] <= 1)
                if not valid_mask.all():
                    invalid_count = (~valid_mask).sum()
                    logger.warning(f"Found {invalid_count} invalid {p_col} values (NaN, Inf, or outside [0,1]). Replacing with 1.0.")
                    result_df.loc[~valid_mask, p_col] = 1.0

                # Apply multiple testing correction
                if correction_method == "fdr_bh":
                    # Use statsmodels implementation for more robust correction
                    from statsmodels.stats.multitest import multipletests
                    _, result_df[f"{p_col}_adjusted"], _, _ = multipletests(
                        result_df[p_col].values, alpha=alpha, method='fdr_bh'
                    )
                elif correction_method == "bonferroni":
                    result_df[f"{p_col}_adjusted"] = np.minimum(
                        result_df[p_col] * len(result_df), 1.0
                    )
                else:
                    logger.warning(f"Unknown correction method: {correction_method}")
                    result_df[f"{p_col}_adjusted"] = result_df[p_col]

                # Ensure adjusted p-values are within reasonable bounds
                result_df[f"{p_col}_adjusted"] = result_df[f"{p_col}_adjusted"].clip(
                    result_df["min_possible_p"], 1.0
                )

            except Exception as e:
                logger.exception(f"Error in multiple testing correction: {e}")
                result_df[f"{p_col}_adjusted"] = result_df[p_col]

        # Add significance flags
        result_df["is_significant"] = (result_df["pearson_p_adjusted"] < alpha) | (
            result_df["spearman_p_adjusted"] < alpha
        )

        # Add effect size (absolute correlation)
        result_df["effect_size"] = result_df[["pearson_r", "spearman_rho"]].abs().mean(axis=1)

        # Sort by effect size and significance
        result_df = result_df.sort_values(
            by=["is_significant", "effect_size"], ascending=[False, False]
        )

        # Drop the helper column
        if "min_possible_p" in result_df.columns:
            result_df = result_df.drop(columns=["min_possible_p"])

        logger.info(
            f"Identified {result_df['is_significant'].sum()} significant features out of {len(result_df)}"
        )
        self.results["feature_significance"] = result_df

        return result_df

    def analyze_feature_interactions(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Examine pairwise interactions between sequence features.

        Args:
            feature_df: DataFrame containing sequence features

        Returns:
            DataFrame with feature interaction statistics
        """
        logger.info("Analyzing feature interactions...")

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return pd.DataFrame()

        # Select numeric features (excluding expression metrics)
        feature_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ["cv", "mean"]]

        if len(feature_cols) < 2:
            logger.error("Need at least two feature columns for interaction analysis")
            return pd.DataFrame()

        # Calculate mutual information matrix
        feature_data = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with column medians
        for col in feature_data.columns:
            feature_data[col] = feature_data[col].fillna(feature_data[col].median())

        # Create result structure for pairwise interactions
        interaction_data = []

        # Analyze each pair of features
        for i, feat1 in enumerate(feature_cols):
            for feat2 in feature_cols[i + 1 :]:
                try:
                    # Calculate mutual information between features
                    mi_score = mutual_info_regression(
                        feature_data[feat1].values.reshape(-1, 1),
                        feature_data[feat2].values,
                        random_state=42,
                    )[0]

                    # Calculate correlation for comparison
                    corr, p_val = stats.spearmanr(feature_data[feat1], feature_data[feat2])

                    # Add to results
                    interaction_data.append(
                        {
                            "feature1": feat1,
                            "feature2": feat2,
                            "mutual_info": mi_score,
                            "correlation": corr,
                            "p_value": p_val,
                            "interaction_type": "synergistic" if corr > 0 else "antagonistic",
                            "strength": abs(corr),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error analyzing interaction between {feat1} and {feat2}: {e}")

        if not interaction_data:
            logger.warning("No valid feature interactions could be calculated")
            return pd.DataFrame()

        interaction_df = pd.DataFrame(interaction_data)

        # Sort by mutual information
        interaction_df = interaction_df.sort_values(by="mutual_info", ascending=False)

        logger.info(f"Analyzed {len(interaction_df)} feature pair interactions")
        self.results["feature_interactions"] = interaction_df

        return interaction_df

    def analyze_utr5_motif_enrichment(
        self, consistent_genes_df: pd.DataFrame, variable_genes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Implement de novo motif discovery in 5' UTRs.

        Args:
            consistent_genes_df: DataFrame with consistently expressed genes and UTR sequences
            variable_genes_df: DataFrame with variably expressed genes and UTR sequences

        Returns:
            DataFrame with motif enrichment statistics
        """
        logger.info("Analyzing 5' UTR motif enrichment...")

        if consistent_genes_df.empty or variable_genes_df.empty:
            logger.error("Empty gene DataFrames provided")
            return pd.DataFrame()

        # Check if required columns exist
        required_cols = ["UTR5_Seq"]
        missing_cols = [
            col
            for col in required_cols
            if col not in consistent_genes_df.columns or col not in variable_genes_df.columns
        ]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Extract sequences
        consistent_seqs = consistent_genes_df["UTR5_Seq"].dropna().tolist()
        variable_seqs = variable_genes_df["UTR5_Seq"].dropna().tolist()

        if not consistent_seqs or not variable_seqs:
            logger.error("No valid 5' UTR sequences found")
            return pd.DataFrame()

        # Find kmers (simple motif discovery approach)
        top_kmers = self._discover_enriched_kmers(
            consistent_seqs,
            variable_seqs,
            k=6,  # 6-mer motifs
            top_n=30,
        )

        if not top_kmers:
            logger.warning("No enriched motifs found in 5' UTR")
            return pd.DataFrame()

        logger.info(f"Found {len(top_kmers)} enriched motifs in 5' UTR")
        self.results["utr5_motifs"] = top_kmers

        return top_kmers

    def analyze_utr3_motif_enrichment(
        self, consistent_genes_df: pd.DataFrame, variable_genes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Implement de novo motif discovery in 3' UTRs.

        Args:
            consistent_genes_df: DataFrame with consistently expressed genes and UTR sequences
            variable_genes_df: DataFrame with variably expressed genes and UTR sequences

        Returns:
            DataFrame with motif enrichment statistics
        """
        logger.info("Analyzing 3' UTR motif enrichment...")

        if consistent_genes_df.empty or variable_genes_df.empty:
            logger.error("Empty gene DataFrames provided")
            return pd.DataFrame()

        # Check if required columns exist
        required_cols = ["UTR3_Seq"]
        missing_cols = [
            col
            for col in required_cols
            if col not in consistent_genes_df.columns or col not in variable_genes_df.columns
        ]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Extract sequences
        consistent_seqs = consistent_genes_df["UTR3_Seq"].dropna().tolist()
        variable_seqs = variable_genes_df["UTR3_Seq"].dropna().tolist()

        if not consistent_seqs or not variable_seqs:
            logger.error("No valid 3' UTR sequences found")
            return pd.DataFrame()

        # Find kmers (simple motif discovery approach)
        top_kmers = self._discover_enriched_kmers(
            consistent_seqs,
            variable_seqs,
            k=6,  # 6-mer motifs
            top_n=30,
        )

        if not top_kmers:
            logger.warning("No enriched motifs found in 3' UTR")
            return pd.DataFrame()

        logger.info(f"Found {len(top_kmers)} enriched motifs in 3' UTR")
        self.results["utr3_motifs"] = top_kmers

        return top_kmers

    def _discover_enriched_kmers(
        self, positive_seqs: list[str], negative_seqs: list[str], k: int = 6, top_n: int = 30
    ) -> pd.DataFrame:
        """Find enriched k-mers in a set of sequences compared to a background set.

        Args:
            positive_seqs: List of sequences to analyze for motif enrichment
            negative_seqs: Background sequences for comparison
            k: Length of k-mers to analyze
            top_n: Number of top enriched k-mers to return

        Returns:
            DataFrame with enriched k-mers and statistics
        """
        # Count k-mers in positive set
        pos_kmer_counts = {}
        for seq in positive_seqs:
            if not seq or len(seq) < k:
                continue
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                pos_kmer_counts[kmer] = pos_kmer_counts.get(kmer, 0) + 1

        # Count k-mers in negative set
        neg_kmer_counts = {}
        for seq in negative_seqs:
            if not seq or len(seq) < k:
                continue
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                neg_kmer_counts[kmer] = neg_kmer_counts.get(kmer, 0) + 1

        # Calculate enrichment
        enrichment_data = []

        total_pos_seqs = len(positive_seqs)
        total_neg_seqs = len(negative_seqs)

        # Pseudocount to avoid division by zero
        pseudocount = 0.5

        for kmer in pos_kmer_counts:
            if kmer not in neg_kmer_counts:
                neg_kmer_counts[kmer] = 0

            # Calculate frequency and enrichment
            pos_freq = pos_kmer_counts[kmer] / total_pos_seqs
            neg_freq = (neg_kmer_counts[kmer] + pseudocount) / total_neg_seqs

            # Fold enrichment
            fold_enrichment = pos_freq / neg_freq

            # Fisher's exact test
            pos_with = sum(1 for seq in positive_seqs if kmer in seq)
            pos_without = total_pos_seqs - pos_with
            neg_with = sum(1 for seq in negative_seqs if kmer in seq)
            neg_without = total_neg_seqs - neg_with

            # Create contingency table
            contingency = np.array([[pos_with, pos_without], [neg_with, neg_without]])

            try:
                # Calculate p-value
                odds_ratio, p_value = stats.fisher_exact(contingency)

                enrichment_data.append(
                    {
                        "motif": kmer,
                        "count_consistent": pos_kmer_counts[kmer],
                        "count_variable": neg_kmer_counts[kmer],
                        "freq_consistent": pos_freq,
                        "freq_variable": neg_freq,
                        "fold_enrichment": fold_enrichment,
                        "p_value": p_value,
                        "odds_ratio": odds_ratio,
                    }
                )
            except Exception as e:
                logger.warning(f"Error calculating enrichment for {kmer}: {e}")

        if not enrichment_data:
            return pd.DataFrame()

        # Create DataFrame and sort by fold enrichment
        enrichment_df = pd.DataFrame(enrichment_data)

        # Apply multiple testing correction
        try:
            enrichment_df["p_adjusted"] = stats.false_discovery_control(
                enrichment_df["p_value"].values, method="bh"
            )
        except Exception as e:
            logger.exception(f"Error in multiple testing correction: {e}")
            enrichment_df["p_adjusted"] = enrichment_df["p_value"]

        # Sort by fold enrichment
        enrichment_df = enrichment_df.sort_values(by=["fold_enrichment"], ascending=False).head(
            top_n
        )

        return enrichment_df


class FeatureSignificanceVisualization:
    """Creates visualizations for feature significance analysis."""

    def __init__(self):
        """Initialize the visualization class."""
        self.figures = {}

    def plot_feature_significance(
        self,
        significance_df: pd.DataFrame,
        highlight_features: list[str] | None = None,
        figsize: tuple[int, int] = (12, 9),
    ) -> Figure:
        """Generate a volcano plot showing significance vs. effect size.

        Args:
            significance_df: DataFrame with feature significance statistics
            highlight_features: List of feature names to highlight in the plot
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if significance_df.empty:
            logger.error("Empty significance DataFrame provided")
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.text(0.5, 0.5, "No data available for volcano plot",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Effect Size (Correlation Magnitude)")
            ax.set_ylabel("-log10(Adjusted p-value)")
            ax.set_title("Feature Significance - No Data")
            return fig

        required_cols = ["feature", "effect_size", "pearson_p_adjusted", "target"]
        missing_cols = [col for col in required_cols if col not in significance_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns for volcano plot: {missing_cols}")
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.text(0.5, 0.5, f"Missing columns: {', '.join(missing_cols)}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Feature Significance - Missing Data")
            return fig

        try:
            # Create separate plots for each target (cv, mean)
            targets = significance_df["target"].unique()

            # Create figure with better styling
            fig = plt.figure(figsize=figsize, dpi=300, facecolor="#f8f9fa")

            # Use GridSpec for more control over layout
            gs = fig.add_gridspec(1, len(targets), wspace=0.35)
            axes = [fig.add_subplot(gs[0, i]) for i in range(len(targets))]

            for i, target in enumerate(targets):
                ax = axes[i]
                ax.set_facecolor("#f8f9fa")

                # Filter data for this target
                target_data = significance_df[significance_df["target"] == target].copy()

                # Ensure p-values are valid for log transformation
                # Replace zeros or extremely small values with a reasonable minimum
                min_valid_p = 1e-10

                # Clean p-values and limit extremely small values
                target_data["pearson_p_adjusted_clean"] = target_data["pearson_p_adjusted"].clip(min_valid_p, 1.0)

                # Check for distribution of p-values
                p_value_counts = target_data["pearson_p_adjusted"].value_counts()

                # If all or most p-values are the same extreme value, adjust visualization strategy
                if len(p_value_counts) <= 2 and target_data["pearson_p_adjusted"].min() < 1e-6:
                    logger.warning("Most p-values are identical extreme values. Using jittered visualization.")

                    # Create a jitter factor based on effect size to spread out points
                    target_data["jitter"] = np.random.normal(0, 0.05, size=len(target_data))

                    # Create a stratified log p-value scale to separate the points
                    # Map all significant p-values to a range between 5 and 10 based on effect size
                    significant_mask = target_data["pearson_p_adjusted"] < 0.05

                    # Base value for all significant points
                    target_data["-log10p"] = np.where(significant_mask, 7.0, 0.5)

                    # Add effect size-based variation to spread points vertically
                    # Normalize effect size to 0-1 range for significant points
                    if significant_mask.any():
                        effect_min = target_data.loc[significant_mask, "effect_size"].min()
                        effect_max = target_data.loc[significant_mask, "effect_size"].max()
                        effect_range = max(0.1, effect_max - effect_min)

                        # Create a normalized effect size (0-1) and scale to add variation (±3)
                        target_data.loc[significant_mask, "effect_norm"] = (
                            (target_data.loc[significant_mask, "effect_size"] - effect_min) / effect_range
                        )
                        target_data.loc[significant_mask, "-log10p"] += (
                            target_data.loc[significant_mask, "effect_norm"] * 3.0 +
                            np.random.normal(0, 0.5, size=significant_mask.sum())
                        )

                    # Add some jitter to non-significant points too
                    if (~significant_mask).any():
                        target_data.loc[~significant_mask, "-log10p"] += np.random.normal(0, 0.2, size=(~significant_mask).sum())
                else:
                    # Regular approach when p-values are well distributed
                    if (target_data["pearson_p_adjusted"] < 1e-100).any():
                        logger.warning("Detected unrealistically small p-values (<1e-100). Capping for visualization.")
                        # Cap the log transformation for visualization purposes
                        max_log_p = 20  # Corresponds to p-value of 1e-20
                        target_data["-log10p"] = -np.log10(target_data["pearson_p_adjusted_clean"]).clip(0, max_log_p)
                    else:
                        # Calculate -log10(p-value)
                        target_data["-log10p"] = -np.log10(target_data["pearson_p_adjusted_clean"])

                # Set plot limits with safety checks for NaN/Inf
                max_effect = target_data["effect_size"].max()
                max_log_p = target_data["-log10p"].max()

                # Replace NaN/Inf with defaults
                if not np.isfinite(max_effect) or max_effect <= 0:
                    max_effect = 1.0
                    logger.warning("Invalid max effect size detected, using default value 1.0")
                else:
                    max_effect = max_effect * 1.1  # Add 10% margin

                if not np.isfinite(max_log_p) or max_log_p <= 0:
                    max_log_p = 4.0
                    logger.warning("Invalid max -log10(p) detected, using default value 4.0")
                else:
                    max_log_p = max_log_p * 1.1  # Add 10% margin

                # Ensure minimum reasonable ranges
                max_effect = max(1.0, max_effect)
                # Use realistic upper limit for -log10(p) in biological context
                max_log_p = min(20.0, max(4.0, max_log_p))

                # Create color mapping for significance
                significant = target_data["pearson_p_adjusted"] < 0.05

                # Improved color scale
                cmap = plt.cm.coolwarm_r
                colors = []

                for is_sig in significant:
                    if is_sig:
                        colors.append(cmap(0.8))  # Significant points (red)
                    else:
                        colors.append(cmap(0.2))  # Non-significant points (blue)

                # Enhanced scatter plot with better marker properties
                scatter = ax.scatter(
                    target_data["effect_size"] + target_data.get("jitter", 0),  # Add jitter if it exists
                    target_data["-log10p"],
                    alpha=0.8,
                    c=colors,
                    s=80,  # Larger points
                    edgecolor='w',  # White edge
                    linewidth=0.5,
                )

                # Add threshold line with improved styling
                threshold_line = ax.axhline(
                    y=-np.log10(0.05),
                    color="#666666",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                )

                # Add threshold annotation
                ax.text(
                    max_effect * 0.95,
                    -np.log10(0.05) * 1.1,
                    "p = 0.05",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#666666",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="#cccccc",
                        boxstyle="round,pad=0.2",
                    ),
                )

                # Highlight specific features if provided with better styling
                if highlight_features:
                    for feature in highlight_features:
                        feature_data = target_data[target_data["feature"] == feature]
                        if not feature_data.empty:
                            # Highlight with green, larger circle and black edge
                            ax.scatter(
                                feature_data["effect_size"],
                                feature_data["-log10p"],
                                color="#2ecc71",  # Green
                                s=120,  # Larger than regular points
                                alpha=0.8,
                                edgecolor="black",
                                linewidth=1.5,
                                zorder=10,  # Draw on top
                            )

                            # Add feature labels with better styling
                            for _, row in feature_data.iterrows():
                                ax.annotate(
                                    feature,
                                    (row["effect_size"], row["-log10p"]),
                                    xytext=(7, 7),
                                    textcoords="offset points",
                                    fontsize=10,
                                    fontweight="bold",
                                    color="#333333",
                                    bbox=dict(
                                        facecolor="white",
                                        alpha=0.7,
                                        edgecolor="#cccccc",
                                        boxstyle="round,pad=0.2",
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="->",
                                        color="#666666",
                                        shrinkA=0,
                                        shrinkB=5,
                                        connectionstyle="arc3,rad=.2",
                                    ),
                                )

                # Add grid for easier reading
                ax.grid(alpha=0.2, linestyle="--")

                # Add axis labels
                ax.set_xlabel(
                    "Effect Size (Correlation Magnitude)",
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                )
                ax.set_ylabel(
                    "-log₁₀(Adjusted p-value)",
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                )

                # Add a title for this specific target
                ax.set_title(
                    f"Feature Significance - Target: {target.upper()}",
                    fontsize=14,
                    fontweight="bold",
                    color="#333333",
                )

                # Set axis limits with some padding
                ax.set_xlim(-0.05, max_effect * 1.05)
                ax.set_ylim(-0.05, max_log_p * 1.05)

                # Improve axes appearance
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#dddddd")
                ax.spines["bottom"].set_color("#dddddd")

                # If using the jittered approach, update the legend explanation
                if "jitter" in target_data.columns:
                    ax.text(
                        0.98,
                        0.02,
                        "Note: Points jittered for visibility",
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=8,
                        fontstyle="italic",
                        color="#666666",
                    )

                # Annotate all significant features
                significant_features = target_data[target_data["is_significant"] == True]

                # If we have many significant features, we'll space them intelligently
                if len(significant_features) > 0:
                    # Sort by effect size to space labels better
                    significant_features = significant_features.sort_values("effect_size", ascending=False)

                    # Create positions for labels to avoid overlap
                    pad = max_effect * 0.05  # Space between feature labels

                    for i, (_, row) in enumerate(significant_features.iterrows()):
                        # Only annotate features with valid effect size
                        if np.isfinite(row["effect_size"]) and row["effect_size"] > 0:
                            # Alternate above/below for better spacing
                            vert_offset = (i % 2) * 2 - 1  # -1 or 1
                            horz_position = row["effect_size"] + row.get("jitter", 0)

                            ax.annotate(
                                row["feature"].replace("_", " ").replace("UTR", "UTR ").title(),
                                (horz_position, row["-log10p"]),
                                xytext=(0, 15 * vert_offset),
                                textcoords="offset points",
                                fontsize=9,
                                fontweight="bold",
                                color="#333333",
                                bbox=dict(
                                    facecolor="white",
                                    alpha=0.8,
                                    edgecolor="#dddddd",
                                    boxstyle="round,pad=0.2",
                                ),
                                arrowprops=dict(
                                    arrowstyle="-",
                                    color="#666666",
                                    shrinkA=0,
                                    shrinkB=5,
                                    connectionstyle="arc3,rad=0",
                                ),
                                ha="center",
                                va="center" if vert_offset > 0 else "top",
                            )

            # Add a subtitle explaining the plot
            fig.suptitle(
                "Volcano Plot: Statistical Significance vs Effect Size",
                fontsize=16,
                fontweight="bold",
                color="#333333",
                y=0.98,
            )

            fig.text(
                0.5,
                0.02,
                "Features above threshold line (p<0.05) are statistically significant",
                ha="center",
                fontsize=10,
                fontstyle="italic",
                color="#666666",
            )

            # Adjust layout to prevent overlapping
            fig.tight_layout(pad=2.0)
            fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.35)

            return fig

        except Exception as e:
            logger.exception(f"Error creating volcano plot: {e}")
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.text(0.5, 0.5, f"Error creating plot: {e!s}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Feature Significance - Error")
            return fig

    def plot_feature_correlation_matrix(
        self,
        feature_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        figsize: tuple[int, int] = (12, 10),
    ) -> Figure:
        """Generate a heatmap showing correlations between features and expression metrics.

        Args:
            feature_df: DataFrame with feature and expression data
            correlation_df: DataFrame with feature significance statistics
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if feature_df.empty or correlation_df.empty:
            logger.error("Empty DataFrames provided")
            return None

        try:
            # Get significant features
            sig_features = (
                correlation_df[correlation_df["is_significant"]]["feature"].unique().tolist()
            )

            if len(sig_features) == 0:
                logger.warning("No significant features found for correlation matrix")
                # Use top features instead
                sig_features = correlation_df.head(15)["feature"].unique().tolist()

            if len(sig_features) > 20:
                # Limit to top 20 features by effect size
                sig_features = (
                    correlation_df[correlation_df["feature"].isin(sig_features)]
                    .sort_values("effect_size", ascending=False)
                    .head(20)["feature"]
                    .tolist()
                )

            # Generate correlation matrix
            selected_cols = sig_features + ["cv", "mean"]
            valid_cols = [col for col in selected_cols if col in feature_df.columns]

            if len(valid_cols) < 3:
                logger.error("Insufficient valid columns for correlation matrix")
                return None

            corr_matrix = feature_df[valid_cols].corr()

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Generate heatmap
            cmap = cm.RdBu_r
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True

            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient")

            # Add labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr_matrix.columns)

            # Add values to cells
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if i > j:  # Only show lower triangle
                        text = ax.text(
                            j,
                            i,
                            f"{corr_matrix.iloc[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                            fontsize=8,
                        )

            ax.set_title("Feature Correlation Matrix")
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating correlation matrix: {e}")
            return None

    def plot_motif_enrichment(
        self, motif_df: pd.DataFrame, region: str = "5'UTR", figsize: tuple[int, int] = (10, 8)
    ) -> Figure:
        """Generate a bar chart showing significantly enriched motifs.

        Args:
            motif_df: DataFrame with motif enrichment statistics
            region: Region description (5'UTR or 3'UTR)
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if motif_df.empty:
            logger.error("Empty motif DataFrame provided")
            return None

        required_cols = ["motif", "fold_enrichment", "p_adjusted"]
        missing_cols = [col for col in required_cols if col not in motif_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns for motif plot: {missing_cols}")
            return None

        try:
            # Get top motifs (significant and highest fold enrichment)
            plot_data = motif_df[motif_df["p_adjusted"] < 0.05].head(15).copy()

            if plot_data.empty:
                logger.warning("No significant motifs found, using top motifs by fold enrichment")
                plot_data = motif_df.head(15).copy()

            # Sort by fold enrichment
            plot_data = plot_data.sort_values("fold_enrichment", ascending=True)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create colormap based on p-value
            colors = cm.viridis(
                np.log10(plot_data["p_adjusted"].clip(1e-10, 0.05)) / np.log10(1e-10)
            )

            # Plot bars
            bars = ax.barh(plot_data["motif"], plot_data["fold_enrichment"], color=colors)

            # Add labels
            ax.set_xlabel("Fold Enrichment")
            ax.set_ylabel("Motif Sequence")
            ax.set_title(f"Enriched Motifs in {region} of Consistently Expressed Genes")

            # Add p-value colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=Normalize(vmin=-10, vmax=-1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("log10(p-value)")
            cbar.set_ticks([-10, -8, -6, -4, -2, -1])
            cbar.set_ticklabels(["1e-10", "1e-8", "1e-6", "1e-4", "0.01", "0.1"])

            # Adjust layout
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating motif enrichment plot: {e}")
            return None

    def plot_feature_interaction_network(
        self, interaction_df: pd.DataFrame, top_n: int = 20, figsize: tuple[int, int] = (12, 10)
    ) -> Figure:
        """Generate a network plot of feature interactions.

        Args:
            interaction_df: DataFrame with feature interaction statistics
            top_n: Number of top interactions to display
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if interaction_df.empty:
            logger.error("Empty interaction DataFrame provided")
            return None

        required_cols = ["feature1", "feature2", "mutual_info", "correlation", "interaction_type"]
        missing_cols = [col for col in required_cols if col not in interaction_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns for network plot: {missing_cols}")
            return None

        try:
            # Get top interactions by mutual information
            plot_data = interaction_df.head(top_n).copy()

            if plot_data.empty:
                logger.warning("No interactions found for network plot")
                return None

            # Create a set of all unique features
            all_features = set(plot_data["feature1"].tolist() + plot_data["feature2"].tolist())

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create a mapping of features to positions
            import networkx as nx

            G = nx.Graph()

            # Add nodes
            for feature in all_features:
                # Get feature type (UTR5, UTR3, CDS)
                if "UTR5" in feature:
                    feature_type = "UTR5"
                elif "UTR3" in feature:
                    feature_type = "UTR3"
                elif "GC3" in feature or "CAI" in feature or "codon" in feature:
                    feature_type = "CDS"
                else:
                    feature_type = "Other"

                G.add_node(feature, type=feature_type)

            # Add edges
            for _, row in plot_data.iterrows():
                G.add_edge(
                    row["feature1"],
                    row["feature2"],
                    weight=row["mutual_info"],
                    corr=row["correlation"],
                    interaction=row["interaction_type"],
                )

            # Set node colors based on type
            node_colors = []
            for node in G.nodes():
                feature_type = G.nodes[node]["type"]
                if feature_type == "UTR5":
                    node_colors.append("tab:blue")
                elif feature_type == "UTR3":
                    node_colors.append("tab:orange")
                elif feature_type == "CDS":
                    node_colors.append("tab:green")
                else:
                    node_colors.append("tab:gray")

            # Set edge colors based on interaction type
            edge_colors = []
            for u, v, d in G.edges(data=True):
                if d["interaction"] == "synergistic":
                    edge_colors.append("tab:green")
                else:
                    edge_colors.append("tab:red")

            # Calculate edge widths based on mutual information
            edge_widths = [d["weight"] * 5 for _, _, d in G.edges(data=True)]

            # Draw the network
            pos = nx.spring_layout(G, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

            # Draw edges
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="tab:blue",
                    markersize=10,
                    label="5' UTR Feature",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="tab:orange",
                    markersize=10,
                    label="3' UTR Feature",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="tab:green",
                    markersize=10,
                    label="CDS Feature",
                ),
                Line2D([0], [0], color="tab:green", lw=2, label="Synergistic"),
                Line2D([0], [0], color="tab:red", lw=2, label="Antagonistic"),
            ]

            ax.legend(handles=legend_elements, loc="best")

            # Remove axis
            ax.set_axis_off()

            ax.set_title("Feature Interaction Network")
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating interaction network plot: {e}")
            return None
