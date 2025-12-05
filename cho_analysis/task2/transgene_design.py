# cho_analysis/task2/transgene_design.py
"""Transgene design recommendations based on sequence feature analysis.

This module provides functionality to generate optimal feature profiles,
specific optimization recommendations, and sequence templates for transgene design.
"""

import logging
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, patches
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class TransgeneDesignRecommendations:
    """Generates design recommendations for transgenes based on sequence analysis."""

    def __init__(self):
        """Initialize the transgene design class."""
        self.results = {}

    def generate_optimal_feature_profiles(
        self, feature_df: pd.DataFrame, prediction_model: Any | None = None
    ) -> pd.DataFrame:
        """Identify optimal ranges for sequence features.

        Args:
            feature_df: DataFrame with sequence features and expression metrics
            prediction_model: Optional trained prediction model

        Returns:
            DataFrame with optimal feature ranges for different expression goals
        """
        logger.info("Generating optimal feature profiles...")

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return pd.DataFrame()

        # Log available columns for debugging
        logger.debug(f"Input DataFrame columns: {feature_df.columns.tolist()}")

        # Check required columns and handle alternative names
        required_metrics = ["cv", "mean"]
        alternative_names = {"mean": "mean_expression"}

        # Create a working copy of the dataframe
        working_df = feature_df.copy()

        # If using alternative column names, rename them for processing
        for metric, alt_name in alternative_names.items():
            if metric not in working_df.columns and alt_name in working_df.columns:
                logger.info(f"Renaming column '{alt_name}' to '{metric}' for processing")
                working_df.rename(columns={alt_name: metric}, inplace=True)

        # Log working dataframe columns for debugging
        logger.debug(f"Working DataFrame columns after renaming: {working_df.columns.tolist()}")

        # Check if any required metrics are still missing
        missing_metrics = [col for col in required_metrics if col not in working_df.columns]

        if missing_metrics:
            logger.error(f"Missing required metrics: {missing_metrics}")
            return pd.DataFrame()

        # Features to consider by region
        utr5_features = [col for col in working_df.columns if col.startswith("UTR5_")]
        utr3_features = [col for col in working_df.columns if col.startswith("UTR3_")]
        cds_features = [
            col
            for col in working_df.columns
            if col.startswith("CDS_") or col in ["GC3_content", "CAI"]
        ]

        # Define three profiles: stability, expression, balanced
        profiles = ["max_stability", "max_expression", "balanced"]

        # Create result structure
        profile_data = []

        # Process each feature category
        categories = [("5UTR", utr5_features), ("CDS", cds_features), ("3UTR", utr3_features)]

        for region, features in categories:
            # Consider only numeric features
            numeric_features = [
                f for f in features if f in working_df.select_dtypes(include=np.number).columns
            ]

            for feature in numeric_features:
                # Skip features with too many missing values
                if working_df[feature].isna().sum() > 0.5 * len(working_df):
                    logger.warning(f"Skipping feature {feature} - too many missing values")
                    continue

                try:
                    # Calculate quantile ranges
                    q10 = working_df[feature].quantile(0.1)
                    q25 = working_df[feature].quantile(0.25)
                    q50 = working_df[feature].quantile(0.5)
                    q75 = working_df[feature].quantile(0.75)
                    q90 = working_df[feature].quantile(0.9)

                    # For max stability profile (low CV)
                    stability_subset = working_df.nsmallest(int(0.2 * len(working_df)), "cv")
                    stability_mean = stability_subset[feature].mean()
                    stability_std = stability_subset[feature].std()

                    # Fix: Ensure we have distinct min/max by adding small offsets when necessary
                    # Use more of the distribution (10-90 instead of just using one quantile value)
                    stability_min = max(stability_subset[feature].quantile(0.1), q10)
                    stability_max = min(stability_subset[feature].quantile(0.9), q90)

                    # If min and max are identical, create a small range around the mean
                    if abs(stability_min - stability_max) < 1e-6:
                        stability_range_width = max(0.05 * stability_mean, 1.0)  # At least 5% of mean or 1.0
                        stability_min = stability_mean - stability_range_width/2
                        stability_max = stability_mean + stability_range_width/2
                        logger.debug(f"Fixed identical stability range for {feature}: ({stability_min}, {stability_max})")

                    # For max expression profile (high mean)
                    expression_subset = working_df.nlargest(int(0.2 * len(working_df)), "mean")
                    expression_mean = expression_subset[feature].mean()
                    expression_std = expression_subset[feature].std()
                    expression_min = max(expression_subset[feature].quantile(0.1), q10)
                    expression_max = min(expression_subset[feature].quantile(0.9), q90)

                    # If min and max are identical, create a small range around the mean
                    if abs(expression_min - expression_max) < 1e-6:
                        expression_range_width = max(0.05 * expression_mean, 1.0)  # At least 5% of mean or 1.0
                        expression_min = expression_mean - expression_range_width/2
                        expression_max = expression_mean + expression_range_width/2
                        logger.debug(f"Fixed identical expression range for {feature}: ({expression_min}, {expression_max})")

                    # For balanced profile (good stability and expression)
                    # Define a composite score: rank on both cv (low) and mean (high)
                    working_df["_cv_rank"] = working_df["cv"].rank()
                    working_df["_mean_rank"] = working_df["mean"].rank(ascending=False)
                    working_df["_composite"] = working_df["_cv_rank"] + working_df["_mean_rank"]

                    balanced_subset = working_df.nsmallest(int(0.2 * len(working_df)), "_composite")
                    balanced_mean = balanced_subset[feature].mean()
                    balanced_std = balanced_subset[feature].std()
                    balanced_min = max(balanced_subset[feature].quantile(0.1), q10)
                    balanced_max = min(balanced_subset[feature].quantile(0.9), q90)

                    # If min and max are identical, create a small range around the mean
                    if abs(balanced_min - balanced_max) < 1e-6:
                        balanced_range_width = max(0.05 * balanced_mean, 1.0)  # At least 5% of mean or 1.0
                        balanced_min = balanced_mean - balanced_range_width/2
                        balanced_max = balanced_mean + balanced_range_width/2
                        logger.debug(f"Fixed identical balanced range for {feature}: ({balanced_min}, {balanced_max})")

                    # Clean up temporary columns
                    working_df = working_df.drop(["_cv_rank", "_mean_rank", "_composite"], axis=1)

                    # Add profile data
                    profile_data.append(
                        {
                            "feature": feature,
                            "region": region,
                            "population_mean": working_df[feature].mean(),
                            "population_median": q50,
                            "population_std": working_df[feature].std(),
                            "population_range_10_90": (q10, q90),
                            "stability_profile_mean": stability_mean,
                            "stability_profile_std": stability_std,
                            "stability_profile_range": (stability_min, stability_max),
                            "expression_profile_mean": expression_mean,
                            "expression_profile_std": expression_std,
                            "expression_profile_range": (expression_min, expression_max),
                            "balanced_profile_mean": balanced_mean,
                            "balanced_profile_std": balanced_std,
                            "balanced_profile_range": (balanced_min, balanced_max),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error processing feature {feature}: {e}")

        if not profile_data:
            logger.warning("No feature profiles could be generated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(profile_data)

        logger.info(f"Generated optimal profiles for {len(result_df)} features")
        self.results["feature_profiles"] = result_df

        return result_df

    def generate_specific_recommendations(self, feature_df: pd.DataFrame) -> dict[str, Any]:
        """Create concrete, actionable transgene design recommendations.

        Args:
            feature_df: DataFrame with sequence features and expression metrics

        Returns:
            Dictionary with specific design recommendations
        """
        logger.info("Generating specific optimization recommendations...")

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return {}

        # Create a working copy with consistent column names
        working_df = feature_df.copy()
        alternative_names = {"mean": "mean_expression"}

        # If using alternative column names, rename them for processing
        for metric, alt_name in alternative_names.items():
            if metric not in working_df.columns and alt_name in working_df.columns:
                working_df.rename(columns={alt_name: metric}, inplace=True)

        # Generate optimal feature profiles if not already done
        if "feature_profiles" not in self.results or self.results["feature_profiles"].empty:
            profiles_df = self.generate_optimal_feature_profiles(working_df)
            if profiles_df.empty:
                logger.error("Failed to generate feature profiles")
                return {}
        else:
            profiles_df = self.results["feature_profiles"]

        # Create recommendations structure
        recommendations = {"general": [], "5UTR": [], "CDS": [], "3UTR": [], "priority": []}

        # Process profiles to generate recommendations
        try:
            # 5' UTR recommendations
            utr5_profiles = profiles_df[profiles_df["region"] == "5UTR"]

            if not utr5_profiles.empty:
                # UTR5 length
                length_row = utr5_profiles[utr5_profiles["feature"] == "UTR5_length"]
                if not length_row.empty:
                    row = length_row.iloc[0]
                    balanced_range = row["balanced_profile_range"]
                    recommendations["5UTR"].append(
                        f"Keep 5' UTR length between {int(balanced_range[0])} and {int(balanced_range[1])} nucleotides"
                    )

                # UTR5 GC content
                gc_row = utr5_profiles[utr5_profiles["feature"] == "UTR5_GC"]
                if not gc_row.empty:
                    row = gc_row.iloc[0]
                    balanced_range = row["balanced_profile_range"]
                    recommendations["5UTR"].append(
                        f"Maintain 5' UTR GC content between {balanced_range[0]:.1f}% and {balanced_range[1]:.1f}%"
                    )

                # General recommendation about 5' UTR
                recommendations["general"].append(
                    "Design a 5' UTR with optimal length and GC content to ensure efficient translation initiation"
                )

            # CDS recommendations
            cds_profiles = profiles_df[profiles_df["region"] == "CDS"]

            if not cds_profiles.empty:
                # CAI
                cai_row = cds_profiles[cds_profiles["feature"] == "CAI"]
                if not cai_row.empty:
                    row = cai_row.iloc[0]
                    balanced_mean = row["balanced_profile_mean"]
                    recommendations["CDS"].append(
                        f"Target a Codon Adaptation Index (CAI) of approximately {balanced_mean:.2f}"
                    )
                    recommendations["priority"].append("Codon optimization to achieve target CAI")

                # GC3 content
                gc3_row = cds_profiles[cds_profiles["feature"] == "GC3_content"]
                if not gc3_row.empty:
                    row = gc3_row.iloc[0]
                    balanced_range = row["balanced_profile_range"]
                    recommendations["CDS"].append(
                        f"Maintain GC content at the third codon position (GC3) between {balanced_range[0]:.1f}% and {balanced_range[1]:.1f}%"
                    )

                # General recommendation about codon usage
                recommendations["general"].append(
                    "Optimize codon usage based on CHO-specific preferences while maintaining moderate GC content"
                )

            # 3' UTR recommendations
            utr3_profiles = profiles_df[profiles_df["region"] == "3UTR"]

            if not utr3_profiles.empty:
                # UTR3 length
                length_row = utr3_profiles[utr3_profiles["feature"] == "UTR3_length"]
                if not length_row.empty:
                    row = length_row.iloc[0]
                    balanced_range = row["balanced_profile_range"]
                    recommendations["3UTR"].append(
                        f"Design 3' UTR with length between {int(balanced_range[0])} and {int(balanced_range[1])} nucleotides"
                    )

                # UTR3 GC content
                gc_row = utr3_profiles[utr3_profiles["feature"] == "UTR3_GC"]
                if not gc_row.empty:
                    row = gc_row.iloc[0]
                    balanced_range = row["balanced_profile_range"]
                    recommendations["3UTR"].append(
                        f"Maintain 3' UTR GC content between {balanced_range[0]:.1f}% and {balanced_range[1]:.1f}%"
                    )

                # General recommendation about 3' UTR
                recommendations["general"].append(
                    "Include a 3' UTR with appropriate stability elements to ensure mRNA longevity"
                )

            # Add motif recommendations
            # These would be based on motif enrichment results if available
            recommendations["5UTR"].append(
                "Include a strong Kozak consensus sequence (GCCACC) around the start codon"
            )

            recommendations["3UTR"].append(
                "Include at least one canonical polyadenylation signal (AATAAA) in the 3' UTR"
            )

            # Add priority list (top 5 recommendations)
            if len(recommendations["priority"]) < 5:
                # Add general recommendations to priority if needed
                for rec in recommendations["general"]:
                    if (
                        rec not in recommendations["priority"]
                        and len(recommendations["priority"]) < 5
                    ):
                        recommendations["priority"].append(rec)

                # Add region-specific recommendations to priority if needed
                for region in ["5UTR", "CDS", "3UTR"]:
                    for rec in recommendations[region]:
                        if (
                            rec not in recommendations["priority"]
                            and len(recommendations["priority"]) < 5
                        ):
                            recommendations["priority"].append(rec)

        except Exception as e:
            logger.exception(f"Error generating recommendations: {e}")

        if all(not recommendations[key] for key in recommendations):
            logger.warning("No recommendations could be generated")
            return {}

        logger.info(
            f"Generated {sum(len(recommendations[key]) for key in recommendations)} recommendations"
        )
        self.results["recommendations"] = recommendations

        return recommendations

    def generate_sequence_templates(
        self, optimal_features: dict[str, Any], target_protein: str | None = None
    ) -> dict[str, str]:
        """Create template sequences incorporating optimal features.

        Args:
            optimal_features: Dictionary with optimal feature values
            target_protein: Optional target protein sequence

        Returns:
            Dictionary with template sequences for each region
        """
        logger.info("Generating sequence templates...")

        # NOTE: This would be implemented with actual algorithms for sequence generation
        # This is a simplified placeholder implementation

        templates = {
            "5UTR_template": "GCCGCCACCAUGG",  # Strong Kozak context
            "3UTR_template": "AAUAAAGAAUUUCUGAUUUUUUUAAAAA",  # PolyA signal
            "CDS_optimized": "",
        }

        if target_protein:
            # NOTE: In a real implementation, this would run codon optimization
            # based on the optimal features
            templates["CDS_optimized"] = f"Optimized sequence for: {target_protein[:20]}..."

        logger.info("Generated template sequences")
        self.results["sequence_templates"] = templates

        return templates

    def analyze_optimization_impact(
        self, feature_ranges: dict[str, tuple[float, float]], prediction_model: Any | None = None
    ) -> pd.DataFrame:
        """Estimate expression improvement from each optimization.

        Args:
            feature_ranges: Dictionary of feature ranges
            prediction_model: Optional prediction model

        Returns:
            DataFrame with projected impact of optimizations
        """
        logger.info("Analyzing optimization impact...")

        # NOTE: This would use the prediction model to estimate the impact
        # of each optimization step
        # This is a simplified placeholder implementation

        if not feature_ranges:
            logger.error("No feature ranges provided")
            return pd.DataFrame()

        # Create impact data
        impact_data = []

        for i, (feature, (min_val, max_val)) in enumerate(feature_ranges.items()):
            # NOTE: Simulate impact (in a real implementation, this would use the model)
            # Order by descending impact
            impact = (10 - i * 0.5) if i < 10 else 0.5

            impact_data.append(
                {
                    "feature": feature,
                    "optimization": f"Optimize {feature} to {(min_val + max_val)/2:.2f}",
                    "estimated_impact": impact,
                    "cumulative_impact": sum(row["estimated_impact"] for row in impact_data)
                    + impact,
                    "priority": i + 1,
                }
            )

        if not impact_data:
            logger.warning("No optimization impact data could be generated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(impact_data)

        # Remove intermediate columns used for calculations
        result_df = result_df.drop(columns=["mean_rank", "cv_rank"], errors="ignore")

        logger.info(f"Estimated impact for {len(result_df)} optimizations")
        self.results["optimization_impact"] = result_df

        return result_df


class TransgeneDesignVisualization:
    """Creates visualizations for transgene design recommendations."""

    def __init__(self):
        """Initialize the visualization class."""
        self.figures = {}

    def plot_optimal_feature_ranges(
        self, profiles_df: pd.DataFrame, figsize: tuple[int, int] = (12, 8)
    ) -> Figure | None:
        """Generate violin plots showing optimal ranges for key features.

        Args:
            profiles_df: DataFrame with feature profiles
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if profiles_df.empty:
            logger.error("Empty profiles DataFrame provided")
            return None

        try:
            # Select important features to plot
            key_features = [
                "UTR5_length",
                "UTR5_GC",
                "CAI",
                "GC3_content",
                "UTR3_length",
                "UTR3_GC",
            ]

            features_to_plot = [f for f in key_features if f in profiles_df["feature"].values]

            if not features_to_plot:
                logger.error("No key features found in profiles")
                return None

            # Create figure with professional styling
            fig, axes = plt.subplots(
                nrows=len(features_to_plot),
                ncols=1,
                figsize=(figsize[0], figsize[1] * len(features_to_plot) / 3),
                squeeze=False,
                dpi=300,
            )

            # Set background color
            fig.patch.set_facecolor("#f8f9fa")

            # Flatten axes array
            axes = axes.flatten()

            # Create plots
            for i, feature in enumerate(features_to_plot):
                if i >= len(axes):
                    break

                ax = axes[i]
                ax.set_facecolor("#f8f9fa")

                # Get feature data
                feature_row = profiles_df[profiles_df["feature"] == feature].iloc[0]

                # Get ranges - using the correct column names from the data
                expression_range = feature_row.get("expression_profile_range", (0, 0))
                stability_range = feature_row.get("stability_profile_range", (0, 0))
                balanced_range = feature_row.get("balanced_profile_range", (0, 0))

                # Add debug logging to help diagnose issues
                logger.debug(f"Feature {feature} ranges:")
                logger.debug(f"  - Expression range: {expression_range}")
                logger.debug(f"  - Stability range: {stability_range}")
                logger.debug(f"  - Balanced range: {balanced_range}")

                # Ensure ranges are tuples with 2 elements and values make sense
                if not isinstance(expression_range, tuple) or len(expression_range) != 2:
                    expression_range = (0, 0)
                    logger.warning(f"Invalid expression_profile_range for {feature}")

                if not isinstance(stability_range, tuple) or len(stability_range) != 2:
                    stability_range = (0, 0)
                    logger.warning(f"Invalid stability_profile_range for {feature}")

                if not isinstance(balanced_range, tuple) or len(balanced_range) != 2:
                    balanced_range = (0, 0)
                    logger.warning(f"Invalid balanced_profile_range for {feature}")

                # Validate ranges are biologically meaningful
                for range_name, range_val in [
                    ("expression_range", expression_range),
                    ("stability_range", stability_range),
                    ("balanced_range", balanced_range)
                ]:
                    if range_val[0] == range_val[1]:
                        logger.warning(f"{range_name} for {feature} has same min and max: {range_val}")
                    if range_val[0] > range_val[1]:
                        logger.warning(f"{range_name} for {feature} has min > max: {range_val}")
                        # Fix the range by swapping values
                        if range_name == "expression_range":
                            expression_range = (range_val[1], range_val[0])
                        elif range_name == "stability_range":
                            stability_range = (range_val[1], range_val[0])
                        elif range_name == "balanced_range":
                            balanced_range = (range_val[1], range_val[0])

                # Ensure ranges have sufficient width for visualization
                min_width = 1.0  # Increased minimum width for better visualization
                expression_range = self.ensure_range_width(expression_range, min_width, feature_row.get("expression_profile_mean", 0))
                stability_range = self.ensure_range_width(stability_range, min_width, feature_row.get("stability_profile_mean", 0))
                balanced_range = self.ensure_range_width(balanced_range, min_width, feature_row.get("balanced_profile_mean", 0))

                # Plot ranges as violin plots
                positions = [1, 2, 3]
                violins = ax.violinplot(
                    [
                        [expression_range[0], expression_range[1]],
                        [stability_range[0], stability_range[1]],
                        [balanced_range[0], balanced_range[1]],
                    ],
                    positions,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )

                # Customize violins with better colors
                violin_colors = ["#5295e3", "#53bb45", "#c44e52"]
                for viol, color in zip(violins["bodies"], violin_colors):
                    viol.set_facecolor(color)
                    viol.set_edgecolor(color)
                    viol.set_alpha(0.7)

                # Add center lines for each violin
                ax.hlines(
                    [
                        (expression_range[0] + expression_range[1]) / 2,
                        (stability_range[0] + stability_range[1]) / 2,
                        (balanced_range[0] + balanced_range[1]) / 2,
                    ],
                    [p - 0.15 for p in positions],
                    [p + 0.15 for p in positions],
                    colors=["#3a6ca8", "#3a8431", "#8c363a"],
                    linestyles="solid",
                    linewidths=2,
                )

                # Add range values with improved alignment and formatting
                # Format for high expression range
                ax.text(
                    1,
                    expression_range[0] - (expression_range[1] - expression_range[0]) * 0.15,
                    self._format_value(expression_range[0], feature),
                    ha="center",
                    va="top",
                    color="#3a6ca8",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )
                ax.text(
                    1,
                    expression_range[1] + (expression_range[1] - expression_range[0]) * 0.15,
                    self._format_value(expression_range[1], feature),
                    ha="center",
                    va="bottom",
                    color="#3a6ca8",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )

                # Format for stable expression range
                ax.text(
                    2,
                    stability_range[0] - (stability_range[1] - stability_range[0]) * 0.15,
                    self._format_value(stability_range[0], feature),
                    ha="center",
                    va="top",
                    color="#3a8431",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )
                ax.text(
                    2,
                    stability_range[1] + (stability_range[1] - stability_range[0]) * 0.15,
                    self._format_value(stability_range[1], feature),
                    ha="center",
                    va="bottom",
                    color="#3a8431",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )

                # Format for balanced profile range
                ax.text(
                    3,
                    balanced_range[0] - (balanced_range[1] - balanced_range[0]) * 0.15,
                    self._format_value(balanced_range[0], feature),
                    ha="center",
                    va="top",
                    color="#8c363a",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )
                ax.text(
                    3,
                    balanced_range[1] + (balanced_range[1] - balanced_range[0]) * 0.15,
                    self._format_value(balanced_range[1], feature),
                    ha="center",
                    va="bottom",
                    color="#8c363a",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3", edgecolor="#dddddd"),
                )

                # Format axes with more space for labels
                ax.set_ylabel(
                    feature.replace("_", " ").title(),
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                )

                # Add some padding to ensure better display of category names
                ax.set_xticks(positions)
                ax.set_xticklabels(
                    ["High Expression", "Stable Expression", "Balanced Profile"],
                    fontsize=11,
                    fontweight="bold",
                )

                # Ensure x-axis labels are displayed properly
                plt.setp(ax.get_xticklabels(), ha="center", rotation=0)

                # Make sure there's enough room at the bottom for labels
                ax.tick_params(axis="y", labelsize=10)
                ax.tick_params(axis="x", pad=10)

                # Remove top and right spines for cleaner look
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#dddddd")
                ax.spines["bottom"].set_color("#dddddd")

                # Adjust y-axis limits to accommodate the labels with more padding
                current_ymin, current_ymax = ax.get_ylim()
                range_height = current_ymax - current_ymin
                ax.set_ylim(current_ymin - range_height * 0.1, current_ymax + range_height * 0.1)

                # Add grid with lighter style
                ax.grid(alpha=0.2, axis="y", linestyle="--")

            # Set title for the figure with improved styling
            fig.suptitle(
                "Optimal Feature Ranges for Transgene Design",
                fontsize=16,
                fontweight="bold",
                color="#333333",
                y=0.98,
            )

            # Adjust layout with more space for x-axis labels
            fig.tight_layout(pad=2.0)
            fig.subplots_adjust(top=0.95, hspace=0.6, bottom=0.1)

            return fig

        except Exception as e:
            logger.exception(f"Error creating optimal feature ranges plot: {e}")
            return None

    def plot_design_rule_flowchart(
        self, recommendations: dict[str, list[str]], figsize: tuple[int, int] = (14, 12)
    ) -> Figure | None:
        """Generate a visual flowchart of design rules.

        Args:
            recommendations: Dictionary with design recommendations
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object or None if error occurs
        """
        # Skip visualization as per requirement
        logger.info("Skipping design rule flowchart visualization as requested")
        return None

    def plot_sequence_templates(
        self, templates: dict[str, str], figsize: tuple[int, int] = (14, 8)
    ) -> Figure | None:
        """Generate sequence logo plots for optimal region templates.

        Args:
            templates: Dictionary with template sequences
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if not templates:
            logger.error("No templates provided")
            return None

        try:
            # Create figure with increased size for better readability
            fig, axes = plt.subplots(
                nrows=len(templates),
                ncols=1,
                figsize=figsize,
                squeeze=False,
                dpi=300
            )

            # Set background color
            fig.patch.set_facecolor("#f8f9fa")

            # Flatten axes array
            axes = axes.flatten()

            # Create a simple sequence visualization for each template
            for i, (region, sequence) in enumerate(templates.items()):
                if i >= len(axes):
                    break

                ax = axes[i]
                ax.set_facecolor("#f8f9fa")

                # Skip if sequence is too long or empty
                if not sequence or len(sequence) > 1000:
                    ax.text(
                        0.5,
                        0.5,
                        f"Template for {region} (too long to display)",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.axis("off")
                    continue

                # Create a simple visualization of the sequence
                nucleotides = list(sequence)
                x = np.arange(len(nucleotides))

                # Color map for nucleotides - using more visually distinct colors
                colors = {"A": "#3498db", "T": "#e74c3c", "U": "#e74c3c", "G": "#2ecc71", "C": "#f39c12"}

                # Draw a background container for the sequence
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (-1, -1.5),
                        len(nucleotides) + 2,
                        3,
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="#dddddd",
                        linewidth=1,
                        zorder=0,
                        alpha=0.7
                    )
                )

                # Draw each nucleotide
                for j, nt in enumerate(nucleotides):
                    color = colors.get(nt.upper(), "gray")
                    # Add circular background for each nucleotide
                    ax.add_patch(
                        patches.Circle(
                            (j, 0),
                            radius=0.4,
                            facecolor=color,
                            alpha=0.2,
                            edgecolor=color,
                            linewidth=1,
                            zorder=1,
                        )
                    )
                    # Add the nucleotide text
                    ax.text(
                        j,
                        0,
                        nt,
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=color,
                        zorder=2,
                    )

                # Highlight key elements if present with improved styling
                if region == "5UTR_template":
                    # Highlight Kozak sequence
                    kozak_match = re.search(r"GCCACC", sequence, re.IGNORECASE)
                    if kozak_match:
                        start, end = kozak_match.span()
                        ax.add_patch(
                            patches.Rectangle(
                                (start - 0.5, -0.6),
                                end - start,
                                1.2,
                                facecolor="lightgreen",
                                alpha=0.4,
                                edgecolor="green",
                                linewidth=1.5,
                                zorder=0,
                                linestyle="--",
                            )
                        )
                        ax.text(
                            (start + end) / 2,
                            -1,
                            "Kozak Sequence",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="green",
                            fontweight="bold",
                            bbox=dict(
                                facecolor="white",
                                alpha=0.7,
                                edgecolor="green",
                                boxstyle="round,pad=0.3",
                            ),
                        )

                elif region == "3UTR_template":
                    # Highlight polyA signal
                    polya_match = re.search(r"AATAAA", sequence, re.IGNORECASE)
                    if polya_match:
                        start, end = polya_match.span()
                        ax.add_patch(
                            patches.Rectangle(
                                (start - 0.5, -0.6),
                                end - start,
                                1.2,
                                facecolor="lightyellow",
                                alpha=0.4,
                                edgecolor="orange",
                                linewidth=1.5,
                                zorder=0,
                                linestyle="--",
                            )
                        )
                        ax.text(
                            (start + end) / 2,
                            -1,
                            "PolyA Signal",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="orange",
                            fontweight="bold",
                            bbox=dict(
                                facecolor="white",
                                alpha=0.7,
                                edgecolor="orange",
                                boxstyle="round,pad=0.3",
                            ),
                        )

                # Set limits and labels
                ax.set_xlim(-1, len(nucleotides))
                ax.set_ylim(-1.5, 1.5)

                # Add region title with better styling
                ax.text(
                    -1,
                    1,
                    f"{region}",
                    ha="left",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="#333333",
                )

                # Add length information
                ax.text(
                    len(nucleotides),
                    1,
                    f"Length: {len(nucleotides)} nt",
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="#666666",
                    style="italic",
                )

                ax.axis("off")

            # Set overall title with better styling
            fig.suptitle(
                "Optimized Sequence Templates",
                fontsize=16,
                fontweight="bold",
                color="#333333",
                y=0.98,
            )

            # Adjust layout with better spacing
            fig.tight_layout(pad=1.5)
            fig.subplots_adjust(top=0.95, hspace=0.4)

            return fig

        except Exception as e:
            logger.exception(f"Error creating sequence templates plot: {e}")
            return None

    def plot_optimization_impact(
        self, impact_df: pd.DataFrame, figsize: tuple[int, int] = (10, 6)
    ) -> Figure | None:
        """Generate a waterfall chart showing cumulative impact of optimizations.

        Args:
            impact_df: DataFrame with optimization impact data
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object or None if error occurs
        """
        if impact_df.empty:
            logger.error("Empty impact DataFrame provided")
            return None

        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Sort by priority
            plot_data = impact_df.sort_values("priority").head(8)

            # Extract data
            features = plot_data["feature"].tolist()
            impacts = plot_data["estimated_impact"].tolist()

            # Calculate positions for waterfall chart
            bottoms = np.zeros(len(impacts))
            for i in range(1, len(impacts)):
                bottoms[i] = bottoms[i - 1] + impacts[i - 1]

            # Create bars
            colormap = cm.get_cmap("viridis", len(features))
            colors = [colormap(i) for i in np.linspace(0, 1, len(features))]

            bars = ax.bar(
                range(len(features)),
                impacts,
                bottom=bottoms,
                width=0.6,
                alpha=0.7,
                color=colors,
            )

            # Add cumulative impact line
            cumulative = np.cumsum(impacts)
            ax.plot(range(len(features)), cumulative, "ro-", alpha=0.7)

            # Add labels
            for i, (bar, impact) in enumerate(zip(bars, impacts, strict=False)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"+{impact:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

                # Add cumulative label
                ax.text(
                    i,
                    cumulative[i] + 0.5,
                    f"{cumulative[i]:.1f}%",
                    ha="center",
                    va="bottom",
                    color="red",
                )

            # Set labels and title
            ax.set_ylabel("Expression Improvement (%)")
            ax.set_title("Cumulative Impact of Optimization Steps")

            # Set x-tick labels
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha="right")

            # Add grid
            ax.grid(alpha=0.3, axis="y")

            # Adjust layout
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating optimization impact plot: {e}")
            return None

    def ensure_range_width(self, range_tuple: tuple[float, float], min_width: float, mean_value: float) -> tuple[float, float]:
        """Ensure a range has a minimum width for visualization purposes.

        Args:
            range_tuple: The original (min, max) range tuple
            min_width: The minimum width the range should have
            mean_value: A central value to expand around if needed

        Returns:
            A tuple with adjusted (min, max) values
        """
        min_val, max_val = range_tuple

        # If we have identical values or min > max
        if min_val >= max_val:
            # If we have a mean value, expand around it
            if mean_value != 0:
                half_width = min_width / 2
                return (mean_value - half_width, mean_value + half_width)
            # Otherwise, make max slightly larger than min
            else:
                if min_val == 0:
                    return (0, min_width)
                else:
                    return (min_val, min_val + min_width)

        # If range is too narrow, expand it
        if max_val - min_val < min_width:
            mid_point = (min_val + max_val) / 2
            half_width = min_width / 2
            return (mid_point - half_width, mid_point + half_width)

        # Otherwise, return the original range
        return range_tuple

    def _format_value(self, value: float, feature: str) -> str:
        """Format values for display in plots based on feature type.

        Args:
            value: The numeric value to format
            feature: The feature name (to determine formatting)

        Returns:
            Formatted string representation of the value
        """
        # Handle length features - format as integers
        if "_length" in feature:
            return f"{int(value)}"

        # Handle GC content features - format with 1 decimal place
        if "GC" in feature:
            return f"{value:.1f}%"

        # Handle CAI - format with 2 decimal places
        if feature == "CAI":
            return f"{value:.2f}"

        # Handle zero or very small values
        if abs(value) < 0.001:
            return "0"

        # Default formatting with 1 decimal place
        return f"{value:.1f}"
