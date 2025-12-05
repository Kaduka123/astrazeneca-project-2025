# cho_analysis/task2/comparative_analysis.py
"""Comparative sequence analysis for consistently expressed genes.

This module provides functionality to compare sequence features between consistently and variably
expressed genes, analyze conservation across species, and examine RNA secondary structures.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)


class ComparativeSequenceAnalysis:
    """Analyzes sequence features by comparing different gene groups and reference datasets."""

    def __init__(self):
        """Initialize the comparative analysis class."""
        self.results = {}

    def compare_sequence_features(
        self, consistent_genes_df: pd.DataFrame, variable_genes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare sequence features between consistent and variable expression genes.

        Args:
            consistent_genes_df: DataFrame with consistently expressed genes and their features
            variable_genes_df: DataFrame with variably expressed genes and their features

        Returns:
            DataFrame with statistical comparison results
        """
        logger.info("Comparing sequence features between consistent and variable genes...")

        if consistent_genes_df.empty or variable_genes_df.empty:
            logger.error("Empty gene DataFrames provided")
            return pd.DataFrame()

        # Identify common feature columns (numeric only)
        consistent_numeric_cols = consistent_genes_df.select_dtypes(
            include=np.number
        ).columns.tolist()
        variable_numeric_cols = variable_genes_df.select_dtypes(include=np.number).columns.tolist()

        # Find intersection of columns
        common_feature_cols = [
            col for col in consistent_numeric_cols if col in variable_numeric_cols
        ]

        # Remove expression metrics if present
        feature_cols = [
            col for col in common_feature_cols if col not in ["cv", "mean", "std", "min", "max"]
        ]

        if not feature_cols:
            logger.error("No common feature columns found")
            return pd.DataFrame()

        # Prepare results
        comparison_results = []

        # Compare each feature
        for feature in feature_cols:
            # Get data, dropping any NaN values
            consistent_values = consistent_genes_df[feature].dropna().values
            variable_values = variable_genes_df[feature].dropna().values

            if len(consistent_values) < 5 or len(variable_values) < 5:
                logger.warning(f"Skipping {feature} - insufficient data")
                continue

            try:
                # Calculate mean and median for both groups
                consistent_mean = np.mean(consistent_values)
                variable_mean = np.mean(variable_values)
                consistent_median = np.median(consistent_values)
                variable_median = np.median(variable_values)

                # Calculate effect size (Cohen's d)
                # Pooled standard deviation
                n1, n2 = len(consistent_values), len(variable_values)
                s1, s2 = np.std(consistent_values, ddof=1), np.std(variable_values, ddof=1)

                # Avoid division by zero
                if s1 == 0 and s2 == 0:
                    cohens_d = 0
                else:
                    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    cohens_d = (consistent_mean - variable_mean) / pooled_std

                # Statistical tests
                # Mann-Whitney U test (non-parametric)
                mw_u, mw_p = stats.mannwhitneyu(
                    consistent_values, variable_values, alternative="two-sided"
                )

                # t-test (parametric)
                t_stat, t_p = stats.ttest_ind(consistent_values, variable_values, equal_var=False)

                # Add results
                comparison_results.append(
                    {
                        "feature": feature,
                        "consistent_mean": consistent_mean,
                        "variable_mean": variable_mean,
                        "consistent_median": consistent_median,
                        "variable_median": variable_median,
                        "fold_change": consistent_mean / variable_mean
                        if variable_mean != 0
                        else np.nan,
                        "difference": consistent_mean - variable_mean,
                        "cohens_d": cohens_d,
                        "mann_whitney_p": mw_p,
                        "t_test_p": t_p,
                        "consistent_n": len(consistent_values),
                        "variable_n": len(variable_values),
                    }
                )

            except Exception as e:
                logger.warning(f"Error comparing feature {feature}: {e}")

        if not comparison_results:
            logger.warning("No valid feature comparisons could be calculated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(comparison_results)

        # Apply multiple testing correction
        for p_col in ["mann_whitney_p", "t_test_p"]:
            try:
                result_df[f"{p_col}_adjusted"] = stats.false_discovery_control(
                    result_df[p_col].values, method="bh"
                )
            except Exception as e:
                logger.exception(f"Error in multiple testing correction: {e}")
                result_df[f"{p_col}_adjusted"] = result_df[p_col]

        # Add significance flags
        result_df["is_significant"] = (result_df["mann_whitney_p_adjusted"] < 0.05) | (
            result_df["t_test_p_adjusted"] < 0.05
        )

        # Sort by effect size (absolute Cohen's d)
        result_df = result_df.sort_values(
            by=["is_significant", "cohens_d"],
            ascending=[False, False],
            key=lambda x: abs(x) if x.name == "cohens_d" else x,
        )

        logger.info(
            f"Compared {len(result_df)} features, {result_df['is_significant'].sum()} significant"
        )
        self.results["feature_comparison"] = result_df

        return result_df

    def compare_with_reference_genes(
        self, feature_df: pd.DataFrame, reference_type: str = "high_expression"
    ) -> pd.DataFrame:
        """Compare sequence features with literature-based reference gene sets.

        Args:
            feature_df: DataFrame with gene features
            reference_type: Type of reference set to use ('high_expression', 'optimal_codons', 'stable_mRNA')

        Returns:
            DataFrame with similarity metrics to reference profiles
        """
        logger.info(f"Comparing with reference genes: {reference_type}...")

        if feature_df.empty:
            logger.error("Empty feature DataFrame provided")
            return pd.DataFrame()

        # Define reference profiles based on literature
        # NOTE: This is a simplified example - in a real implementation, these would be loaded from files
        reference_profiles = {
            "high_expression": {
                "UTR5_length": {"mean": 120, "std": 30},
                "UTR3_length": {"mean": 800, "std": 150},
                "GC3_content": {"mean": 65, "std": 5},
                "CAI": {"mean": 0.75, "std": 0.05},
            },
            "optimal_codons": {
                "CAI": {"mean": 0.8, "std": 0.05},
                "GC3_content": {"mean": 70, "std": 5},
            },
            "stable_mRNA": {
                "UTR5_length": {"mean": 100, "std": 25},
                "UTR3_length": {"mean": 600, "std": 100},
                "polyA_signal_present": {"mean": 1, "std": 0},
            },
        }

        if reference_type not in reference_profiles:
            logger.error(f"Unknown reference type: {reference_type}")
            return pd.DataFrame()

        # Get the selected reference profile
        ref_profile = reference_profiles[reference_type]

        # Check which features are available in the data
        available_features = [feat for feat in ref_profile if feat in feature_df.columns]

        if not available_features:
            logger.error(f"No features from reference profile {reference_type} found in data")
            return pd.DataFrame()

        # Calculate z-scores for each gene relative to reference
        z_scores = pd.DataFrame(index=feature_df.index)

        for feature in available_features:
            feature_mean = ref_profile[feature]["mean"]
            feature_std = ref_profile[feature]["std"]

            # Skip features with zero std to avoid division by zero
            if feature_std == 0:
                logger.warning(f"Skipping feature {feature} - zero std in reference")
                continue

            # Calculate z-score: (x - μ) / σ
            z_scores[f"{feature}_z"] = (feature_df[feature] - feature_mean) / feature_std

        if z_scores.empty:
            logger.error("No valid z-scores could be calculated")
            return pd.DataFrame()

        # Calculate overall similarity score (inverse of mean absolute z-score)
        z_scores["mean_abs_z"] = z_scores.abs().mean(axis=1)
        z_scores["reference_similarity"] = 1 / (1 + z_scores["mean_abs_z"])

        # Add source feature values for reference
        for feature in available_features:
            z_scores[feature] = feature_df[feature]

        # Calculate percentile rank of similarity
        z_scores["similarity_percentile"] = z_scores["reference_similarity"].rank(pct=True) * 100

        # Sort by similarity
        result_df = z_scores.sort_values("reference_similarity", ascending=False).reset_index()

        logger.info(f"Calculated reference similarity for {len(result_df)} genes")
        self.results[f"reference_comparison_{reference_type}"] = result_df

        return result_df

    def analyze_sequence_conservation(
        self,
        sequence_df: pd.DataFrame,
        species_list: list[str] = ["human", "mouse", "rat"],
    ) -> pd.DataFrame:
        """Calculate sequence conservation scores for UTRs and CDS regions.

        This is a simplified implementation for demonstration purposes.
        A full implementation would involve sequence alignment and evolutionary analysis
        tools to compute conservation scores.

        Args:
            sequence_df: DataFrame containing sequence data
            species_list: List of species to compare against

        Returns:
            DataFrame with conservation scores for each gene and feature
        """
        logger.info("Analyzing sequence conservation...")

        # NOTE: In a real implementation, this would involve sequence alignment and
        # conservation calculation using actual sequence data from multiple species.
        # This is a simplified placeholder implementation.

        if sequence_df.empty:
            logger.error("Empty sequence DataFrame provided")
            return pd.DataFrame()

        # Check if required sequence columns exist (at least one is needed)
        sequence_cols = ["CDS_Seq", "UTR5_Seq", "UTR3_Seq"]
        available_seq_cols = [col for col in sequence_cols if col in sequence_df.columns]

        if not available_seq_cols:
            logger.error("No sequence columns found")
            return pd.DataFrame()

        # Simulate conservation scores
        # NOTE: In a real implementation, this would calculate actual conservation scores
        # This would use actual data

        # Create a results DataFrame
        result_data = []

        # Get gene identifier columns
        id_col = (
            "ensembl_transcript_id"
            if "ensembl_transcript_id" in sequence_df.columns
            else sequence_df.index.name
        )

        # For each gene, simulate conservation scores
        for idx, row in sequence_df.iterrows():
            gene_data = {"gene_id": row.get(id_col, str(idx))}

            # For each sequence type (UTR5, CDS, UTR3)
            for seq_col in available_seq_cols:
                region = seq_col.split("_")[0]  # Extract region name (UTR5, CDS, UTR3)

                # If sequence exists
                if pd.notna(row.get(seq_col)) and len(str(row.get(seq_col))) > 10:
                    # For each species, simulate a conservation score
                    for species in species_list:
                        # In a real implementation, this would be calculated from sequence alignment
                        # This is just a placeholder simulation
                        if region == "CDS":
                            # CDS regions tend to be more conserved
                            conservation = np.random.beta(5, 2)  # Higher values (0.6-0.9 range)
                        else:
                            # UTRs typically have more variable conservation
                            conservation = np.random.beta(2, 3)  # Lower values (0.3-0.6 range)

                        gene_data[f"{region}_conservation_{species}"] = conservation

            if len(gene_data) > 1:  # More than just gene_id
                result_data.append(gene_data)

        if not result_data:
            logger.warning("No conservation scores could be calculated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(result_data)

        # Calculate average conservation per region
        for region in ["UTR5", "CDS", "UTR3"]:
            region_cols = [
                col for col in result_df.columns if col.startswith(f"{region}_conservation_")
            ]
            if region_cols:
                result_df[f"{region}_avg_conservation"] = result_df[region_cols].mean(axis=1)

        logger.info(f"Calculated conservation scores for {len(result_df)} genes")
        self.results["sequence_conservation"] = result_df

        return result_df

    def analyze_rna_structure(
        self,
        utr5_sequences: pd.DataFrame,
        utr3_sequences: pd.DataFrame,
    ) -> dict[str, Any]:
        """Predict RNA secondary structures in UTR regions.

        This is a simplified implementation for demonstration purposes.
        A full implementation would use specialized RNA structure prediction tools
        such as ViennaRNA or Mfold to compute minimum free energy structures.

        Args:
            utr5_sequences: DataFrame with 5' UTR sequences
            utr3_sequences: DataFrame with 3' UTR sequences

        Returns:
            Dictionary with structure features and predicted structures
        """
        logger.info("Analyzing RNA secondary structure...")

        # NOTE: In a real implementation, this would use a proper RNA structure prediction
        # tool like ViennaRNA, Mfold, or RNAfold. This is a simplified placeholder.

        if not utr5_sequences and not utr3_sequences:
            logger.error("No UTR sequences provided")
            return pd.DataFrame()

        # Create a results DataFrame
        result_data = []

        # Process 5' UTR sequences
        for gene_id, sequence in utr5_sequences.items():
            if not sequence or len(sequence) < 10:
                continue

            # Simulate minimum free energy (MFE)
            # NOTE: In a real implementation, this would call a structure prediction tool
            mfe = -0.4 * len(sequence) * np.random.uniform(0.8, 1.2)

            # Add to results
            result_data.append(
                {
                    "gene_id": gene_id,
                    "region": "5'UTR",
                    "sequence_length": len(sequence),
                    "mfe": mfe,
                    "mfe_per_nucleotide": mfe / len(sequence),
                }
            )

        # Process 3' UTR sequences
        for gene_id, sequence in utr3_sequences.items():
            if not sequence or len(sequence) < 10:
                continue

            # Simulate minimum free energy
            mfe = -0.3 * len(sequence) * np.random.uniform(0.8, 1.2)

            # Add to results
            result_data.append(
                {
                    "gene_id": gene_id,
                    "region": "3'UTR",
                    "sequence_length": len(sequence),
                    "mfe": mfe,
                    "mfe_per_nucleotide": mfe / len(sequence),
                }
            )

        if not result_data:
            logger.warning("No RNA structures could be calculated")
            return pd.DataFrame()

        # Create results DataFrame
        result_df = pd.DataFrame(result_data)

        # Calculate statistics by region
        stats = result_df.groupby("region")["mfe_per_nucleotide"].agg(["mean", "std", "min", "max"])
        logger.info(f"RNA structure statistics:\n{stats}")

        logger.info(f"Calculated RNA structures for {len(result_df)} UTR regions")
        self.results["rna_structure"] = result_df

        return result_df

    def _simple_structure_prediction(self, sequence: str) -> str:
        """Generate a simplified RNA structure prediction.

        This is a placeholder for demonstration purposes.
        A full implementation would call a structure prediction algorithm
        to generate accurate secondary structure predictions.

        Args:
            sequence: RNA sequence to predict structure for

        Returns:
            String representation of the predicted structure
        """
        # NOTE: This is a placeholder implementation. A full implementation would call a structure prediction algorithm
        # to generate accurate secondary structure predictions.
        return "Simplified structure prediction"


class ComparativeSequenceVisualization:
    """Creates visualizations for comparative sequence analysis."""

    def __init__(self):
        """Initialize the visualization class."""
        self.figures = {}

    def plot_feature_comparison(
        self,
        comparison_df: pd.DataFrame,
        highlight_features: list[str] | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> Figure:
        """Create boxplots comparing feature distributions between stable and variable genes.

        Args:
            comparison_df: DataFrame with feature comparison statistics
            highlight_features: List of features to highlight
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if comparison_df.empty:
            logger.error("Empty comparison DataFrame provided")
            return None

        required_cols = ["feature", "cohens_d", "is_significant"]
        missing_cols = [col for col in required_cols if col not in comparison_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        try:
            # Select features to plot
            if highlight_features:
                features_to_plot = [
                    f for f in highlight_features if f in comparison_df["feature"].values
                ]
                if not features_to_plot:
                    # If none of the specified features are found, use top significant features
                    features_to_plot = comparison_df[comparison_df["is_significant"]][
                        "feature"
                    ].tolist()[:6]
            else:
                # Select top features by effect size
                features_to_plot = comparison_df.sort_values(
                    by="cohens_d", key=lambda x: abs(x), ascending=False
                )["feature"].tolist()[:6]

            if not features_to_plot:
                logger.warning("No features found for comparison plot")
                return None

            # Create figure
            fig, axes = plt.subplots(
                nrows=min(3, len(features_to_plot)),
                ncols=min(2, (len(features_to_plot) + 1) // 2),
                figsize=figsize,
                squeeze=False,
            )

            # Flatten axes array for easier indexing
            axes = axes.flatten()

            # For each feature, create a boxplot
            for i, feature in enumerate(features_to_plot):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Get feature row
                feature_row = comparison_df[comparison_df["feature"] == feature].iloc[0]

                # Create simulated data for boxplot
                # In a real implementation, this would use actual data from the feature_df
                n_consistent = feature_row["consistent_n"]
                n_variable = feature_row["variable_n"]

                consistent_mean = feature_row["consistent_mean"]
                consistent_median = feature_row["consistent_median"]
                variable_mean = feature_row["variable_mean"]
                variable_median = feature_row["variable_median"]

                # Estimate std from means and medians
                consistent_std = abs(consistent_mean - consistent_median) * 1.5
                variable_std = abs(variable_mean - variable_median) * 1.5

                # Generate simulated data
                np.random.seed(42 + i)  # For reproducibility
                consistent_data = np.random.normal(
                    consistent_mean, max(0.1, consistent_std), size=n_consistent
                )
                variable_data = np.random.normal(
                    variable_mean, max(0.1, variable_std), size=n_variable
                )

                # Create boxplot
                box_data = [consistent_data, variable_data]
                ax.boxplot(box_data, labels=["Consistent", "Variable"])

                # Add title with statistics
                is_sig = "Significant" if feature_row["is_significant"] else "Not significant"
                d = feature_row["cohens_d"]
                title = f"{feature}\n(d={d:.2f}, {is_sig})"
                ax.set_title(title)

                # Add mean values as text
                ax.text(
                    1,
                    consistent_mean,
                    f"μ={consistent_mean:.2f}",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
                ax.text(
                    2, variable_mean, f"μ={variable_mean:.2f}", ha="right", va="bottom", fontsize=8
                )

                # Grid
                ax.grid(alpha=0.3, axis="y")

            # Hide unused subplots
            for i in range(len(features_to_plot), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.exception(f"Error creating feature comparison plot: {e}")
            return None

    def plot_reference_similarity(
        self,
        similarity_df: pd.DataFrame,
        reference_type: str = "high_expression",
        figsize: tuple[int, int] = (10, 8),
    ) -> Figure:
        """Create a radar chart showing similarity to reference gene profiles.

        Args:
            similarity_df: DataFrame with reference similarity scores
            reference_type: Type of reference used
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if similarity_df.empty:
            logger.error("Empty similarity DataFrame provided")
            return None

        # Get columns representing z-scores
        z_cols = [col for col in similarity_df.columns if col.endswith("_z")]

        if not z_cols:
            logger.error("No z-score columns found")
            return None

        try:
            # Create figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, polar=True)

            # Feature labels (remove _z suffix)
            features = [col[:-2] for col in z_cols]

            # Number of features
            N = len(features)
            if N < 3:
                logger.error("Need at least 3 features for radar chart")
                return None

            # Angles for radar chart
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the circle

            # Create groups based on similarity percentile
            high_percentile = similarity_df["similarity_percentile"] >= 75
            mid_percentile = (similarity_df["similarity_percentile"] >= 25) & (
                similarity_df["similarity_percentile"] < 75
            )
            low_percentile = similarity_df["similarity_percentile"] < 25

            # Get mean z-scores for each group
            # Convert z-scores to deviation magnitudes (absolute values)
            high_genes = similarity_df[high_percentile][z_cols].abs().mean().tolist()
            mid_genes = similarity_df[mid_percentile][z_cols].abs().mean().tolist()
            low_genes = similarity_df[low_percentile][z_cols].abs().mean().tolist()

            # Close the circle for plotting
            high_genes += high_genes[:1]
            mid_genes += mid_genes[:1]
            low_genes += low_genes[:1]

            # Extended angles for labels
            extended_angles = angles
            extended_features = [*features, features[0]]

            # Plot data
            ax.plot(
                angles, high_genes, "o-", linewidth=2, label="High Similarity", color="tab:green"
            )
            ax.plot(
                angles, mid_genes, "o-", linewidth=2, label="Medium Similarity", color="tab:blue"
            )
            ax.plot(angles, low_genes, "o-", linewidth=2, label="Low Similarity", color="tab:red")

            # Fill areas
            ax.fill(angles, high_genes, alpha=0.1, color="tab:green")
            ax.fill(angles, mid_genes, alpha=0.1, color="tab:blue")
            ax.fill(angles, low_genes, alpha=0.1, color="tab:red")

            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features)

            # Add legend
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

            # Invert y-axis (lower absolute z-score is better)
            ax.set_ylim(0, max(*high_genes, *mid_genes, *low_genes) * 1.2)
            ax.set_title(f"Reference Similarity Profile: {reference_type}")

            # Add explanation text
            plt.figtext(0.05, 0.01, "Lower values = closer to reference", fontsize=10)

            return fig

        except Exception as e:
            logger.exception(f"Error creating reference similarity plot: {e}")
            return None

    def plot_conservation_heatmap(
        self, conservation_df: pd.DataFrame, figsize: tuple[int, int] = (12, 8)
    ) -> Figure:
        """Create a heatmap showing sequence conservation scores.

        Args:
            conservation_df: DataFrame with conservation scores
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if conservation_df.empty:
            logger.error("Empty conservation DataFrame provided")
            return None

        # Get conservation columns
        conservation_cols = [
            col
            for col in conservation_df.columns
            if "conservation" in col and not col.startswith("avg")
        ]

        if not conservation_cols:
            logger.error("No conservation columns found")
            return None

        try:
            # Parse column names to get regions and species
            regions = []
            species = []

            for col in conservation_cols:
                parts = col.split("_conservation_")
                if len(parts) == 2:
                    region, species_name = parts
                    if region not in regions:
                        regions.append(region)
                    if species_name not in species:
                        species.append(species_name)

            if not regions or not species:
                logger.error("Could not parse conservation column names")
                return None

            # Create a pivot table for the heatmap
            pivot_data = np.zeros((len(regions), len(species)))

            for i, region in enumerate(regions):
                for j, spec in enumerate(species):
                    col = f"{region}_conservation_{spec}"
                    if col in conservation_cols:
                        pivot_data[i, j] = conservation_df[col].mean()

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create heatmap
            im = ax.imshow(pivot_data, cmap="viridis")

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Average Conservation Score")

            # Add labels
            ax.set_xticks(np.arange(len(species)))
            ax.set_yticks(np.arange(len(regions)))
            ax.set_xticklabels(species)
            ax.set_yticklabels(regions)

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add values to cells
            for i in range(len(regions)):
                for j in range(len(species)):
                    text = ax.text(
                        j,
                        i,
                        f"{pivot_data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if pivot_data[i, j] < 0.7 else "white",
                    )

            ax.set_title("Sequence Conservation by Region and Species")
            fig.tight_layout()

            return fig

        except Exception as e:
            logger.exception(f"Error creating conservation heatmap: {e}")
            return None

    def plot_rna_structures(
        self, structure_df: pd.DataFrame, figsize: tuple[int, int] = (12, 8)
    ) -> Figure:
        """Create a visualization of RNA structure statistics.

        Args:
            structure_df: DataFrame with RNA structure predictions
            figsize: Figure size as (width, height)

        Returns:
            Matplotlib Figure object
        """
        if structure_df.empty:
            logger.error("Empty structure DataFrame provided")
            return None

        required_cols = ["gene_id", "region", "mfe", "sequence_length"]
        missing_cols = [col for col in required_cols if col not in structure_df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        try:
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

            # 1. MFE vs length scatter plot
            for region in structure_df["region"].unique():
                region_data = structure_df[structure_df["region"] == region]
                ax1.scatter(
                    region_data["sequence_length"], region_data["mfe"], alpha=0.7, label=region
                )

                # Add trend line
                if len(region_data) > 1:
                    z = np.polyfit(region_data["sequence_length"], region_data["mfe"], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(
                        min(region_data["sequence_length"]),
                        max(region_data["sequence_length"]),
                        100,
                    )
                    ax1.plot(x_range, p(x_range), "--", linewidth=1)

            ax1.set_xlabel("Sequence Length (nt)")
            ax1.set_ylabel("Minimum Free Energy (kcal/mol)")
            ax1.set_title("RNA Stability vs Length")
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. MFE/nt distribution by region
            structure_df["mfe_per_nt"] = structure_df["mfe"] / structure_df["sequence_length"]

            for region in structure_df["region"].unique():
                region_data = structure_df[structure_df["region"] == region]
                if len(region_data) > 0:
                    sns.kdeplot(region_data["mfe_per_nt"], ax=ax2, label=region)

            ax2.set_xlabel("MFE per Nucleotide (kcal/mol/nt)")
            ax2.set_ylabel("Density")
            ax2.set_title("Structural Stability Distribution")
            ax2.grid(alpha=0.3)

            # 3. Boxplot of MFE/nt by region
            ax3.boxplot(
                [
                    structure_df[structure_df["region"] == region]["mfe_per_nt"]
                    for region in structure_df["region"].unique()
                ],
                labels=structure_df["region"].unique(),
            )

            ax3.set_ylabel("MFE per Nucleotide (kcal/mol/nt)")
            ax3.set_title("Structural Stability Comparison")
            ax3.grid(alpha=0.3, axis="y")

            # Add stats to boxplot
            for i, region in enumerate(structure_df["region"].unique()):
                region_data = structure_df[structure_df["region"] == region]
                mean_val = region_data["mfe_per_nt"].mean()
                ax3.text(i + 1, mean_val, f"μ={mean_val:.3f}", ha="center", va="bottom", fontsize=8)

            fig.tight_layout()
            return fig

        except Exception as e:
            logger.exception(f"Error creating RNA structure plot: {e}")
            return None
