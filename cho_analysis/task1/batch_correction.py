# cho_analysis/task1/batch_correction.py
"""Batch correction for CHO cell line gene expression data.

This module provides methods to detect and correct batch effects in gene expression data
across different sequencing platforms (HiSeq, NovaSeq, NextSeq).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cho_analysis.core.logging import setup_logging
from cho_analysis.core.visualization_utils import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_FIGURE_TITLE,
)

# Initialize logger for this module
logger = setup_logging(__name__)


class BatchCorrection:
    """Class for batch effect detection and correction in CHO cell line RNA-seq data."""

    def __init__(self):
        """Initialize the batch correction module."""
        self.supported_methods = ["combat", "combat_preserve", "ruv"]
        self.figures = {}

    def detect_batch_effect(
        self, expression_df: pd.DataFrame, manifest_df: pd.DataFrame
    ) -> tuple[list[str], float, float]:
        """Detect batch effects in expression data based on sequencing platform.

        Args:
            expression_df: DataFrame with expression data
            manifest_df: DataFrame with sample metadata

        Returns:
            tuple: (batch_labels, platform_effect_percent, silhouette_score)
        """
        # Extract expression data (no metadata)
        if "ensembl_transcript_id" in expression_df.columns:
            expr_matrix = expression_df.iloc[:, 3:].copy()  # Remove metadata columns
        else:
            expr_matrix = expression_df.copy()

        # Extract platform information
        manifest_df = manifest_df.copy()
        if "platform" not in manifest_df.columns and "description" in manifest_df.columns:
            manifest_df["platform"] = (
                manifest_df["description"].str.extract(r"(HiSeq|NovaSeq|NextSeq)").iloc[:, 0]
            )

        # Create batch dictionary
        if "Sample" in manifest_df.columns:
            batch_dict = dict(zip(manifest_df["Sample"], manifest_df["platform"], strict=False))
        else:
            batch_dict = dict(zip(manifest_df.index, manifest_df["platform"], strict=False))

        # Create batch list in same order as expression matrix columns
        batch_labels = [batch_dict.get(col, "Unknown") for col in expr_matrix.columns]

        # Log platform information
        platform_counts = pd.Series(batch_labels).value_counts()
        logger.info(f"Detected platforms: {', '.join(platform_counts.index)}")
        logger.info(f"Platform counts: {dict(platform_counts)}")

        # Calculate percentage of genes affected by platform
        # Use ANOVA to test each gene for platform effect
        from scipy.stats import f_oneway

        platform_effect_percent = 0.0
        significant_genes = 0

        try:
            # Group samples by platform
            platforms = np.unique([b for b in batch_labels if b != "Unknown"])
            if len(platforms) > 1:  # At least 2 platforms needed for comparison
                # For each gene, test if expression differs by platform
                for gene_idx in range(expr_matrix.shape[0]):
                    gene_expr = expr_matrix.iloc[gene_idx, :].values

                    # Group expression values by platform
                    platform_groups = []
                    for platform in platforms:
                        platform_indices = [i for i, b in enumerate(batch_labels) if b == platform]
                        platform_groups.append(gene_expr[platform_indices])

                    # Run ANOVA
                    try:
                        f_stat, p_val = f_oneway(*platform_groups)
                        if p_val < 0.05:  # Significant platform effect
                            significant_genes += 1
                    except:
                        continue  # Skip genes where ANOVA fails

                # Calculate percentage of affected genes
                platform_effect_percent = (significant_genes / expr_matrix.shape[0]) * 100
                logger.info(
                    f"Found {significant_genes} genes ({platform_effect_percent:.1f}%) with significant platform effect"
                )

                # Always consider batch effect significant if more than 20% of genes are affected
                if platform_effect_percent > 20:
                    logger.info(
                        f"Significant batch effect detected: {platform_effect_percent:.1f}% of genes affected"
                    )
                else:
                    logger.info(
                        f"Batch effect present but moderate: {platform_effect_percent:.1f}% of genes affected"
                    )
        except Exception as e:
            logger.warning(f"Error calculating platform effect: {e}")

        # Transpose for PCA - samples as rows, genes as columns
        data_for_pca = expr_matrix.T

        # Apply PCA to detect batch effect
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_for_pca)

        # Calculate silhouette score
        sil_score = 0.0
        try:
            # Filter out unknown platforms
            valid_indices = [i for i, b in enumerate(batch_labels) if b != "Unknown"]
            if len(valid_indices) > 0 and len({b for b in batch_labels if b != "Unknown"}) > 1:
                valid_pca = pca_result[valid_indices]
                valid_labels = [batch_labels[i] for i in valid_indices]
                sil_score = silhouette_score(valid_pca, valid_labels)
                logger.info(
                    f"Batch effect silhouette score: {sil_score:.4f} (higher values indicate stronger batch effect)"
                )
            else:
                logger.info("Not enough valid platforms to calculate silhouette score")
        except Exception as e:
            logger.warning(f"Error calculating silhouette score: {e}")

        return batch_labels, platform_effect_percent, sil_score

    def compare_pca_stages(
        self,
        raw_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        corrected_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        figsize: tuple[int, int] = (22, 7),  # Slightly wider default
    ) -> Figure:
        """Compare PCA plots at three stages of data processing."""
        manifest_df = manifest_df.copy()
        if "platform" not in manifest_df.columns and "description" in manifest_df.columns:
            manifest_df["platform"] = (
                manifest_df["description"].str.extract(r"(HiSeq|NovaSeq|NextSeq)").iloc[:, 0]
            )
        if "Sample" in manifest_df.columns:
            batch_dict = dict(zip(manifest_df["Sample"], manifest_df["platform"], strict=False))
        else:
            batch_dict = dict(zip(manifest_df.index, manifest_df["platform"], strict=False))

        fig, axes = plt.subplots(1, 3, figsize=figsize)  # Use parameter
        plt.subplots_adjust(wspace=0.35, bottom=0.2)  # Adjust spacing and bottom margin

        datasets = [
            ("Unfiltered Data", raw_df),
            ("Filtered Data", filtered_df),
            ("Batch-Corrected Data", corrected_df),
        ]

        for i, (title, df) in enumerate(datasets):
            if "ensembl_transcript_id" in df.columns:
                expr_matrix = df.iloc[:, 3:].copy()
            else:
                expr_matrix = df.copy()
            common_samples = [col for col in expr_matrix.columns if col in batch_dict]
            expr_matrix = expr_matrix[common_samples]
            batch_labels = [batch_dict.get(col, "Unknown") for col in expr_matrix.columns]
            data_for_pca = expr_matrix.T
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data_for_pca)

            ax = axes[i]
            platforms = np.unique([b for b in batch_labels if b != "Unknown"])
            platform_colors = sns.color_palette(
                "tab10", n_colors=len(platforms)
            )  # Use consistent palette
            color_map = dict(zip(platforms, platform_colors, strict=False))

            for platform in platforms:
                platform_indices = [j for j, b in enumerate(batch_labels) if b == platform]
                ax.scatter(
                    pca_result[platform_indices, 0],
                    pca_result[platform_indices, 1],
                    label=platform,
                    alpha=0.7,
                    s=80,  # Slightly larger points
                    color=color_map.get(platform),  # Use mapped color
                )

            explained_var_ratio = pca.explained_variance_ratio_
            ax.set_title(
                f"{title}\nPC1: {explained_var_ratio[0]*100:.1f}%, PC2: {explained_var_ratio[1]*100:.1f}%"
            )
            ax.set_xlabel(f"First Principal Component ({explained_var_ratio[0]*100:.1%} variance)")
            ax.set_ylabel(f"Second Principal Component ({explained_var_ratio[1]*100:.1%} variance)")
            ax.legend(title="Platform")
            ax.grid(True, linestyle="--", alpha=0.6)  # Add grid

            # Add annotation about explained variance by PC1 (proxy for main variance source)
            ax.text(
                0.5,
                -0.18,  # Lowered y-position
                f"PC1 variance: {explained_var_ratio[0]*100:.1f}%",
                ha="center",
                transform=ax.transAxes,
                fontsize=FONT_SIZE_ANNOTATION,  # Use constant
            )

            # Calculate and add silhouette score text
            try:
                valid_indices = [j for j, b in enumerate(batch_labels) if b != "Unknown"]
                if (
                    len(valid_indices) > 0
                    and len({b for b in batch_labels if b != "Unknown"}) > 1
                ):
                    valid_pca = pca_result[valid_indices]
                    valid_labels = [batch_labels[j] for j in valid_indices]
                    sil_score = silhouette_score(valid_pca, valid_labels)
                    ax.text(
                        0.5,
                        -0.25,
                        f"Silhouette score: {sil_score:.3f}",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=FONT_SIZE_ANNOTATION,
                    )
            except Exception as e:
                logger.warning(f"Error calculating silhouette score: {e}")

        plt.suptitle(
            "PCA Analysis Across Data Processing Stages", fontsize=FONT_SIZE_FIGURE_TITLE, y=1.0
        )

        self.figures["pca_comparison"] = fig
        return fig

    def correct_batch(
        self,
        expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        target_col: str = "PRODUCT-TG",
        method: str = "combat",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply batch correction to expression data, excluding the target gene.

        Args:
            expression_df: DataFrame with expression data
            manifest_df: DataFrame with sample metadata
            target_col: Name of the target gene column to exclude from correction
            method: Correction method (only 'combat' supported)

        Returns:
            tuple: (corrected_df, batch_labels)
        """
        logger.info(f"Starting batch correction using method: {method}")
        logger.info(f"Target gene to preserve: {target_col}")

        # Make a deep copy to avoid modifying the input
        expression_df_copy = expression_df.copy()

        # Check if the target gene exists in the data
        target_gene_present = False
        target_gene_values = None
        target_gene_idx = None
        target_metadata = None

        # Extract metadata columns (assuming first 3 columns are metadata)
        if "ensembl_transcript_id" in expression_df_copy.columns:
            # Extract metadata
            metadata_cols = ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
            metadata = expression_df_copy[metadata_cols].copy()
            expression_matrix = expression_df_copy.iloc[:, 3:].copy()  # Rest are expression values

            # Check for target gene in metadata
            target_gene_row = expression_df_copy[expression_df_copy["sym"] == target_col]
            if len(target_gene_row) > 0:
                target_gene_present = True
                target_gene_idx = target_gene_row.index[0]

                # Extract target gene metadata
                target_metadata = metadata.loc[target_gene_idx].copy()

                # Extract target gene values
                target_gene_values = expression_matrix.loc[target_gene_idx].copy()

                # Remove target gene from expression matrix for batch correction
                expression_matrix = expression_matrix.drop(target_gene_idx)
                metadata = metadata.drop(target_gene_idx)

                logger.info(f"Target gene '{target_col}' found and excluded from batch correction")
        else:
            # Simple case - no metadata columns
            metadata = None
            expression_matrix = expression_df_copy.copy()

            # Check for target gene
            if target_col in expression_matrix.index:
                target_gene_present = True
                target_gene_values = expression_matrix.loc[target_col].copy()
                expression_matrix = expression_matrix.drop(target_col)
                logger.info(f"Target gene '{target_col}' found and excluded from batch correction")
            else:
                logger.warning(f"Target gene '{target_col}' not found in expression data")

        # Extract platform information from manifest
        manifest_df = manifest_df.copy()
        if "platform" not in manifest_df.columns and "description" in manifest_df.columns:
            manifest_df["platform"] = (
                manifest_df["description"].str.extract(r"(HiSeq|NovaSeq|NextSeq)").iloc[:, 0]
            )

        # Create batch dictionary
        if "Sample" in manifest_df.columns:
            batch_dict = dict(zip(manifest_df["Sample"], manifest_df["platform"], strict=False))
        else:
            batch_dict = dict(zip(manifest_df.index, manifest_df["platform"], strict=False))

        # Create batch list in same order as expression matrix columns
        batch_labels = [batch_dict.get(col, "Unknown") for col in expression_matrix.columns]

        # Log batch information
        platform_counts = pd.Series(batch_labels).value_counts()
        logger.info(f"Detected platforms: {', '.join(platform_counts.index)}")
        logger.info(f"Platform counts: {dict(platform_counts)}")

        # Get input values
        input_values = expression_matrix.values.copy()

        # Ensure all values are positive (ComBat-seq requirement)
        min_nonzero = np.min(input_values[input_values > 0])
        input_values[input_values == 0] = min_nonzero * 0.1

        # Apply batch correction with ComBat-seq
        try:
            from inmoose.pycombat import pycombat_seq

            logger.info("Applying ComBat-seq for batch correction...")
            corrected_matrix = pycombat_seq(input_values, batch_labels)

        except ImportError:
            try:
                # Try alternative import
                from pycombat import pycombat_seq

                logger.info("Using alternative pycombat_seq import...")
                corrected_matrix = pycombat_seq(input_values, batch_labels)
            except ImportError:
                msg = "pycombat package is required for ComBat-seq correction. Install with 'pip install pycombat' or 'pip install inmoose'."
                logger.exception(msg)
                raise ImportError(msg)

        # Convert back to dataframe with proper indices
        corrected_df = pd.DataFrame(
            corrected_matrix, index=expression_matrix.index, columns=expression_matrix.columns
        )

        # Prepare the final dataframe
        if metadata is not None:
            # Add metadata back to corrected expression data
            final_df = pd.concat([metadata, corrected_df], axis=1)

            # If target gene was found, add it back unchanged
            if (
                target_gene_present
                and target_gene_values is not None
                and target_metadata is not None
            ):
                # Create a dataframe for the target gene
                target_df = pd.DataFrame(
                    {
                        "ensembl_transcript_id": [target_metadata["ensembl_transcript_id"]],
                        "sym": [target_metadata["sym"]],
                        "ensembl_peptide_id": [target_metadata["ensembl_peptide_id"]],
                    }
                )

                # Add expression values
                for col in target_gene_values.index:
                    target_df[col] = target_gene_values[col]

                # Set the index to match the original
                target_df.index = [target_gene_idx]

                # Append target gene to the final dataframe
                final_df = pd.concat([final_df, target_df])

                # Sort index to maintain original order
                if isinstance(expression_df_copy.index, pd.RangeIndex):
                    final_df = final_df.sort_index()
        else:
            # No metadata case - just use expression matrix
            final_df = corrected_df

            # Add target gene back if it was found
            if target_gene_present and target_gene_values is not None:
                final_df.loc[target_col] = target_gene_values

        logger.info("Batch correction completed successfully")

        # Verify target gene was preserved
        if target_gene_present:
            try:
                # Ensure 'sym' column exists for lookup
                if "sym" in expression_df_copy.columns and "sym" in final_df.columns:
                    orig_row = expression_df_copy[expression_df_copy["sym"] == target_col]
                    corr_row = final_df[final_df["sym"] == target_col]

                    if not orig_row.empty and not corr_row.empty:
                        # Define metadata columns to exclude from comparison
                        meta_cols = ["ensembl_transcript_id", "sym", "ensembl_peptide_id"]
                        # Get numeric columns present in BOTH dataframes
                        numeric_cols_orig = orig_row.select_dtypes(include=np.number).columns
                        numeric_cols_corr = corr_row.select_dtypes(include=np.number).columns
                        common_numeric_cols = list(set(numeric_cols_orig) & set(numeric_cols_corr))

                        if common_numeric_cols:
                            # Extract only numeric values for comparison
                            orig_vals = orig_row.iloc[0][common_numeric_cols].values.astype(float)
                            corr_vals = corr_row.iloc[0][common_numeric_cols].values.astype(float)

                            # Check for NaNs after conversion (should be rare if data is clean)
                            if np.isnan(orig_vals).any() or np.isnan(corr_vals).any():
                                logger.warning(
                                    f"NaN values found in target gene '{target_col}' numeric data during preservation check."
                                )
                            elif np.allclose(orig_vals, corr_vals, rtol=1e-5, atol=1e-8):
                                logger.info(
                                    f"Target gene '{target_col}' values appear numerically identical based on np.allclose."
                                )
                            else:
                                logger.info(
                                    f"Target gene '{target_col}' values differ slightly based on np.allclose (check Pearson correlation in main script)."
                                )
                        else:
                            logger.warning(
                                "No common numeric columns found for target gene preservation check."
                            )
                    else:
                        logger.warning(
                            f"Target gene '{target_col}' not found in original or corrected data for check."
                        )
                else:
                    logger.warning("Cannot verify target gene preservation ('sym' column missing).")
            except Exception as e:
                # Log the warning observed in the output, using the actual error message
                logger.warning(
                    f"Could not verify target gene preservation (np.allclose check failed): {e}"
                )
                # The Pearson correlation check will still run later in run_analysis.py

        return final_df, batch_labels

    def correct_batch_with_design(
        self,
        expression_df: pd.DataFrame,
        manifest_df: pd.DataFrame,
        product_tg_values: pd.Series = None,
        method: str = "combat_preserve",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply batch correction to expression data with a design matrix to preserve biological signal.

        Args:
            expression_df: DataFrame with expression data
            manifest_df: DataFrame with sample metadata
            product_tg_values: Series with PRODUCT-TG values to preserve (optional)
            method: Correction method ('combat_preserve' or 'ruv')

        Returns:
            tuple: (corrected_df, batch_labels)
        """
        logger.info(f"Starting batch correction with design matrix using method: {method}")

        # Extract expression data (samples in columns, genes in rows)
        if "ensembl_transcript_id" in expression_df.columns:
            expression_matrix = expression_df.iloc[:, 3:].copy()  # Remove metadata columns
            metadata = expression_df.iloc[:, :3].copy()  # First 3 columns are metadata
        else:
            expression_matrix = expression_df.copy()
            metadata = None

        # Extract platform information from manifest
        manifest_df = manifest_df.copy()
        if "platform" not in manifest_df.columns and "description" in manifest_df.columns:
            manifest_df["platform"] = (
                manifest_df["description"].str.extract(r"(HiSeq|NovaSeq|NextSeq)").iloc[:, 0]
            )

        # Create batch dictionary
        if "Sample" in manifest_df.columns:
            batch_dict = dict(zip(manifest_df["Sample"], manifest_df["platform"], strict=False))
        else:
            batch_dict = dict(zip(manifest_df.index, manifest_df["platform"], strict=False))

        # Create batch list in same order as expression matrix columns
        batch_labels = [batch_dict.get(col, "Unknown") for col in expression_matrix.columns]

        # Log batch information
        platform_counts = pd.Series(batch_labels).value_counts()
        logger.info(f"Detected platforms: {', '.join(platform_counts.index)}")
        logger.info(f"Platform counts: {dict(platform_counts)}")

        # Process input values
        input_values = expression_matrix.values.copy()

        # Ensure all values are positive (ComBat-seq requirement)
        min_nonzero = np.min(input_values[input_values > 0])
        input_values[input_values == 0] = min_nonzero * 0.1

        # Prepare design matrix for biological signal preservation
        if product_tg_values is not None:
            # Ensure product_tg_values align with expression matrix columns
            if len(product_tg_values) != expression_matrix.shape[1]:
                logger.warning(
                    "Warning: PRODUCT-TG values length doesn't match sample count. Not using design matrix."
                )
                design_matrix = None
            else:
                # Create a design matrix with PRODUCT-TG values
                design_matrix = pd.DataFrame({"PRODUCT-TG": product_tg_values})
                logger.info(
                    "Using PRODUCT-TG values in design matrix to preserve biological signal"
                )
        else:
            design_matrix = None

        # Apply batch correction with chosen method
        if method == "combat_preserve":
            try:
                # Use modified ComBat that takes a design matrix
                logger.info("Applying ComBat with design matrix for softer batch correction...")

                # Convert categorical batch labels to numeric
                batch_categories = pd.Categorical(batch_labels)
                numeric_batch = batch_categories.codes

                # Try using pycombat directly with mod parameter
                try:
                    from inmoose.pycombat import pycombat

                    if design_matrix is not None:
                        try:
                            # Use mod parameter for design matrix
                            logger.info("Using design matrix with pycombat")
                            corrected_matrix = pycombat(
                                input_values, batch_labels, mod=design_matrix.values
                            )
                        except Exception as e:
                            logger.warning(f"Failed to use design matrix: {e!s}")
                            logger.info("Falling back to standard pycombat")
                            corrected_matrix = pycombat(input_values, batch_labels)
                    else:
                        corrected_matrix = pycombat(input_values, batch_labels)
                except ImportError:
                    try:
                        # Try alternative import
                        from pycombat import pycombat

                        logger.info("Using alternative pycombat import...")

                        if design_matrix is not None:
                            corrected_matrix = pycombat(
                                input_values, batch_labels, mod=design_matrix.values
                            )
                        else:
                            corrected_matrix = pycombat(input_values, batch_labels)
                    except ImportError:
                        # If pycombat not available, fall back to ComBat-seq
                        logger.warning("pycombat not available, falling back to ComBat-seq")
                        try:
                            from inmoose.pycombat import pycombat_seq

                            corrected_matrix = pycombat_seq(input_values, batch_labels)
                        except ImportError:
                            from pycombat import pycombat_seq

                            corrected_matrix = pycombat_seq(input_values, batch_labels)

            except Exception as e:
                logger.exception(f"Error in ComBat correction: {e!s}")
                logger.warning("Falling back to standard ComBat-seq")
                try:
                    from inmoose.pycombat import pycombat_seq

                    corrected_matrix = pycombat_seq(input_values, batch_labels)
                except ImportError:
                    from pycombat import pycombat_seq

                    corrected_matrix = pycombat_seq(input_values, batch_labels)

        elif method == "ruv":
            try:
                logger.info("Applying RUV (Remove Unwanted Variation) for batch correction...")

                # Try to import RUV
                from sklearn.decomposition import PCA

                # RUV requires control genes or replicate samples
                # Let's use a PCA-based approach to estimate unwanted variation

                # Step 1: Run PCA on data
                pca = PCA(n_components=min(10, input_values.shape[1] - 1))
                pca_result = pca.fit_transform(input_values.T)  # Transpose so samples are rows

                # Step 2: Find PCs associated with batch effect
                # Calculate correlation between PC scores and batch labels
                batch_pc_corr = []
                for i in range(pca_result.shape[1]):
                    pc_scores = pca_result[:, i]

                    # Calculate ANOVA F-value between PC scores and batch groups
                    from scipy.stats import f_oneway

                    # Group PC scores by batch
                    batch_groups = []
                    for batch in np.unique(batch_labels):
                        batch_indices = [j for j, b in enumerate(batch_labels) if b == batch]
                        if batch_indices:  # Only add if we have samples for this batch
                            batch_groups.append(pc_scores[batch_indices])

                    # Run ANOVA
                    if len(batch_groups) > 1:  # Need at least 2 groups
                        f_stat, p_val = f_oneway(*batch_groups)
                        batch_pc_corr.append((i, f_stat, p_val))

                # Find PCs with significant batch effect (p < 0.05)
                batch_pcs = [i for i, _, p in batch_pc_corr if p < 0.05]

                # Step 3: Project out batch effect PCs
                # Reconstruct data without batch-associated PCs
                components_to_keep = [i for i in range(pca_result.shape[1]) if i not in batch_pcs]

                if len(components_to_keep) == 0:
                    logger.warning("All PCs associated with batch effect. Using gentle adjustment.")
                    # Fall back to gentle adjustment - remove 50% of each PC's effect
                    factor = 0.5
                    reconstruction = pca.inverse_transform(pca_result)
                    corrected_matrix = input_values - 0.5 * (reconstruction.T - input_values)
                else:
                    # Reconstruct data without batch effect PCs
                    partial_reconstruction = np.zeros_like(pca_result)
                    for i in components_to_keep:
                        partial_reconstruction[:, i] = pca_result[:, i]

                    # Transform back to original space
                    corrected_values = pca.inverse_transform(partial_reconstruction)
                    corrected_matrix = corrected_values.T  # Transpose back

                    logger.info(f"Removed {len(batch_pcs)} PCs associated with batch effect")

            except Exception as e:
                logger.exception(f"Error in RUV correction: {e!s}")
                logger.warning("Falling back to standard ComBat-seq")
                try:
                    from inmoose.pycombat import pycombat_seq

                    corrected_matrix = pycombat_seq(input_values, batch_labels)
                except ImportError:
                    from pycombat import pycombat_seq

                    corrected_matrix = pycombat_seq(input_values, batch_labels)

        else:
            # Fall back to standard ComBat-seq
            try:
                from inmoose.pycombat import pycombat_seq

                logger.info("Applying standard ComBat-seq for batch correction...")
                corrected_matrix = pycombat_seq(input_values, batch_labels)
            except ImportError:
                from pycombat import pycombat_seq

                logger.info("Applying standard ComBat-seq for batch correction...")
                corrected_matrix = pycombat_seq(input_values, batch_labels)

        # Convert back to dataframe with proper indices
        corrected_df = pd.DataFrame(
            corrected_matrix, index=expression_matrix.index, columns=expression_matrix.columns
        )

        # Add metadata columns back if they exist
        if metadata is not None:
            final_corrected_df = pd.concat([metadata, corrected_df], axis=1)
        else:
            final_corrected_df = corrected_df

        logger.info("Batch correction with design matrix completed successfully")
        return final_corrected_df, batch_labels
