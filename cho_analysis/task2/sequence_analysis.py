# cho_analysis/task2/sequence_analysis.py
"""Core analysis functions for Task 2: Sequence Feature Characterization.

Calculates features like GC content, codon usage, CAI, and motif presence
for UTR and CDS regions of consistently expressed genes.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

try:
    from Bio.SeqUtils import gc_fraction

    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False

    def gc_fraction(seq: str) -> float:
        logging.getLogger(__name__).exception("gc_fraction called but Biopython not installed.")
        return np.nan


from cho_analysis.core.config import get_sequence_analysis_config, get_sequence_constants

logger = logging.getLogger(__name__)

# --- Constants and Compiled Regex (Module Level) ---
try:
    _SEQ_CONSTANTS = get_sequence_constants()
    STANDARD_GENETIC_CODE = _SEQ_CONSTANTS.get("genetic_code", {})
    if not STANDARD_GENETIC_CODE:
        logger.critical("Standard Genetic Code constant not found!")
except Exception as e:
    logger.critical(f"Failed to load sequence constants from config: {e}", exc_info=True)
    STANDARD_GENETIC_CODE = {}

try:
    _SEQ_CONFIG = get_sequence_analysis_config()
    REGEX_KOZAK = re.compile(_SEQ_CONFIG.get("regex_kozak", r"GCCACC|CCACC"))
    REGEX_TOP = re.compile(_SEQ_CONFIG.get("regex_top", r"^C[CT]{4,15}"))
    REGEX_G_QUAD = re.compile(
        _SEQ_CONFIG.get("regex_g_quad", r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}")
    )
    REGEX_POLYA = re.compile(_SEQ_CONFIG.get("regex_polya", r"AATAAA"))
    REGEX_ARE = re.compile(_SEQ_CONFIG.get("regex_are", r"ATTTA"))
    REGEX_MIRNA_SEED = re.compile(_SEQ_CONFIG.get("regex_mirna_seed", r"A[ACGT]{6}A"))
    logger.info("Successfully compiled sequence motif regex patterns.")
except re.error as e:
    logger.exception(f"Failed to compile regex pattern from config: {e}. Motif analysis may fail.")
    REGEX_KOZAK, REGEX_TOP, REGEX_G_QUAD, REGEX_POLYA, REGEX_ARE, REGEX_MIRNA_SEED = (
        re.compile(p)
        for p in [
            r"GCCACC|CCACC",
            r"^C[CT]{4,15}",
            r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}",
            r"AATAAA",
            r"ATTTA",
            r"A[ACGT]{6}A",
        ]
    )
except Exception as e:
    logger.critical(f"Failed to load sequence analysis config: {e}", exc_info=True)
    REGEX_KOZAK, REGEX_TOP, REGEX_G_QUAD, REGEX_POLYA, REGEX_ARE, REGEX_MIRNA_SEED = (
        re.compile(p)
        for p in [
            r"GCCACC|CCACC",
            r"^C[CT]{4,15}",
            r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}",
            r"AATAAA",
            r"ATTTA",
            r"A[ACGT]{6}A",
        ]
    )


class SequenceFeatureAnalysis:
    """Encapsulates methods for sequence feature characterization."""

    def __init__(self):
        """Initializes the analysis class."""
        self.config = get_sequence_analysis_config()
        if not BIO_AVAILABLE:
            logger.warning("Biopython not installed. GC content calculation will return NaN.")
        if not STANDARD_GENETIC_CODE:
            logger.error("Standard Genetic Code not loaded. Codon analysis methods will fail.")

    def _safe_gc_fraction(self, seq: str) -> float:
        """Calculate GC fraction safely."""
        if not seq or not BIO_AVAILABLE:
            return np.nan if BIO_AVAILABLE else np.nan  # Return NaN regardless if no Bio
        try:
            return gc_fraction(seq) * 100
        except Exception as e:
            logger.exception(f"Error calculating GC fraction: {e}")
            return np.nan

    def _preprocess_fasta_df(
        self, seq_data: list[tuple[str, str]] | None, seq_type: str
    ) -> pd.DataFrame | None:
        """Converts raw FASTA data list to a preprocessed DataFrame."""
        if seq_data is None:
            logger.warning(f"No sequence data list provided for {seq_type}.")
            return None
        if not seq_data:
            logger.warning(f"Empty sequence data list provided for {seq_type}.")
            return pd.DataFrame(
                columns=[
                    "ensembl_transcript_id",
                    f"{seq_type}_ID",
                    f"{seq_type}_sym",
                    f"{seq_type}_Seq",
                ]
            )

        logger.info(f"Preprocessing {len(seq_data)} raw {seq_type} sequence records...")
        processed_records = []
        skipped_count = 0
        for record_id, sequence in seq_data:
            if not record_id or not sequence:
                skipped_count += 1
                continue
            parts = record_id.split("|")
            seq_upper = sequence.upper()
            if len(parts) == 3:
                processed_records.append(
                    {
                        "ensembl_transcript_id": parts[0],
                        f"{seq_type}_ID": parts[1],
                        f"{seq_type}_sym": parts[2],
                        f"{seq_type}_Seq": seq_upper,
                    }
                )
            elif len(parts) == 1:
                processed_records.append(
                    {
                        "ensembl_transcript_id": parts[0],
                        f"{seq_type}_ID": None,
                        f"{seq_type}_sym": None,
                        f"{seq_type}_Seq": seq_upper,
                    }
                )
            else:
                logger.debug(f"Unexpected FASTA ID format for {seq_type}: '{record_id}'. Skipping.")
                skipped_count += 1
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} records during {seq_type} preprocessing.")
        if not processed_records:
            logger.error(f"No sequences could be successfully preprocessed for {seq_type}.")
            return None

        df = pd.DataFrame(processed_records)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["ensembl_transcript_id"], keep="first")
        if len(df) < initial_count:
            logger.warning(
                f"Removed {initial_count - len(df)} duplicate transcript IDs during {seq_type} preprocessing."
            )
        logger.info(f"Successfully preprocessed {len(df)} unique {seq_type} sequences.")
        return df

    def find_consistently_expressed_genes(
        self, expression_df: pd.DataFrame, min_expr_threshold: float = 0.0
    ) -> pd.DataFrame:
        """Identifies consistently expressed genes based on expression and CV."""
        logger.info("Identifying consistently expressed genes...")
        required_meta_cols = ["ensembl_transcript_id", "sym"]
        if not all(col in expression_df.columns for col in required_meta_cols):
            logger.error(
                f"Expression data missing required metadata columns: {required_meta_cols}."
            )
            return pd.DataFrame()

        expr_cols = expression_df.select_dtypes(include=np.number).columns.tolist()
        if not expr_cols:
            logger.error("No numeric expression columns found.")
            return pd.DataFrame()

        cv_quantile_threshold = self.config.get("constant_expression_cv_threshold")
        min_expr_threshold = max(0.0, min_expr_threshold)

        try:
            expressed_mask = (expression_df[expr_cols] > min_expr_threshold).all(axis=1)
            consistent_genes = expression_df.loc[expressed_mask].copy()
        except Exception as e:
            logger.exception(f"Error during expression threshold filtering: {e}")
            return pd.DataFrame()

        if consistent_genes.empty:
            logger.warning(
                f"No genes found consistently expressed above threshold {min_expr_threshold}."
            )
            return consistent_genes

        logger.info(
            f"Found {len(consistent_genes)} genes expressed > {min_expr_threshold} in all samples."
        )

        try:
            consistent_genes["mean"] = consistent_genes[expr_cols].mean(axis=1)
            consistent_genes["std"] = consistent_genes[expr_cols].std(axis=1)
            consistent_genes["cv"] = np.divide(
                consistent_genes["std"],
                consistent_genes["mean"],
                out=np.full_like(consistent_genes["mean"], np.nan),
                where=consistent_genes["mean"] != 0,
            )
        except Exception as e:
            logger.exception(f"Error calculating mean/std/cv: {e}")
            return consistent_genes[required_meta_cols]

        valid_cv_quantile = cv_quantile_threshold is not None and 0 < cv_quantile_threshold < 1
        if valid_cv_quantile:
            try:
                cv_dropna = consistent_genes["cv"].dropna()
                if not cv_dropna.empty:
                    cv_cutoff = cv_dropna.quantile(cv_quantile_threshold)
                    if not pd.isna(cv_cutoff):
                        consistent_genes = consistent_genes[
                            consistent_genes["cv"] <= cv_cutoff
                        ].copy()
                        logger.info(
                            f"Filtered to {len(consistent_genes)} genes with CV <= {cv_cutoff:.4f} (quantile {cv_quantile_threshold})."
                        )
                    else:
                        logger.warning("CV cutoff NaN. Skipping CV filtering.")
                else:
                    logger.warning("No valid CV values. Skipping CV filtering.")
            except Exception as e:
                logger.exception(f"Error during CV quantile filtering: {e}")
        elif cv_quantile_threshold is not None:
            logger.warning(
                f"Invalid cv_quantile_threshold: {cv_quantile_threshold}. Skipping CV filtering."
            )

        consistent_genes = consistent_genes.sort_values("cv", ascending=True, na_position="last")
        logger.info(
            f"Final set of {len(consistent_genes)} consistently expressed genes identified."
        )
        return consistent_genes

    def analyze_utr_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes 5' and 3' UTR sequences for various features."""
        logger.info("Analyzing UTR sequence features...")
        results = []
        required_cols = ["ensembl_transcript_id", "sym", "cv", "mean", "UTR5_Seq", "UTR3_Seq"]
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            logger.error(f"Merged DataFrame missing columns for UTR analysis: {missing_cols}")
            return pd.DataFrame()

        for row in merged_df[required_cols].itertuples(index=False, name="GeneData"):
            utr5_seq = str(row.UTR5_Seq).upper() if pd.notna(row.UTR5_Seq) else ""
            utr3_seq = str(row.UTR3_Seq).upper() if pd.notna(row.UTR3_Seq) else ""

            utr5_len, utr3_len = len(utr5_seq), len(utr3_seq)
            utr5_gc, utr3_gc = self._safe_gc_fraction(utr5_seq), self._safe_gc_fraction(utr3_seq)
            utr5_aug = utr5_seq.count("ATG")
            kozak = bool(REGEX_KOZAK.search(utr5_seq)) if utr5_seq else False
            top_motif = bool(REGEX_TOP.search(utr5_seq)) if utr5_seq else False
            g_quad = bool(REGEX_G_QUAD.search(utr5_seq)) if utr5_seq else False
            polya_signal = bool(REGEX_POLYA.search(utr3_seq)) if utr3_seq else False
            are_motifs = bool(REGEX_ARE.search(utr3_seq)) if utr3_seq else False
            mirna_matches = len(REGEX_MIRNA_SEED.findall(utr3_seq)) if utr3_seq else 0

            results.append(
                {
                    "ensembl_transcript_id": row.ensembl_transcript_id,
                    "symbol": row.sym,
                    "cv": row.cv,
                    "mean_expression": row.mean,
                    "UTR5_length": utr5_len,
                    "UTR5_GC": utr5_gc,
                    "UTR5_AUG_count": utr5_aug,
                    "kozak_sequence_present": kozak,
                    "TOP_motif_present": top_motif,
                    "G_quadruplex_present": g_quad,
                    "UTR3_length": utr3_len,
                    "UTR3_GC": utr3_gc,
                    "polyA_signal_present": polya_signal,
                    "ARE_motifs_present": are_motifs,
                    "miRNA_seed_matches": mirna_matches,
                }
            )
        logger.info(f"UTR analysis completed for {len(results)} transcripts.")
        return pd.DataFrame(results)

    def _calculate_codon_stats(self, sequence: str) -> tuple[dict[str, float], float, float]:
        """Calculates codon usage frequency, GC3 content, and CAI."""
        codon_usage: dict[str, float] = {}
        gc3_content: float = 0.0
        cai: float = 0.0
        if not isinstance(sequence, str) or len(sequence) % 3 != 0 or not STANDARD_GENETIC_CODE:
            return codon_usage, gc3_content, cai

        sequence_upper = sequence.upper()
        codons = [sequence_upper[i : i + 3] for i in range(0, len(sequence_upper), 3)]
        total_codons = len(codons)
        if total_codons == 0:
            return codon_usage, gc3_content, cai
        counts = Counter(codons)

        codon_usage = {codon: count / total_codons for codon, count in counts.items()}
        gc3_count = sum(1 for codon in codons if codon[2] in "GC")
        gc3_content = (gc3_count / total_codons) * 100 if total_codons > 0 else 0.0

        # --- CAI Calculation ---
        amino_to_codons = defaultdict(list)
        for codon, aa in STANDARD_GENETIC_CODE.items():
            amino_to_codons[aa].append(codon)
        most_common_codons: dict[str, str] = {}
        for aa, aa_codons in amino_to_codons.items():
            if aa == "*" or len(aa_codons) == 1:
                continue
            codon_freqs = {codon: counts.get(codon, 0) for codon in aa_codons}
            if sum(codon_freqs.values()) > 0:
                most_common_codons[aa] = max(codon_freqs.items(), key=lambda item: item[1])[0]

        cai_codon_values: list[float] = []
        for codon in codons:
            if codon in STANDARD_GENETIC_CODE:
                aa = STANDARD_GENETIC_CODE[codon]
                if aa != "*" and aa in most_common_codons:
                    optimal_codon_for_aa = most_common_codons[aa]
                    count_codon, count_optimal = (
                        counts.get(codon, 0),
                        counts.get(optimal_codon_for_aa, 0),
                    )
                    if count_optimal > 0:
                        cai_codon_values.append(count_codon / count_optimal)

        if cai_codon_values:
            positive_vals = [val for val in cai_codon_values if val > 0]
            if positive_vals:
                with np.errstate(divide="ignore"):
                    log_vals = np.log(positive_vals)
                cai = np.exp(np.mean(log_vals))
        return codon_usage, gc3_content, cai

    def analyze_cds_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Analyzes CDS sequences for codon usage, GC3 content, and CAI."""
        logger.info("Analyzing CDS sequence features...")
        required_cols = ["ensembl_transcript_id", "CDS_Seq"]
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            logger.error(f"Merged DataFrame missing columns for CDS analysis: {missing_cols}")
            return pd.DataFrame()
        if not STANDARD_GENETIC_CODE:
            logger.error("Cannot analyze CDS features: Standard Genetic Code not loaded.")
            return pd.DataFrame()

        def apply_codon_stats(row):
            usage, gc3, cai = self._calculate_codon_stats(row["CDS_Seq"])
            return pd.Series({"codon_usage": usage, "GC3_content": gc3, "CAI": cai})

        try:
            cds_data = merged_df[required_cols].set_index("ensembl_transcript_id")
            cds_features = cds_data.apply(apply_codon_stats, axis=1)
            cds_features.reset_index(inplace=True)
        except Exception as e:
            logger.exception(f"Error applying codon stats calculation: {e}")
            return pd.DataFrame()
        logger.info(f"CDS analysis completed for {len(cds_features)} transcripts.")
        return cds_features

    def extract_codon_usage_table(self, df_codon_results: pd.DataFrame) -> pd.DataFrame:
        """Extracts codon usage dictionaries into a wide DataFrame."""
        logger.info("Extracting codon usage frequencies into a table...")
        required_cols = ["ensembl_transcript_id", "codon_usage"]
        if not all(col in df_codon_results.columns for col in required_cols):
            logger.error(f"Input DataFrame missing required columns: {required_cols}")
            return pd.DataFrame()
        if not STANDARD_GENETIC_CODE:
            logger.error("Cannot extract codon usage table: Standard Genetic Code not loaded.")
            return pd.DataFrame()

        all_codons = list(STANDARD_GENETIC_CODE.keys())
        data_for_df, index_for_df = [], []
        for _, row in df_codon_results.iterrows():
            transcript_id, codon_dict = row["ensembl_transcript_id"], row["codon_usage"]
            if isinstance(codon_dict, dict):
                usage_row = {codon: codon_dict.get(codon, 0.0) for codon in all_codons}
                data_for_df.append(usage_row)
                index_for_df.append(transcript_id)
            else:
                logger.warning(
                    f"Invalid codon usage dict for transcript {transcript_id}. Skipping."
                )

        if not data_for_df:
            logger.error("No valid codon usage data found to create table.")
            return pd.DataFrame(columns=all_codons)

        codon_usage_wide_df = pd.DataFrame(data_for_df, index=index_for_df)
        codon_usage_wide_df.index.name = "ensembl_transcript_id"
        codon_usage_wide_df = codon_usage_wide_df[all_codons]  # Ensure standard order
        logger.info(f"Codon usage table created with shape {codon_usage_wide_df.shape}.")
        return codon_usage_wide_df

    def analyze_kmers(
        self, sequences: list[str], k: int = 6, top_n: int = 20
    ) -> list[tuple[str, int]]:
        """Counts k-mer occurrences and returns the most frequent."""
        valid_sequences = [str(s).upper() for s in sequences if pd.notna(s) and isinstance(s, str)]
        if not valid_sequences:
            logger.warning("No valid sequences provided for k-mer analysis.")
            return []

        logger.info(f"Analyzing {k}-mers for {len(valid_sequences)} sequences...")
        kmer_counts = defaultdict(int)
        for seq in valid_sequences:
            if len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    kmer_counts[seq[i : i + k]] += 1
        if not kmer_counts:
            logger.warning(f"No valid {k}-mers found.")
            return []

        sorted_kmers = sorted(kmer_counts.items(), key=lambda item: item[1], reverse=True)
        n_unique, n_return = len(kmer_counts), min(top_n, len(kmer_counts))
        logger.info(f"Found {n_unique} unique {k}-mers. Returning top {n_return}.")
        return sorted_kmers[:n_return]

    def categorize_stability(self, feature_df: pd.DataFrame, n_quantiles: int = 3) -> pd.Series:
        """Categorizes genes based on expression stability (CV) quantiles."""
        if "cv" not in feature_df.columns:
            logger.error("Cannot categorize stability: 'cv' column missing.")
            return pd.Series(dtype=str)
        if feature_df["cv"].isnull().all():
            logger.warning("Cannot categorize stability: 'cv' column contains only NaNs.")
            return pd.Series(np.nan, index=feature_df.index, dtype=str)
        if feature_df["cv"].nunique() < n_quantiles:
            logger.warning(
                f"Unique CV values ({feature_df['cv'].nunique()}) < quantiles ({n_quantiles}). Skipping categorization."
            )
            return pd.Series(np.nan, index=feature_df.index, dtype=str)

        labels = (
            ["High", "Medium", "Low"]  # Low CV -> High Stability
            if n_quantiles == 3
            else [f"Stability_Q{i+1}" for i in range(n_quantiles)]
        )

        try:
            stability_categories, bins = pd.qcut(
                feature_df["cv"], q=n_quantiles, labels=False, retbins=True, duplicates="drop"
            )
            num_actual_bins = len(bins) - 1
            if num_actual_bins == len(labels):
                stability_categories = pd.qcut(
                    feature_df["cv"], q=n_quantiles, labels=labels, duplicates="drop"
                )
                value_counts = stability_categories.value_counts().to_dict()
                logger.info(
                    f"Categorized stability into {n_quantiles} quantiles. Counts: {value_counts}"
                )
            else:
                logger.warning(
                    f"Could not apply labels due to duplicate bin edges ({num_actual_bins} bins created). Returning numeric quantile indices."
                )
                value_counts = stability_categories.value_counts().to_dict()
                logger.info(
                    f"Categorized stability into {num_actual_bins} numeric bins. Counts: {value_counts}"
                )

            return stability_categories
        except Exception as e:
            logger.exception(f"Failed to categorize stability using qcut: {e}")
            return pd.Series(np.nan, index=feature_df.index, dtype=str)

    def run_analysis(
        self,
        expression_df: pd.DataFrame,
        cds_seqs: list[tuple[str, str]] | None,
        utr5_seqs: list[tuple[str, str]] | None,
        utr3_seqs: list[tuple[str, str]] | None,
    ) -> dict[str, pd.DataFrame | None]:
        """Runs the full sequence feature analysis pipeline."""
        results: dict[str, pd.DataFrame | None] = {
            "consistent_genes": None,
            "merged_data": None,
            "combined_features": None,
            "codon_usage_table": None,
        }

        logger.info("Step 1: Finding consistently expressed genes...")
        consistent_genes_df = self.find_consistently_expressed_genes(expression_df)
        if consistent_genes_df.empty:
            logger.error("Aborting Task 2: No consistently expressed genes found.")
            return results
        results["consistent_genes"] = consistent_genes_df

        logger.info("Step 2: Preprocessing sequence data...")
        cds_df = self._preprocess_fasta_df(cds_seqs, "CDS")
        utr5_df = self._preprocess_fasta_df(utr5_seqs, "UTR5")
        utr3_df = self._preprocess_fasta_df(utr3_seqs, "UTR3")
        if cds_df is None or cds_df.empty:
            logger.error("Aborting Task 2: Preprocessed CDS sequence data is missing or empty.")
            return results

        logger.info("Step 3: Merging expression stats with sequences...")
        merged_df = consistent_genes_df[
            ["ensembl_transcript_id", "sym", "mean", "std", "cv"]
        ].copy()
        seq_dfs = {"CDS": cds_df, "UTR5": utr5_df, "UTR3": utr3_df}
        for name, seq_df in seq_dfs.items():
            col_name = f"{name}_Seq"
            if seq_df is not None and not seq_df.empty:
                merged_df = pd.merge(
                    merged_df,
                    seq_df[["ensembl_transcript_id", col_name]],
                    on="ensembl_transcript_id",
                    how="left",
                )
            else:
                merged_df[col_name] = pd.NA
                logger.warning(f"No preprocessed {name} sequence data to merge.")
        if merged_df.empty:
            logger.error("Aborting Task 2: Merged data frame is empty.")
            return results
        results["merged_data"] = merged_df

        logger.info("Step 4: Analyzing sequence features...")
        utr_features = self.analyze_utr_features(merged_df)
        cds_features = self.analyze_cds_features(merged_df)  # Includes codon usage dict

        # Extract codon usage table (if cds_features were generated)
        if not cds_features.empty:
            results["codon_usage_table"] = self.extract_codon_usage_table(cds_features)
        else:
            logger.warning(
                "CDS feature analysis yielded no results, skipping codon usage table extraction."
            )
            results["codon_usage_table"] = pd.DataFrame()  # Return empty frame

        # Combine UTR and CDS features (excluding the raw dict)
        combined_features = pd.DataFrame()  # Initialize
        # Prepare CDS features without the dict column
        cds_features_nodict = (
            cds_features.drop(columns=["codon_usage"])
            if "codon_usage" in cds_features.columns
            else cds_features
        )

        # --- Start Refined Combination Logic ---
        if not utr_features.empty:
            combined_features = utr_features.copy()
            if not cds_features_nodict.empty:
                combined_features = pd.merge(
                    combined_features,
                    cds_features_nodict,
                    on="ensembl_transcript_id",
                    how="left",
                )
            else:
                # UTR exists, CDS doesn't. Add empty columns for CDS features for schema consistency.
                if "GC3_content" not in combined_features.columns:
                    combined_features["GC3_content"] = pd.NA
                if "CAI" not in combined_features.columns:
                    combined_features["CAI"] = pd.NA
                logger.warning(
                    "CDS analysis produced no results. Combined features lack GC3 and CAI."
                )
        elif not cds_features_nodict.empty:
            # Only CDS features exist. Try to add 'symbol', 'cv', 'mean' back from consistent_genes
            logger.warning(
                "UTR analysis produced no results. Attempting to merge CDS features with base gene info."
            )
            base_info_cols = [
                "ensembl_transcript_id",
                "symbol",
                "cv",
                "mean",
            ]  # Use 'mean', not 'mean_expression' from utr_features
            base_info = consistent_genes_df[
                [col for col in base_info_cols if col in consistent_genes_df.columns]
            ]
            if not base_info.empty:
                combined_features = pd.merge(
                    base_info,
                    cds_features_nodict,
                    on="ensembl_transcript_id",
                    how="inner",  # Only keep rows where we have CDS features
                )
                # Add empty UTR columns for schema consistency
                utr_cols_to_add = [
                    "UTR5_length",
                    "UTR5_GC",
                    "UTR5_AUG_count",
                    "kozak_sequence_present",
                    "TOP_motif_present",
                    "G_quadruplex_present",
                    "UTR3_length",
                    "UTR3_GC",
                    "polyA_signal_present",
                    "ARE_motifs_present",
                    "miRNA_seed_matches",
                ]
                for col in utr_cols_to_add:
                    if col not in combined_features.columns:
                        combined_features[col] = pd.NA
                # Rename 'mean' to 'mean_expression' if UTR features uses that name, for consistency
                if (
                    "mean" in combined_features.columns
                    and "mean_expression" not in combined_features.columns
                ):
                    combined_features.rename(columns={"mean": "mean_expression"}, inplace=True)

            else:
                logger.error(
                    "Cannot construct combined features: UTR analysis failed AND base gene info is missing."
                )
        else:
            # Both are empty
            logger.warning(
                "Both UTR and CDS feature analyses yielded empty results. Combined features will be empty."
            )

        # Add stability category if combined_features is not empty
        if not combined_features.empty:
            combined_features["stability_category"] = self.categorize_stability(combined_features)
            results["combined_features"] = combined_features
        else:
            results["combined_features"] = pd.DataFrame()  # Ensure it's an empty frame if needed

        logger.info("Sequence feature analysis pipeline completed.")
        return results

    # --- New Methods for Statistical Feature Significance Analysis ---
    def analyze_feature_significance(
        self, feature_df: pd.DataFrame, alpha: float = 0.05, correction_method: str = "fdr_bh"
    ) -> pd.DataFrame:
        """Analyzes the statistical significance of sequence features for expression.

        Args:
            feature_df: DataFrame containing sequence features and expression metrics
            alpha: Significance level for statistical tests
            correction_method: Method for multiple testing correction ('fdr_bh', 'bonferroni')

        Returns:
            DataFrame with correlation statistics and significance values
        """
        logger.info("Analyzing feature significance for expression metrics...")

        if feature_df is None or feature_df.empty:
            logger.error("Empty feature dataframe provided to analyze_feature_significance")
            return pd.DataFrame()

        # Identify expression metrics and feature columns
        expression_metrics = ["mean", "cv"]
        expression_metrics = [col for col in expression_metrics if col in feature_df.columns]

        if not expression_metrics:
            logger.error("No expression metrics (mean, cv) found in feature dataframe")
            return pd.DataFrame()

        # Identify feature columns (numeric columns excluding expression metrics)
        exclude_cols = [*expression_metrics, "ensembl_transcript_id", "sym"]
        feature_cols = [
            col
            for col in feature_df.select_dtypes(include=np.number).columns
            if col not in exclude_cols
        ]

        if not feature_cols:
            logger.error("No feature columns found in feature dataframe")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(feature_cols)} features for {len(expression_metrics)} metrics")

        # Initialize results storage
        results = []

        # Calculate correlations for each expression metric
        for metric in expression_metrics:
            metric_values = feature_df[metric].values

            # Skip if all values are NaN or constant
            if np.isnan(metric_values).all() or len(np.unique(metric_values)) <= 1:
                logger.warning(f"Expression metric '{metric}' is constant or all NaN. Skipping.")
                continue

            # Get feature region information
            def get_feature_region(feature_name):
                if feature_name.startswith("UTR5_"):
                    return "5' UTR"
                elif feature_name.startswith("UTR3_"):
                    return "3' UTR"
                elif feature_name.startswith("CDS_"):
                    return "CDS"
                else:
                    return "Other"

            # Analyze each feature
            for feature in feature_cols:
                feature_values = feature_df[feature].values
                # Skip if all values are NaN or constant
                if np.isnan(feature_values).all() or len(np.unique(feature_values)) <= 1:
                    continue

                # Drop NaN rows for analysis
                valid_idx = ~(np.isnan(feature_values) | np.isnan(metric_values))
                if np.sum(valid_idx) < 3:  # Need at least 3 valid pairs for correlation
                    continue

                x = feature_values[valid_idx]
                y = metric_values[valid_idx]

                # Calculate Pearson correlation
                try:
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                except Exception:
                    pearson_r, pearson_p = np.nan, np.nan

                # Calculate Spearman correlation
                try:
                    spearman_r, spearman_p = stats.spearmanr(x, y)
                except Exception:
                    spearman_r, spearman_p = np.nan, np.nan

                # Calculate Mutual Information
                try:
                    x_reshaped = x.reshape(-1, 1)
                    mi = mutual_info_regression(x_reshaped, y, random_state=42)[0]
                except Exception:
                    mi = np.nan

                # Store results
                results.append(
                    {
                        "feature": feature,
                        "feature_region": get_feature_region(feature),
                        "expression_metric": metric,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "mutual_info": mi,
                        "n_samples": np.sum(valid_idx),
                    }
                )

        # Convert to DataFrame
        if not results:
            logger.warning("No valid feature-metric correlations found")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Apply multiple testing correction
        for test_type in ["pearson_p", "spearman_p"]:
            if test_type in results_df.columns:
                p_values = results_df[test_type].values
                valid_p = ~np.isnan(p_values)
                if np.sum(valid_p) > 0:
                    try:
                        corrected_p = np.full_like(p_values, np.nan)
                        corrected_p[valid_p] = multipletests(
                            p_values[valid_p], alpha=alpha, method=correction_method
                        )[1]
                        results_df[f"{test_type}_corrected"] = corrected_p
                    except Exception as e:
                        logger.exception(f"Error in multiple testing correction: {e}")

        # Add significance flags based on corrected p-values
        for test_type in ["pearson", "spearman"]:
            corrected_col = f"{test_type}_p_corrected"
            if corrected_col in results_df.columns:
                results_df[f"{test_type}_significant"] = results_df[corrected_col] < alpha

        # Sort by significance
        if "pearson_p_corrected" in results_df.columns:
            results_df = results_df.sort_values("pearson_p_corrected", ascending=True)
        elif "spearman_p_corrected" in results_df.columns:
            results_df = results_df.sort_values("spearman_p_corrected", ascending=True)

        logger.info(f"Completed feature significance analysis with {len(results_df)} results")
        return results_df

    def analyze_feature_interactions(
        self, feature_df: pd.DataFrame, alpha: float = 0.05, top_features: int = 15
    ) -> pd.DataFrame:
        """Analyzes interactions between top sequence features.

        Args:
            feature_df: DataFrame containing sequence features and expression metrics
            alpha: Significance level for interaction tests
            top_features: Number of top features to analyze for interactions

        Returns:
            DataFrame with feature interaction statistics
        """
        logger.info("Analyzing feature interactions...")

        if feature_df is None or feature_df.empty:
            logger.error("Empty feature dataframe provided to analyze_feature_interactions")
            return pd.DataFrame()

        # Identify expression metrics
        expression_metrics = ["mean", "cv"]
        expression_metrics = [col for col in expression_metrics if col in feature_df.columns]

        if not expression_metrics:
            logger.error("No expression metrics (mean, cv) found in feature dataframe")
            return pd.DataFrame()

        # Get feature significance to select top features
        try:
            # Try to calculate feature significance first
            sig_results = self.analyze_feature_significance(feature_df, alpha=alpha)

            if sig_results.empty:
                logger.warning(
                    "No significant features found. Using all numeric features for interaction analysis."
                )
                # Use all numeric features if no significance results
                exclude_cols = [*expression_metrics, "ensembl_transcript_id", "sym"]
                feature_cols = [
                    col
                    for col in feature_df.select_dtypes(include=np.number).columns
                    if col not in exclude_cols
                ]
                if len(feature_cols) > top_features:
                    feature_cols = feature_cols[:top_features]
            else:
                # Get top features for each expression metric
                top_features_by_metric = {}
                for metric in expression_metrics:
                    metric_results = sig_results[sig_results["expression_metric"] == metric]
                    if not metric_results.empty:
                        # Sort by corrected p-value (pearson or spearman)
                        if "pearson_p_corrected" in metric_results.columns:
                            sorted_df = metric_results.sort_values(
                                "pearson_p_corrected", ascending=True
                            )
                        elif "spearman_p_corrected" in metric_results.columns:
                            sorted_df = metric_results.sort_values(
                                "spearman_p_corrected", ascending=True
                            )
                        else:
                            sorted_df = metric_results

                        # Get top features
                        top_features_by_metric[metric] = (
                            sorted_df["feature"].unique()[:top_features].tolist()
                        )

                # Combine top features from all metrics
                feature_cols = []
                for features in top_features_by_metric.values():
                    feature_cols.extend([f for f in features if f not in feature_cols])

                if len(feature_cols) > top_features:
                    feature_cols = feature_cols[:top_features]

        except Exception as e:
            logger.exception(f"Error selecting top features for interaction analysis: {e}")
            exclude_cols = [*expression_metrics, "ensembl_transcript_id", "sym"]
            feature_cols = [
                col
                for col in feature_df.select_dtypes(include=np.number).columns
                if col not in exclude_cols
            ]
            if len(feature_cols) > top_features:
                feature_cols = feature_cols[:top_features]

        if not feature_cols:
            logger.error("No feature columns found for interaction analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing interactions among {len(feature_cols)} top features")

        # Initialize results
        interaction_results = []

        # Analyze interactions for each expression metric
        for metric in expression_metrics:
            y = feature_df[metric].values

            # Skip if target is constant or all NaN
            if np.isnan(y).all() or len(np.unique(y)) <= 1:
                logger.warning(f"Expression metric '{metric}' is constant or all NaN. Skipping.")
                continue

            # Analyze pairwise interactions
            for i, feature1 in enumerate(feature_cols):
                x1 = feature_df[feature1].values

                # Skip if feature1 is constant or all NaN
                if np.isnan(x1).all() or len(np.unique(x1)) <= 1:
                    continue

                for j in range(i + 1, len(feature_cols)):
                    feature2 = feature_cols[j]
                    x2 = feature_df[feature2].values

                    # Skip if feature2 is constant or all NaN
                    if np.isnan(x2).all() or len(np.unique(x2)) <= 1:
                        continue

                    # Remove rows with NaN values
                    valid_idx = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
                    if np.sum(valid_idx) < 5:  # Need enough samples for interaction analysis
                        continue

                    x1_valid = x1[valid_idx]
                    x2_valid = x2[valid_idx]
                    y_valid = y[valid_idx]

                    try:
                        # Fit models with and without interaction
                        X_main = np.column_stack([x1_valid, x2_valid])
                        X_interact = np.column_stack([x1_valid, x2_valid, x1_valid * x2_valid])

                        # Calculate R² for model without interaction
                        model_main = LinearRegression().fit(X_main, y_valid)
                        r2_main = model_main.score(X_main, y_valid)

                        # Calculate R² for model with interaction
                        model_interact = LinearRegression().fit(X_interact, y_valid)
                        r2_interact = model_interact.score(X_interact, y_valid)

                        # Calculate interaction strength and direction
                        interaction_strength = r2_interact - r2_main
                        interaction_coefficient = model_interact.coef_[
                            2
                        ]  # coefficient for interaction term

                        # Calculate significance of interaction
                        _, p_values = f_regression(X_interact, y_valid)
                        interaction_p = p_values[2]  # p-value for interaction term

                        # Determine interaction type
                        if interaction_coefficient > 0:
                            interaction_type = "synergistic"
                        else:
                            interaction_type = "antagonistic"

                        # Store results
                        interaction_results.append(
                            {
                                "feature1": feature1,
                                "feature2": feature2,
                                "expression_metric": metric,
                                "interaction_strength": interaction_strength,
                                "interaction_coefficient": interaction_coefficient,
                                "interaction_p_value": interaction_p,
                                "interaction_type": interaction_type,
                                "r2_without_interaction": r2_main,
                                "r2_with_interaction": r2_interact,
                                "n_samples": np.sum(valid_idx),
                            }
                        )

                    except Exception as e:
                        logger.exception(
                            f"Error analyzing interaction between {feature1} and {feature2}: {e}"
                        )
                        continue

        # Convert to DataFrame
        if not interaction_results:
            logger.warning("No valid feature interactions found")
            return pd.DataFrame()

        interactions_df = pd.DataFrame(interaction_results)

        # Apply multiple testing correction
        if "interaction_p_value" in interactions_df.columns:
            p_values = interactions_df["interaction_p_value"].values
            valid_p = ~np.isnan(p_values)
            if np.sum(valid_p) > 0:
                try:
                    corrected_p = np.full_like(p_values, np.nan)
                    corrected_p[valid_p] = multipletests(
                        p_values[valid_p], alpha=alpha, method="fdr_bh"
                    )[1]
                    interactions_df["interaction_p_value_corrected"] = corrected_p
                except Exception as e:
                    logger.exception(f"Error in multiple testing correction for interactions: {e}")

        # Add significance flag
        if "interaction_p_value_corrected" in interactions_df.columns:
            interactions_df["interaction_significant"] = (
                interactions_df["interaction_p_value_corrected"] < alpha
            )

        # Sort by significance and strength
        interactions_df = interactions_df.sort_values(
            ["interaction_p_value_corrected", "interaction_strength"], ascending=[True, False]
        )

        logger.info(
            f"Completed feature interaction analysis with {len(interactions_df)} interactions"
        )
        return interactions_df

    def analyze_utr5_motif_enrichment(
        self, consistent_genes_df: pd.DataFrame, variable_genes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyzes enrichment of motifs in 5' UTRs of consistently expressed genes.

        Args:
            consistent_genes_df: DataFrame with consistently expressed genes and sequences
            variable_genes_df: DataFrame with variable genes and sequences

        Returns:
            DataFrame with motif enrichment statistics
        """
        logger.info("Analyzing 5' UTR motif enrichment...")

        if (
            consistent_genes_df is None
            or consistent_genes_df.empty
            or variable_genes_df is None
            or variable_genes_df.empty
        ):
            logger.error("Empty dataframes provided to analyze_utr5_motif_enrichment")
            return pd.DataFrame()

        # Check if sequence columns exist
        if (
            "UTR5_Seq" not in consistent_genes_df.columns
            or "UTR5_Seq" not in variable_genes_df.columns
        ):
            logger.error("UTR5_Seq column not found in input dataframes")
            return pd.DataFrame()

        # Extract sequences
        consistent_seqs = consistent_genes_df["UTR5_Seq"].dropna().tolist()
        variable_seqs = variable_genes_df["UTR5_Seq"].dropna().tolist()

        if not consistent_seqs or not variable_seqs:
            logger.error("No valid 5' UTR sequences found in input dataframes")
            return pd.DataFrame()

        logger.info(
            f"Analyzing motif enrichment in {len(consistent_seqs)} consistent and {len(variable_seqs)} variable genes"
        )

        # Define motifs to analyze (predefined + de novo)
        predefined_motifs = {
            "Kozak": REGEX_KOZAK,
            "TOP": REGEX_TOP,
            "G-quadruplex": REGEX_G_QUAD,
            "TISU": re.compile(r"SAASATGGCGGC"),  # Translation Initiator of Short 5'UTR
            "IRES": re.compile(r"CCTATTGTT"),  # Internal Ribosome Entry Site motif
            "uORF": re.compile(r"ATG[ACGT]{3,30}?(TAA|TAG|TGA)"),  # upstream Open Reading Frame
            "GAIT": re.compile(r"ACTCCCC"),  # Gamma interferon activated inhibitor of translation
        }

        # Add additional regulatory elements
        try:
            from Bio import motifs

            BIOPYTHON_MOTIFS_AVAILABLE = True
        except ImportError:
            BIOPYTHON_MOTIFS_AVAILABLE = False

        # Perform de novo motif discovery if Bio.motifs is available
        de_novo_motifs = {}
        if BIOPYTHON_MOTIFS_AVAILABLE and len(consistent_seqs) >= 10:
            try:
                # Simple k-mer based motif discovery
                for k in [4, 5, 6]:
                    # Count k-mers in both sets
                    consistent_kmers = self._count_kmers(consistent_seqs, k)
                    variable_kmers = self._count_kmers(variable_seqs, k)

                    # Find enriched k-mers
                    enriched_kmers = {}
                    for kmer, count in consistent_kmers.items():
                        consistent_freq = count / len(consistent_seqs)
                        variable_freq = variable_kmers.get(kmer, 0) / max(1, len(variable_seqs))

                        if consistent_freq > 0 and consistent_freq > 2 * variable_freq:
                            enrichment = consistent_freq / max(variable_freq, 0.001)
                            enriched_kmers[kmer] = (consistent_freq, variable_freq, enrichment)

                    # Add top enriched motifs
                    top_kmers = sorted(enriched_kmers.items(), key=lambda x: x[1][2], reverse=True)[
                        :5
                    ]
                    for kmer, (_, _, _) in top_kmers:
                        de_novo_motifs[f"de_novo_{k}mer_{kmer}"] = re.compile(kmer)

            except Exception as e:
                logger.exception(f"Error in de novo motif discovery: {e}")

        # Combine predefined and de novo motifs
        all_motifs = {**predefined_motifs, **de_novo_motifs}

        # Analyze motif enrichment
        enrichment_results = []
        for motif_name, motif_regex in all_motifs.items():
            try:
                # Count occurrences in consistent genes
                consistent_count = sum(
                    1 for seq in consistent_seqs if motif_regex.search(seq) is not None
                )
                consistent_freq = consistent_count / len(consistent_seqs)

                # Count occurrences in variable genes
                variable_count = sum(
                    1 for seq in variable_seqs if motif_regex.search(seq) is not None
                )
                variable_freq = variable_count / len(variable_seqs)

                # Calculate enrichment statistics
                fold_enrichment = consistent_freq / max(variable_freq, 0.001)

                # Calculate statistical significance (Fisher's exact test)
                from scipy.stats import fisher_exact

                contingency_table = [
                    [consistent_count, len(consistent_seqs) - consistent_count],
                    [variable_count, len(variable_seqs) - variable_count],
                ]
                odds_ratio, p_value = fisher_exact(contingency_table)

                # Store results
                enrichment_results.append(
                    {
                        "motif_name": motif_name,
                        "motif_pattern": motif_regex.pattern,
                        "consistent_count": consistent_count,
                        "consistent_frequency": consistent_freq,
                        "variable_count": variable_count,
                        "variable_frequency": variable_freq,
                        "fold_enrichment": fold_enrichment,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                    }
                )

            except Exception as e:
                logger.exception(f"Error analyzing enrichment for motif {motif_name}: {e}")
                continue

        # Convert to DataFrame
        if not enrichment_results:
            logger.warning("No motif enrichment results found")
            return pd.DataFrame()

        enrichment_df = pd.DataFrame(enrichment_results)

        # Apply multiple testing correction
        if "p_value" in enrichment_df.columns:
            p_values = enrichment_df["p_value"].values
            valid_p = ~np.isnan(p_values)
            if np.sum(valid_p) > 0:
                try:
                    corrected_p = np.full_like(p_values, np.nan)
                    corrected_p[valid_p] = multipletests(
                        p_values[valid_p], alpha=0.05, method="fdr_bh"
                    )[1]
                    enrichment_df["p_value_corrected"] = corrected_p
                except Exception as e:
                    logger.exception(
                        f"Error in multiple testing correction for motif enrichment: {e}"
                    )

        # Add significance flag
        if "p_value_corrected" in enrichment_df.columns:
            enrichment_df["significant"] = enrichment_df["p_value_corrected"] < 0.05

        # Sort by significance and fold enrichment
        enrichment_df = enrichment_df.sort_values(
            ["p_value_corrected", "fold_enrichment"], ascending=[True, False]
        )

        logger.info(f"Completed 5' UTR motif enrichment analysis with {len(enrichment_df)} motifs")
        return enrichment_df

    def _count_kmers(self, sequences: list[str], k: int) -> dict[str, int]:
        """Count k-mers in a list of sequences.

        Args:
            sequences: List of DNA/RNA sequences
            k: k-mer length

        Returns:
            Dictionary of k-mer counts
        """
        kmer_counts = defaultdict(int)
        for seq in sequences:
            if len(seq) >= k:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    if "N" not in kmer:  # Skip k-mers with ambiguous bases
                        kmer_counts[kmer] += 1
        return kmer_counts

    def analyze_utr3_motif_enrichment(
        self, consistent_genes_df: pd.DataFrame, variable_genes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyzes enrichment of motifs in 3' UTRs of consistently expressed genes.

        Args:
            consistent_genes_df: DataFrame with consistently expressed genes and sequences
            variable_genes_df: DataFrame with variable genes and sequences

        Returns:
            DataFrame with motif enrichment statistics
        """
        logger.info("Analyzing 3' UTR motif enrichment...")

        if (
            consistent_genes_df is None
            or consistent_genes_df.empty
            or variable_genes_df is None
            or variable_genes_df.empty
        ):
            logger.error("Empty dataframes provided to analyze_utr3_motif_enrichment")
            return pd.DataFrame()

        # Check if sequence columns exist
        if (
            "UTR3_Seq" not in consistent_genes_df.columns
            or "UTR3_Seq" not in variable_genes_df.columns
        ):
            logger.error("UTR3_Seq column not found in input dataframes")
            return pd.DataFrame()

        # Extract sequences
        consistent_seqs = consistent_genes_df["UTR3_Seq"].dropna().tolist()
        variable_seqs = variable_genes_df["UTR3_Seq"].dropna().tolist()

        if not consistent_seqs or not variable_seqs:
            logger.error("No valid 3' UTR sequences found in input dataframes")
            return pd.DataFrame()

        logger.info(
            f"Analyzing motif enrichment in {len(consistent_seqs)} consistent and {len(variable_seqs)} variable genes"
        )

        # Define motifs to analyze (predefined)
        predefined_motifs = {
            "Polyadenylation Signal": REGEX_POLYA,
            "AU-rich Element": REGEX_ARE,
            "miRNA Seed Match": REGEX_MIRNA_SEED,
            "GU-rich Element": re.compile(r"GTGTGT"),
            "Stem-loop Structure": re.compile(
                r"[ACGT]{4,8}[ACGT]{4,8}"
            ),  # Simple stem-loop pattern
            "CPE": re.compile(r"TTTTAT"),  # Cytoplasmic Polyadenylation Element
            "Musashi Binding Element": re.compile(r"GUAGU|AUAGU"),  # Musashi protein binding site
        }

        # Perform de novo motif discovery similar to 5' UTR analysis
        de_novo_motifs = {}
        try:
            # Simple k-mer based motif discovery
            for k in [4, 5, 6]:
                # Count k-mers in both sets
                consistent_kmers = self._count_kmers(consistent_seqs, k)
                variable_kmers = self._count_kmers(variable_seqs, k)

                # Find enriched k-mers
                enriched_kmers = {}
                for kmer, count in consistent_kmers.items():
                    consistent_freq = count / len(consistent_seqs)
                    variable_freq = variable_kmers.get(kmer, 0) / max(1, len(variable_seqs))

                    if consistent_freq > 0 and consistent_freq > 2 * variable_freq:
                        enrichment = consistent_freq / max(variable_freq, 0.001)
                        enriched_kmers[kmer] = (consistent_freq, variable_freq, enrichment)

                # Add top enriched motifs
                top_kmers = sorted(enriched_kmers.items(), key=lambda x: x[1][2], reverse=True)[:5]
                for kmer, (_, _, _) in top_kmers:
                    de_novo_motifs[f"de_novo_{k}mer_{kmer}"] = re.compile(kmer)

        except Exception as e:
            logger.exception(f"Error in de novo motif discovery for 3' UTR: {e}")

        # Combine predefined and de novo motifs
        all_motifs = {**predefined_motifs, **de_novo_motifs}

        # Analyze motif enrichment
        enrichment_results = []
        for motif_name, motif_regex in all_motifs.items():
            try:
                # Count occurrences in consistent genes
                consistent_count = sum(
                    1 for seq in consistent_seqs if motif_regex.search(seq) is not None
                )
                consistent_freq = consistent_count / len(consistent_seqs)

                # Count occurrences in variable genes
                variable_count = sum(
                    1 for seq in variable_seqs if motif_regex.search(seq) is not None
                )
                variable_freq = variable_count / len(variable_seqs)

                # Calculate enrichment statistics
                fold_enrichment = consistent_freq / max(variable_freq, 0.001)

                # Calculate statistical significance (Fisher's exact test)
                from scipy.stats import fisher_exact

                contingency_table = [
                    [consistent_count, len(consistent_seqs) - consistent_count],
                    [variable_count, len(variable_seqs) - variable_count],
                ]
                odds_ratio, p_value = fisher_exact(contingency_table)

                # Store results
                enrichment_results.append(
                    {
                        "motif_name": motif_name,
                        "motif_pattern": motif_regex.pattern,
                        "consistent_count": consistent_count,
                        "consistent_frequency": consistent_freq,
                        "variable_count": variable_count,
                        "variable_frequency": variable_freq,
                        "fold_enrichment": fold_enrichment,
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                    }
                )

            except Exception as e:
                logger.exception(f"Error analyzing enrichment for motif {motif_name}: {e}")
                continue

        # Convert to DataFrame
        if not enrichment_results:
            logger.warning("No motif enrichment results found")
            return pd.DataFrame()

        enrichment_df = pd.DataFrame(enrichment_results)

        # Apply multiple testing correction
        if "p_value" in enrichment_df.columns:
            p_values = enrichment_df["p_value"].values
            valid_p = ~np.isnan(p_values)
            if np.sum(valid_p) > 0:
                try:
                    corrected_p = np.full_like(p_values, np.nan)
                    corrected_p[valid_p] = multipletests(
                        p_values[valid_p], alpha=0.05, method="fdr_bh"
                    )[1]
                    enrichment_df["p_value_corrected"] = corrected_p
                except Exception as e:
                    logger.exception(
                        f"Error in multiple testing correction for motif enrichment: {e}"
                    )

        # Add significance flag
        if "p_value_corrected" in enrichment_df.columns:
            enrichment_df["significant"] = enrichment_df["p_value_corrected"] < 0.05

        # Sort by significance and fold enrichment
        enrichment_df = enrichment_df.sort_values(
            ["p_value_corrected", "fold_enrichment"], ascending=[True, False]
        )

        logger.info(f"Completed 3' UTR motif enrichment analysis with {len(enrichment_df)} motifs")
        return enrichment_df

    # --- Comparative Sequence Analysis Methods ---
    def compare_with_reference_genes(
        self,
        consistent_genes_df: pd.DataFrame,
        variable_genes_df: pd.DataFrame,
        reference_genome: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compares sequence features between consistently expressed and variably expressed genes.

        Args:
            consistent_genes_df: DataFrame containing consistently expressed genes
            variable_genes_df: DataFrame containing variably expressed genes
            reference_genome: Optional reference genome identifier (e.g., 'CHO-K1')

        Returns:
            Dictionary of DataFrames with comparison results
        """
        logger.info("Comparing sequence features between consistent and variable genes...")

        results = {}

        if consistent_genes_df is None or consistent_genes_df.empty:
            logger.error(
                "Empty consistent genes DataFrame provided to compare_with_reference_genes"
            )
            return results

        if variable_genes_df is None or variable_genes_df.empty:
            logger.error("Empty variable genes DataFrame provided to compare_with_reference_genes")
            return results

        # Feature categories to compare
        feature_categories = {
            "5'UTR Features": [
                col
                for col in consistent_genes_df.columns
                if col.startswith("UTR5_") and col not in ["UTR5_Seq"]
            ],
            "3'UTR Features": [
                col
                for col in consistent_genes_df.columns
                if col.startswith("UTR3_") and col not in ["UTR3_Seq"]
            ],
            "CDS Features": [
                col
                for col in consistent_genes_df.columns
                if col.startswith("CDS_") and col not in ["CDS_Seq"]
            ],
            "General Features": [
                col
                for col in consistent_genes_df.columns
                if not (
                    col.startswith(("UTR5_", "UTR3_", "CDS_"))
                    or col in ["ensembl_transcript_id", "sym", "mean", "cv"]
                )
            ],
        }

        # Compare each feature category
        for category, features in feature_categories.items():
            if not features:
                logger.warning(f"No features found for category: {category}")
                continue

            logger.info(f"Comparing {len(features)} features in category: {category}")

            # Initialize comparison results
            comparison_data = []

            for feature in features:
                if (
                    feature not in consistent_genes_df.columns
                    or feature not in variable_genes_df.columns
                ):
                    logger.warning(f"Feature {feature} not found in both DataFrames. Skipping.")
                    continue

                # Get feature values, dropping NaNs
                consistent_values = consistent_genes_df[feature].dropna()
                variable_values = variable_genes_df[feature].dropna()

                if len(consistent_values) < 3 or len(variable_values) < 3:
                    logger.warning(f"Insufficient non-NaN values for feature {feature}. Skipping.")
                    continue

                # Calculate statistics
                try:
                    # Calculate basic statistics
                    consistent_mean = np.mean(consistent_values)
                    consistent_median = np.median(consistent_values)
                    consistent_std = np.std(consistent_values)

                    variable_mean = np.mean(variable_values)
                    variable_median = np.median(variable_values)
                    variable_std = np.std(variable_values)

                    # Calculate fold change
                    if variable_mean == 0:
                        fold_change = np.inf if consistent_mean > 0 else np.nan
                    else:
                        fold_change = consistent_mean / variable_mean

                    # Perform statistical test (Mann-Whitney U test)
                    u_stat, p_value = stats.mannwhitneyu(
                        consistent_values, variable_values, alternative="two-sided"
                    )

                    # Calculate effect size (Cohen's d)
                    n1, n2 = len(consistent_values), len(variable_values)
                    pooled_std = np.sqrt(
                        ((n1 - 1) * consistent_std**2 + (n2 - 1) * variable_std**2) / (n1 + n2 - 2)
                    )

                    if pooled_std == 0:
                        effect_size = np.nan
                    else:
                        effect_size = (consistent_mean - variable_mean) / pooled_std

                    # Store results
                    comparison_data.append(
                        {
                            "feature": feature,
                            "consistent_mean": consistent_mean,
                            "consistent_median": consistent_median,
                            "consistent_std": consistent_std,
                            "variable_mean": variable_mean,
                            "variable_median": variable_median,
                            "variable_std": variable_std,
                            "fold_change": fold_change,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "n_consistent": len(consistent_values),
                            "n_variable": len(variable_values),
                        }
                    )

                except Exception as e:
                    logger.exception(f"Error comparing feature {feature}: {e}")
                    continue

            # Convert to DataFrame
            if not comparison_data:
                logger.warning(f"No comparison results for category: {category}")
                continue

            comparison_df = pd.DataFrame(comparison_data)

            # Apply multiple testing correction
            if "p_value" in comparison_df.columns:
                p_values = comparison_df["p_value"].values
                valid_p = ~np.isnan(p_values)
                if np.sum(valid_p) > 0:
                    try:
                        corrected_p = np.full_like(p_values, np.nan)
                        corrected_p[valid_p] = multipletests(
                            p_values[valid_p], alpha=0.05, method="fdr_bh"
                        )[1]
                        comparison_df["p_value_corrected"] = corrected_p
                    except Exception as e:
                        logger.exception(
                            f"Error in multiple testing correction for {category}: {e}"
                        )

            # Add significance flag
            if "p_value_corrected" in comparison_df.columns:
                comparison_df["significant"] = comparison_df["p_value_corrected"] < 0.05

            # Sort by significance and effect size
            comparison_df = comparison_df.sort_values(
                ["p_value_corrected", "effect_size"], ascending=[True, False]
            )

            # Store results
            results[category] = comparison_df

            logger.info(f"Completed comparison for {category} with {len(comparison_df)} features")

        # Add overall summary
        significant_features = []
        for category, df in results.items():
            if "significant" in df.columns:
                sig_features = df[df["significant"]]["feature"].tolist()
                significant_features.extend([(category, feature) for feature in sig_features])

        if significant_features:
            summary_data = []
            for category, feature in significant_features:
                df = results[category]
                feature_row = df[df["feature"] == feature].iloc[0]
                summary_data.append(
                    {
                        "category": category,
                        "feature": feature,
                        "fold_change": feature_row["fold_change"],
                        "p_value_corrected": feature_row["p_value_corrected"],
                        "effect_size": feature_row["effect_size"],
                        "direction": "higher in consistent"
                        if feature_row["fold_change"] > 1
                        else "lower in consistent",
                    }
                )

            results["significant_features_summary"] = pd.DataFrame(summary_data).sort_values(
                "p_value_corrected", ascending=True
            )

            logger.info(f"Found {len(significant_features)} significant differential features")

        return results

    def analyze_rna_structure(
        self,
        sequences_df: pd.DataFrame,
        region: str = "UTR5",
        max_sequences: int = 100,
        structure_method: str = "mfe",
    ) -> dict[str, Any]:
        """Analyzes RNA secondary structure characteristics of gene sequences.

        Args:
            sequences_df: DataFrame containing sequence data
            region: Sequence region to analyze ('UTR5', 'UTR3', or 'CDS')
            max_sequences: Maximum number of sequences to analyze
            structure_method: Method for structure prediction ('mfe' or 'centroid')

        Returns:
            Dictionary with RNA structure analysis results
        """
        logger.info(f"Analyzing RNA secondary structure for {region} regions...")

        results = {"structures": {}, "metrics": pd.DataFrame(), "status": "failed"}

        if sequences_df is None or sequences_df.empty:
            logger.error("Empty sequences DataFrame provided to analyze_rna_structure")
            return results

        # Check for required column
        seq_column = f"{region}_Seq"
        if seq_column not in sequences_df.columns:
            logger.error(f"Required column {seq_column} not found in sequences_df")
            return results

        # Try to import ViennaRNA Python bindings
        try:
            import RNA

            VIENNA_RNA_AVAILABLE = True
        except ImportError:
            logger.warning("ViennaRNA Python bindings not available. Using fallback method.")
            VIENNA_RNA_AVAILABLE = False

        # Extract sequence IDs and sequences
        sequences = {}
        for _, row in sequences_df.iterrows():
            if pd.notna(row.get(seq_column)) and row.get("ensembl_transcript_id"):
                seq = str(row[seq_column]).upper()
                if "T" in seq:  # Convert DNA to RNA
                    seq = seq.replace("T", "U")
                sequences[row["ensembl_transcript_id"]] = seq

                # Limit number of sequences to analyze
                if len(sequences) >= max_sequences:
                    break

        if not sequences:
            logger.error(f"No valid {region} sequences found for RNA structure analysis")
            return results

        logger.info(f"Analyzing RNA structure for {len(sequences)} {region} sequences")

        # Initialize results storage
        structure_results = {}
        metrics_data = []

        # Process each sequence
        for seq_id, seq in sequences.items():
            try:
                # Skip sequences that are too short
                if len(seq) < 10:
                    logger.warning(f"Sequence {seq_id} too short for RNA structure analysis")
                    continue

                # Calculate structures and metrics
                if VIENNA_RNA_AVAILABLE:
                    # Use ViennaRNA for structure prediction
                    fc = RNA.fold_compound(seq)

                    if structure_method == "centroid":
                        structure, mfe = fc.centroid()
                    else:  # Default to MFE
                        structure, mfe = fc.mfe()

                    # Calculate ensemble diversity
                    ensemble_diversity = fc.mean_bp_distance()

                    # Calculate base pairing probabilities
                    fc.pf()  # Run partition function calculation
                    bp_probabilities = fc.bpp()

                    # Calculate base pair entropy
                    bp_entropy = np.sum(-bp_probabilities * np.log(bp_probabilities + 1e-10))

                    # Identify structural elements
                    hairpins = structure.count("(") - structure.count(".")
                    loops = structure.count(".")

                    # Store structure
                    structure_results[seq_id] = {
                        "sequence": seq,
                        "structure": structure,
                        "mfe": mfe,
                        "ensemble_diversity": ensemble_diversity,
                        "bp_entropy": bp_entropy,
                    }

                    # Store metrics
                    metrics_data.append(
                        {
                            "ensembl_transcript_id": seq_id,
                            "length": len(seq),
                            "mfe": mfe,
                            "normalized_mfe": mfe / len(seq),
                            "ensemble_diversity": ensemble_diversity,
                            "bp_entropy": bp_entropy,
                            "hairpins": hairpins,
                            "loops": loops,
                            "gc_content": sum(1 for n in seq if n in "GC") / len(seq),
                        }
                    )

                else:
                    # Fallback to simple structure prediction
                    # This is a simple energy model and not very accurate
                    structure = self._simple_structure_prediction(seq)

                    # Calculate simple metrics
                    hairpins = structure.count("(") - structure.count(".")
                    loops = structure.count(".")
                    gc_content = sum(1 for n in seq if n in "GC") / len(seq)

                    # Store structure
                    structure_results[seq_id] = {
                        "sequence": seq,
                        "structure": structure,
                        "simple_estimate": True,
                    }

                    # Store metrics
                    metrics_data.append(
                        {
                            "ensembl_transcript_id": seq_id,
                            "length": len(seq),
                            "hairpins": hairpins,
                            "loops": loops,
                            "gc_content": gc_content,
                        }
                    )

            except Exception as e:
                logger.exception(f"Error analyzing RNA structure for sequence {seq_id}: {e}")
                continue

        # Convert metrics to DataFrame
        if metrics_data:
            results["metrics"] = pd.DataFrame(metrics_data)
            results["structures"] = structure_results
            results["status"] = "success"
            logger.info(f"Completed RNA structure analysis for {len(structure_results)} sequences")
        else:
            logger.warning("No RNA structure results generated")

        return results

    def _simple_structure_prediction(self, sequence: str) -> str:
        """Simple RNA secondary structure prediction (fallback method).

        This is a very simplified model and should only be used when ViennaRNA is not available.

        Args:
            sequence: RNA sequence

        Returns:
            Dot-bracket notation of predicted structure
        """
        # Convert to uppercase and ensure it's RNA
        sequence = sequence.upper().replace("T", "U")
        n = len(sequence)

        # Initialize structure with unpaired bases
        structure = ["." for _ in range(n)]

        # Simple stack to track opening brackets
        stack = []

        # Simple complementary base pairs
        complementary = {"A": "U", "U": "A", "G": "C", "C": "G"}

        # Minimum loop size
        min_loop_size = 3

        # Scan for potential base pairs
        for i in range(n):
            if i + min_loop_size + 1 < n:  # Ensure there's enough space for a minimal loop
                for j in range(n - 1, i + min_loop_size, -1):
                    # Check if bases can pair
                    if sequence[i] in complementary and sequence[j] == complementary[sequence[i]]:
                        # Check if both positions are unpaired
                        if structure[i] == "." and structure[j] == ".":
                            # Mark as paired
                            structure[i] = "("
                            structure[j] = ")"
                            break

        return "".join(structure)
