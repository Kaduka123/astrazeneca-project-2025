# CHO Analysis - Project Methodology

## TOC

- [CHO Analysis - Project Methodology](#cho-analysis---project-methodology)
  - [TOC](#toc)
  - [Introduction](#introduction)
  - [Shared Infrastructure](#shared-infrastructure)
    - [Data Loading Framework](#data-loading-framework)
    - [Configuration Management](#configuration-management)
    - [Logging System](#logging-system)
  - [Task 1 Pipeline: Marker Gene Identification](#task-1-pipeline-marker-gene-identification)
    - [1. Data Loading](#1-data-loading)
    - [2. Batch Effect Correction](#2-batch-effect-correction)
    - [3. Correlation Analysis](#3-correlation-analysis)
    - [4. Gene Ranking](#4-gene-ranking)
    - [5. Marker Panel Optimization](#5-marker-panel-optimization)
    - [6. Binary Marker Identification](#6-binary-marker-identification)
    - [7. Platform Consensus Analysis](#7-platform-consensus-analysis)
    - [8. Task 1 Visualizations](#8-task-1-visualizations)
    - [Experimental Approach for Task 1](#experimental-approach-for-task-1)
  - [Task 2 Pipeline: Sequence Feature Analysis](#task-2-pipeline-sequence-feature-analysis)
    - [1. Consistent Gene Identification](#1-consistent-gene-identification)
    - [2. Sequence Data Processing and Merging](#2-sequence-data-processing-and-merging)
    - [3. UTR Feature Analysis (`analyze_utr_features()`)](#3-utr-feature-analysis-analyze_utr_features)
    - [4. CDS \& Codon Usage Analysis (`analyze_cds_features()`, `_calculate_codon_stats()`)](#4-cds--codon-usage-analysis-analyze_cds_features-_calculate_codon_stats)
    - [5. Feature Integration and Categorization](#5-feature-integration-and-categorization)
    - [6. Sequence-Based Expression Prediction](#6-sequence-based-expression-prediction)
      - [Feature Selection and Model Building](#feature-selection-and-model-building)
      - [Prediction and Analysis](#prediction-and-analysis)
      - [Sequence Optimization Simulation](#sequence-optimization-simulation)
      - [Visualizations](#visualizations)
      - [Integration with Task 2 Pipeline](#integration-with-task-2-pipeline)
    - [7. Task 2 Visualizations (`task2/visualization.py`)](#7-task-2-visualizations-task2visualizationpy)
    - [Experimental Approach for Task 2](#experimental-approach-for-task-2)
  - [Execution Framework](#execution-framework)
  - [Technical Implementation Details](#technical-implementation-details)
    - [Key Libraries](#key-libraries)
    - [Performance Considerations](#performance-considerations)
    - [Visualization Consistency](#visualization-consistency)
  - [Data Preprocessing](#data-preprocessing)
    - [Data Loading and Cleaning](#data-loading-and-cleaning)
    - [Batch Effect Detection](#batch-effect-detection)
    - [Batch Effect Correction](#batch-effect-correction)
  - [Correlation Analysis](#correlation-analysis)
    - [Correlation Calculation](#correlation-calculation)
    - [Bootstrap Analysis](#bootstrap-analysis)
    - [Multi-criteria Ranking](#multi-criteria-ranking)
  - [Sequence Analysis](#sequence-analysis)
    - [Sequence Feature Extraction](#sequence-feature-extraction)
    - [Feature Analysis](#feature-analysis)
    - [Expression Prediction](#expression-prediction)
    - [Optimization Strategy Development](#optimization-strategy-development)

## Introduction

This document details the technical methodology for the CHO cell line analysis project. The primary goal is to identify robust gene expression markers correlated with monoclonal antibody production (Light Chain - LC) and to characterize sequence features of highly stable genes. The project implements two distinct analytical pipelines:

1. **Task 1**: A comprehensive workflow for identifying and ranking potential marker genes based on correlation, stability, and expression metrics, including batch correction and panel optimization strategies. The aim is to provide reliable, data-driven candidates for accelerating clone selection in biotherapeutic production.
2. **Task 2**: An analysis pipeline to characterize sequence features (UTR motifs, codon usage) of consistently expressed genes identified across experiments. The objective is to uncover sequence characteristics associated with high, stable expression, potentially informing future transgene design for improved protein yield.

These pipelines leverage shared core components for data handling, configuration, and logging, ensuring modularity and consistency. A key feature is the ability to run predefined, reproducible experiments using a configuration file ([`experiments.toml`](../../experiments.toml)), facilitating systematic evaluation of different analytical strategies.

## Shared Infrastructure

All shared infrastructure is located in the [`cho_analysis/core`](../../cho_analysis/core) directory.

### Data Loading Framework

- Found in [`core/data_loader.py`](../../cho_analysis/core/data_loader.py)
- The `DataLoader` class centralizes access to input data, promoting consistency and reducing redundancy.
  - `load_expression_data()`: Parses tab-separated expression files (using `pandas`), designed to handle common bioinformatics formats with potential metadata columns (`ensembl_transcript_id`, `sym`). Handles potential file variations and ensures basic data integrity checks.
  - `load_manifest_data()`: Loads sample metadata (`MANIFEST.txt`), essential for linking expression data to experimental conditions like sequencing platform, which is crucial for batch correction.
  - `load_all_sequence_data()`: Uses the robust BioPython library (`Bio.SeqIO`) to parse standard FASTA formatted sequence files, handling potential variations in header formats.
- Implements error handling (e.g., `FileNotFoundError`, `pd.errors.EmptyDataError`, `ValueError`) to manage common issues gracefully.
- Sequence loading is conditional on BioPython installation to allow core functionality even if sequence analysis dependencies are missing.

### Configuration Management

- Found in [`core/config.py`](../../cho_analysis/core/config.py)
- Employs a flexible, layered configuration system prioritizing runtime flexibility and reproducibility:
  - **Code Defaults**: Sensible defaults are defined within `config.py` getter functions (e.g., `get_correlation_config`) as a baseline.
  - **`.env` File**: Allows users to set persistent base configurations or secrets (though not used for secrets here) via a standard `.env` file in the project root.
  - **`experiments.toml`**: Enables defining specific, named parameter sets for reproducible experimental runs (see Execution Framework). Environment variables set from the selected experiment override `.env` and defaults.
  - **Command-Line Arguments**: Provides the highest level of control, allowing users to override any setting for a specific run via flags (e.g., `--methods`, `--skip-batch-correction`).
- `get_env()` helper uses `os.environ.get` to read environment variables (respecting the hierarchy above) and includes type conversion (`_convert_value`) for robustness.
- Centralizes path management (`get_path`, `get_file_path`) and ensures necessary output directories exist (`setup_directories`).

### Logging System

- Found in [`core/logging.py`](../../cho_analysis/core/logging.py)
- Provides informative and structured logging essential for monitoring pipeline execution and debugging.
- Leverages Python's standard `logging` integrated with `rich` for enhanced readability.
  - `RichHandler`: Delivers colorized, formatted console output with timestamps and log levels. Enables features like `rich_tracebacks`.
  - `RotatingFileHandler`: Archives logs to timestamped files (`logs/YYYY-MM-DD_*.log`), preventing excessively large log files.
- Configuration (level, format, targets) is managed via `get_logging_config()`, integrating with the overall configuration system.

## Task 1 Pipeline: Marker Gene Identification

The objective of Task 1 is to identify genes whose expression levels reliably correlate with the expression of the target recombinant protein (LC), making them potential biomarkers for selecting high-producing CHO cell clones.

### 1. Data Loading

- **Orchestrator:** `load_data_task1` function in `scripts/run_analysis.py`.
- **Mechanism:** Utilizes the shared `DataLoader` to load `expression_counts.txt` and `MANIFEST.txt`.
- **Justification:** Centralized loading ensures consistent data handling and initial validation (checking for file existence, basic structure).

### 2. Batch Effect Correction

- **Modules:** [`task1/batch_correction.py`](../../cho_analysis/task1/batch_correction.py), [`task1/advanced_batch_correction.py`](../../cho_analysis/task1/advanced_batch_correction.py).
- **Rationale:** Integrating RNA-Seq data from diverse sources (different labs, sequencing platforms) necessitates addressing technical batch effects. These non-biological variations can mask true signals or introduce spurious correlations, impacting the reliability of marker identification. This module aims to mitigate such effects.
- **Standard Correction (`BatchCorrection` class):**
  - `detect_batch_effect()`: Performs initial assessment.
    - *ANOVA*: Identifies the percentage of genes significantly differing (p < 0.05) across platforms, indicating the extent of the batch effect.
    - *Silhouette Score*: Quantifies batch separation in PCA space. The score ($S$) ranges from -1 to +1: $S = (b - a) / \max(a, b)$, where $a$ is the mean intra-batch distance and $b$ is the mean nearest-batch distance. Values near +1 indicate strong batch separation. [Reference: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of computational and applied mathematics*, 20, 53-65].
    - Platform labels are derived from manifest metadata (column `platform` or extracted from `description`). Correction is triggered if effects appear substantial (default: >20% genes significant by ANOVA or |Silhouette Score| > 0.1).
  - `correct_batch()`: Applies ComBat-Seq [Reference: Zhang, Y., et al. (2020). ComBat-seq: batch effect adjustment for RNA-seq count data. *NAR genomics and bioinformatics*, 2(3), lqaa078] using the `pycombat_seq` implementation. This empirical Bayes method adjusts data for known batches (platform) while attempting to preserve biological heterogeneity, crucial for downstream correlation. The target gene (`PRODUCT-TG`) is explicitly excluded from adjustment calculations to retain its original variance structure for correlation analysis.
  - `compare_pca_stages()`: Generates PCA plots (saved as `results/figures/task1/batch_correction_standard_pca.png`) comparing sample distribution before and after standard correction.
- **Advanced Correction (`AdvancedBatchCorrection` class - Optional):** Provides deeper diagnostics and targeted correction if `CHO_ANALYSIS_BATCH_CORRECTION_RUN_ADVANCED=true` and standard correction was performed.
  - `detect_residual_platform_effects()`: Assesses if platform effects persist post-ComBat using ANOVA and Kruskal-Wallis H tests [Reference: Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American statistical Association*, 47(260), 583-621] on a per-gene basis. Controls the False Discovery Rate (FDR) using Benjamini-Hochberg [Reference: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate... *Journal of the Royal statistical society: series B*, 57(1), 289-300]. Results saved to `results/tables/task1/residual_platform_effects.csv`. A heatmap visualizing expression patterns of top affected genes can be generated (`results/figures/task1/residual_platform_effects_heatmap.png`).
  - `quantify_platform_bias()`: Measures the magnitude of residual bias per platform using Cohen's d effect size ($d = (\mu_1 - \mu_2) / \sigma_{pooled}$) between a platform and all others. Results saved (e.g., `results/tables/task1/platform_bias_HiSeq.csv`).
  - `hierarchical_batch_correction()`: If residual effects are significant (default >5% genes with adjusted p < `advanced_alpha`) and a biased platform is identified, applies a targeted location-scale adjustment ($X_{adj} = (X_{prob} - \mu_{prob}) / \sigma_{prob} \times \sigma_{ref} + \mu_{ref}$) to the data from the problematic platform (`prob`) relative to reference platforms (`ref`), preserving the target gene.
  - `validate_correction()`: Quantifies overall improvement by comparing the final corrected data to the original using metrics like Silhouette score, kNN mixing rate (proportion of nearest neighbors belonging to the same batch), and the average proportion of genes showing significant distributional differences between platforms via the Kolmogorov-Smirnov test [Pointer: See `scipy.stats.ks_2samp` documentation]. Results saved to `results/tables/task1/correction_validation_metrics.json`.
  - The correction validation supports the requirement to "Validate effectiveness of batch correction" as specified in the project requirements.

### 3. Correlation Analysis

- **Module:** [`task1/correlation.py`](../../cho_analysis/task1/correlation.py).
- **Rationale:** To identify potential markers, we must quantify the association between each host gene's expression and the target product's (LC) expression. Employing multiple methods captures diverse relationship types (linear, monotonic, complex non-linear).
- **Class:** `GeneCorrelationAnalysis`.
- `prepare_data()`: Converts data into an `AnnData` object [Reference: Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome biology*, 19(1), 1-5], storing LC expression in `adata.obs`.
- `calculate_correlations()`: Computes correlations using methods selected via `CHO_ANALYSIS_CORRELATION_METHODS`:
  - *Statistical*:
    - Spearman's rank correlation ($\rho$): Measures monotonic relationships by correlating the ranks. Formula: $\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ where $d_i$ is the difference in ranks.
    - Pearson's product-moment correlation ($r$): Measures linear relationships. Formula: $r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$
    - Kendall's rank correlation ($\tau$): Measures ordinal association based on concordant and discordant pairs. Formula: $\tau = \frac{2(n_c - n_d)}{n(n-1)}$ where $n_c$ is the number of concordant pairs and $n_d$ is the number of discordant pairs.
  - *ML-based* (addressing the non-linear relationships requirement):
    - Linear Regression: Calculates $\beta$ coefficient and OLS p-value via `statsmodels.api.OLS`. The coefficient represents the expected change in LC expression for a one-unit change in gene expression.
    - Decision Tree: Uses cross-validated $R^2$ score to capture non-linear relationships through recursive binary splitting.
    - Random Forest: Ensemble method that averages multiple decision trees, measuring feature importance for each gene. Implemented via `sklearn.ensemble.RandomForestRegressor`.
  - FDR correction (Benjamini-Hochberg) applied to p-values from statistical methods to control for multiple testing.
  - `Correlation_Rank_Score`: Defined as $|\text{Correlation Coefficient}| / (p_{value} + \epsilon)$ (where $\epsilon=10^{-10}$) to combine magnitude and significance.
- **Bootstrap Analysis** (`bootstrap_correlation_analysis()`): Optional (`CHO_ANALYSIS_CORRELATION_RUN_BOOTSTRAP=true`). Assesses result stability.
  - *Principle*: Repeatedly resamples the dataset with replacement (N=`CHO_ANALYSIS_CORRELATION_BOOTSTRAP_ITERATIONS`), recalculating correlations for each sample to build empirical distributions.
  - *Mathematical Basis*: For each gene, the bootstrap generates an empirical distribution of correlation coefficients $\{r_1, r_2, ..., r_B\}$ where $B$ is the number of bootstrap iterations. From this, we calculate:
    - Bootstrap mean: $\bar{r}_{boot} = \frac{1}{B}\sum_{i=1}^{B} r_i$
    - Bootstrap standard error: $SE_{boot} = \sqrt{\frac{1}{B-1}\sum_{i=1}^{B}(r_i - \bar{r}_{boot})^2}$
    - Percentile-based 95% Confidence Interval: $[r_{(0.025B)}, r_{(0.975B)}]$ where $r_{(k)}$ is the $k$-th ordered statistic
  - *Rank Stability %*: Calculated as the frequency a gene appears in the top `CHO_ANALYSIS_CORRELATION_TOP_N` ranks across bootstrap iterations.
  - *Visualizations:* `results/figures/task1/bootstrap_confidence_{method}.png`, `results/figures/task1/rank_stability_heatmap_{method}.png`.
- `compare_methods()`: Evaluates and ranks correlation methods based on:
  1. Average correlation strength of top genes
  2. Statistical significance (mean -log10(p-value))
  3. Predictive power (cross-validated $R^2$ of Linear Regression using top genes from each method)
- `find_consensus_markers()`: Identifies genes significant across multiple methods, based on `min_correlation` and `max_p_value` thresholds.
- **Output:** `results/tables/task1/correlation_results_full.csv` contains all calculated metrics.

### 4. Gene Ranking

- **Module:** [`task1/ranking.py`](../../cho_analysis/task1/ranking.py).
- **Rationale:** Prioritize markers based not only on correlation strength/significance but also on expression characteristics relevant for biomarker utility (detectability, stability) and statistical robustness (bootstrap rank stability).
- **Class:** `GeneRanking`.
- `run_ranking_analysis()`: Calculates a weighted `final_score` per gene combining four key components:
  - *Mathematical Definition*: $Score_{final} = w_{corr} \cdot Score_{corr} + w_{level} \cdot Score_{level} + w_{stab} \cdot Score_{stab} + w_{rank} \cdot Score_{rank}$
  - *Component Scores* (all normalized to 0-1 range):
    - $Score_{corr}$: Normalized correlation score based on correlation strength and statistical significance
    - $Score_{level}$: Expression level score favoring genes with higher expression (more easily detectable)
    - $Score_{stab}$: Expression stability score based on inverse of Coefficient of Variation ($CV = \sigma / \mu$)
    - $Score_{rank}$: Bootstrap rank stability score measuring consistency of gene's importance across resampled datasets
  - *Default Weights*: $w_{corr}=0.40$, $w_{level}=0.30$, $w_{stab}=0.15$, $w_{rank}=0.15$
- **Output:** Ranked list saved to `results/tables/task1/ranked_markers.csv`.
- *Visualization:* `results/figures/task1/gene_ranking_scores.png` (includes radar plot of score components for top genes).

### 5. Marker Panel Optimization

- **Module:** [`task1/marker_panels.py`](../../cho_analysis/task1/marker_panels.py).
- **Rationale:** Evaluate if small combinations (panels) of top markers offer better predictive performance than single markers, potentially capturing complementary information or increasing robustness. Optional step via `CHO_ANALYSIS_CORRELATION_RUN_PANEL_OPTIMIZATION=true`.
- **Class:** `MarkerPanelOptimization`.
  - Selects candidates from top ranked genes (`CHO_ANALYSIS_CORRELATION_PANEL_CANDIDATES`).
  - `_calculate_mutual_information()`: Estimates pairwise redundancy using Mutual Information (MI):
    - *Mathematical Definition*: $I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log(\frac{p(x,y)}{p(x)p(y)})$
    - This metric quantifies the shared information between genes, capturing non-linear dependencies
  - `_greedy_forward_selection()`: Builds panels of sizes `CHO_ANALYSIS_CORRELATION_PANEL_SIZES` using two optimization strategies:
    - `minimal_redundancy`: Balances individual gene score with low mutual information to other selected genes.
      - *Selection Score*: $S(g_i) = Score(g_i) - \frac{1}{|P|}\sum_{g_j \in P} MI(g_i, g_j)$ where $P$ is the current panel
    - `max_score`: Prioritizes genes with highest `final_score` regardless of redundancy
  - `_evaluate_panel_loocv()`: Assesses panel performance using Leave-One-Out Cross-Validation:
    - For each sample $i$, train a linear model on all samples except $i$
    - Predict sample $i$ and compute $R^2$ and RMSE across all held-out predictions
    - This provides a nearly unbiased estimate of panel's predictive performance
- **Output:** Panel evaluations saved to `results/tables/task1/marker_panel_evaluation.csv`.
- *Visualizations:* `results/figures/task1/panel_comparison.png`, `results/figures/task1/panel_information_gain.png`.

### 6. Binary Marker Identification

- **Module:** [`task1/ranking.py`](../../cho_analysis/task1/ranking.py).
- **Rationale:** Identify genes useful as simple threshold-based indicators of high/low productivity states, important for rapid clone screening.
- `binary_marker_identification()`:
  - Binarizes the target LC gene based on a percentile threshold (e.g., top 25% as "high producers").
  - Evaluates genes as binary classifiers using:
    - Area Under the ROC Curve (AUC): Measures discrimination ability across all possible thresholds
    - Cross-validated AUC: Uses `LogisticRegression` with `StratifiedKFold` to estimate generalization performance
  - Mathematical basis: The AUC equals the probability that a randomly chosen positive instance ranks higher than a randomly chosen negative one:
    - $AUC = P(score(x^+) > score(x^-))$ where $x^+$ is a positive instance and $x^-$ is a negative instance
- `find_optimal_thresholds()`: Determines gene expression thresholds that maximize balanced accuracy:
  - *Balanced Accuracy*: $\frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$, which gives equal weight to positive and negative class accuracy
- **Output:** Tables `results/tables/task1/binary_markers_auc.csv`, `results/tables/task1/optimal_thresholds.csv`.

### 7. Platform Consensus Analysis

- **Module:** [`task1/correlation.py`](../../cho_analysis/task1/correlation.py).
- **Rationale:** Ensure identified markers are robust across potential technical variations introduced by different sequencing platforms.
- `analyze_platform_consensus()`:
  - Performs correlation analysis separately within each platform's data subset.
  - Ranks genes based on:
    - `platforms_count`: Number of platforms where the gene shows significant correlation
    - `same_direction`: Whether the correlation direction (positive/negative) is consistent across platforms
    - `avg_abs_correlation`: Average magnitude of correlation across platforms
  - Mathematical formulation: For each gene $g$:
    - $Consensus_{g} = \frac{\sum_{p=1}^{P} I(|corr_{g,p}| > threshold \land p\_value_{g,p} < \alpha)}{P}$ where $I$ is the indicator function and $P$ is the number of platforms
- `analyze_method_consensus_overlap()`: Compares method results with platform consensus using overlap metrics like Jaccard Index:
  - $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$ where $A$ and $B$ are sets of top genes identified by different methods
- **Output:** `results/tables/task1/platform_consensus_markers.csv`.
- *Visualizations:* `results/figures/task1/platform_consensus_ranking.png`, `results/figures/task1/method_consensus_overlap_barplot.png`, `results/figures/task1/method_consensus_overlap_venn.png`.

### 8. Task 1 Visualizations

- **Module:** [`task1/visualization.py`](../../cho_analysis/task1/visualization.py).
- **Rationale:** Provide comprehensive visual summaries and diagnostics for effective interpretation and communication of Task 1 findings.
- **Class:** `CorrelationVisualization`. Generates plots saved to `results/figures/task1/`. Key visualizations include:
  - **Correlation Analysis Visualizations:**
    - Co-correlation heatmap (`plot_correlation_heatmap`): Visualizes the product of correlation coefficients between gene pairs.
    - Pairwise scatter plots (`plot_scatter_matrix`): Shows direct relationships between top genes and target LC expression.
    - Expression distribution by platform (`plot_expression_by_platform`): Visualizes expression differences across platforms.
  - **Bootstrap Analysis Visualizations:**
    - Bootstrap confidence intervals (`plot_bootstrap_confidence`): Shows correlation uncertainty through empirical resampling.
    - Rank stability heatmap (`plot_rank_stability_heatmap`): Displays consistency of gene ranks across bootstrap iterations.
  - **Panel Analysis Visualizations:**
    - Panel performance comparison (`plot_panel_comparison`): Compares $R^2$ and RMSE of different panels.
    - Performance vs. Panel Size (`plot_information_gain`): Shows how predictive power changes with additional genes.
  - **Consensus Analysis Visualizations:**
    - Consensus ranking bar chart (`visualize_consensus_ranking`): Highlights genes consistently identified across methods.
    - Method/Consensus overlap plots (`visualize_method_overlap`): Venn diagrams or upset plots showing agreement between methods.
  - **Batch Effect Visualizations:**
    - PCA plots (`plot_pca_comparison`): Shows sample clustering before/after batch correction.
    - Residual batch effect heatmap (`plot_platform_effect_heatmap`): Displays genes with persistent platform effects.

### Experimental Approach for Task 1

To systematically investigate the impact of different analysis parameters and methods within Task 1, a series of predefined experiments are configured in [`experiments.toml`](../../experiments.toml). These allow for reproducible runs focusing on specific aspects:

- **Experiment 1: Baseline Correlation**
  - **Purpose:** Establish a reference point using only Spearman correlation and minimal, data-triggered batch correction. Skips computationally intensive steps like bootstrapping and panel optimization.
  - **Focus:** Core correlation identification (`calculate_correlations` with Spearman).
- **Experiment 2: Assess Advanced Batch Correction**
  - **Purpose:** Isolate the effect of the advanced batch correction module (residual detection, bias quantification, potential hierarchical correction) compared to the baseline.
  - **Focus:** Enables `CHO_ANALYSIS_BATCH_CORRECTION_RUN_ADVANCED=true`. Compares the resulting Spearman correlations and validation metrics to Experiment 1.
- **Experiment 3: Explore All Correlation Methods**
  - **Purpose:** Evaluate how different correlation metrics (statistical and basic ML) influence marker discovery, keeping advanced batch correction enabled.
  - **Focus:** Enables all methods in `CHO_ANALYSIS_CORRELATION_METHODS`. Generates data for method comparison (`compare_methods`) and consensus ranking.
- **Experiment 4: Add Bootstrap Confidence & Stability**
  - **Purpose:** Quantify the robustness of markers identified by all methods using bootstrap resampling.
  - **Focus:** Enables `CHO_ANALYSIS_CORRELATION_RUN_BOOTSTRAP=true`. Adds confidence intervals and rank stability scores to the correlation results.
- **Experiment 5: Full Analysis with Panel Optimization**
  - **Purpose:** Execute the complete, most comprehensive Task 1 workflow, including the design and evaluation of small marker panels.
  - **Focus:** Enables `CHO_ANALYSIS_CORRELATION_RUN_PANEL_OPTIMIZATION=true`. Leverages results from previous steps (ranking, potentially bootstrapping) to propose and evaluate marker panels (`design_marker_panels`).

Running these experiments sequentially allows for a step-by-step evaluation of each major analytical component's contribution to the final set of proposed markers.

## Task 2 Pipeline: Sequence Feature Analysis

The objective of Task 2 is to identify sequence characteristics (UTR motifs, codon usage) associated with genes exhibiting high and stable expression, potentially providing insights for optimizing future transgene constructs.

### 1. Consistent Gene Identification

- **Module:** [`task2/sequence_analysis.py`](../../cho_analysis/task2/sequence_analysis.py).
- **Class:** `SequenceFeatureAnalysis`.
- `find_consistently_expressed_genes()`: Selects a subset of genes for detailed sequence analysis.
  - **Rationale:** Focusing on genes reliably expressed across *all* conditions ensures that identified sequence features are associated with general expression robustness, not condition-specific regulation. It also reduces computational load.
  - Filters based on minimum expression level (ensuring presence) and low Coefficient of Variation (CV) across samples (indicating stability), using configurable thresholds.

### 2. Sequence Data Processing and Merging

- `_preprocess_fasta_df()`: Parses raw FASTA sequence lists (from `DataLoader`) into structured DataFrames, linking sequences to Ensembl transcript IDs and handling header variations.
- Merging (in `run_analysis()` orchestrator): Joins the expression statistics (mean, CV) of the consistently expressed genes with their corresponding CDS, 5'UTR, and 3'UTR sequences. Handles cases where sequences might be missing for some transcripts.

### 3. UTR Feature Analysis (`analyze_utr_features()`)

- **Rationale:** UTRs contain numerous regulatory elements influencing mRNA stability, localization, and translation efficiency. Identifying enriched motifs in stable, highly expressed genes can guide transgene UTR design.
- Calculates basic features: Length, GC Content (often linked to stability/structure).
- Detects known regulatory motifs via optimized regex searches (patterns defined in config):
  - 5'UTR: Kozak (translation initiation), TOP (translation regulation), G-quadruplexes (structure/regulation).
  - 3'UTR: PolyA signals (cleavage/polyadenylation), AREs (mRNA decay), miRNA seed matches (post-transcriptional silencing).

### 4. CDS & Codon Usage Analysis (`analyze_cds_features()`, `_calculate_codon_stats()`)

- **Rationale:** Codon usage is non-random and affects translation speed, efficiency, and accuracy, impacting protein folding and yield. Analyzing usage in highly expressed native genes can inform codon optimization strategies for the transgene.
- Calculates:
  - **Codon Usage Frequency**: The proportion of each codon relative to all codons in the CDS.
  - **GC3 Content**: GC content specifically at the third, often 'wobble', codon position, which strongly correlates with expression levels in many organisms.
  - **Codon Adaptation Index (CAI)**: Measures how closely a gene's codon usage matches the usage pattern of a reference set (here, implicitly, the set of consistently expressed genes themselves), indicating translational adaptation.
- `extract_codon_usage_table()`: Creates a gene-by-codon matrix for easier comparison and visualization.

### 5. Feature Integration and Categorization

- Combines all calculated UTR and CDS features with expression metrics (mean, CV) for each consistently expressed gene.
- `categorize_stability()`: Groups genes into stability quantiles (e.g., 'High', 'Medium', 'Low' stability based on CV) to facilitate comparisons of sequence features between stability groups.

### 6. Sequence-Based Expression Prediction

- **Module:** [`task2/sequence_modeling.py`](../../cho_analysis/task2/sequence_modeling.py).
- **Rationale:** Establishing a quantitative relationship between sequence features and expression metrics enables prediction of gene expression based solely on sequence characteristics. This predictive capacity allows for informed optimization of transgene sequences before experimental validation.
- **Classes:** `SequenceExpressionModeling` implements the core predictive functionality, while `SequenceModelingVisualization` provides visual analysis tools.

#### Feature Selection and Model Building

- `select_predictive_features()`: Identifies sequence features with the highest predictive power using multiple methods:
  - **Statistical**: Uses F-regression to select features with the strongest univariate relationship with the target (CV or mean expression)
  - **Recursive Feature Elimination (RFE)**: Iteratively eliminates features, retaining those that maximize cross-validated performance
  - **LASSO**: Performs feature selection through L1 regularization, which naturally produces sparse feature sets
  - Features are ranked by normalized importance scores, with a configurable maximum number of features retained (default: 15)

- `build_expression_prediction_model()`: Constructs regression models to predict expression metrics from selected sequence features:
  - **Linear**: Ridge regression, which adds L2 regularization to mitigate overfitting
  - **Tree**: Gradient Boosting Regressor, which captures non-linear relationships
  - **Ensemble**: Combines predictions from multiple model types (default option)
  - Performance is assessed through cross-validation using multiple metrics (R², RMSE)
  - Mathematical basis for cross-validation: Data is split into K folds; for each fold i, the model is trained on all folds except i and tested on fold i, producing K performance estimates that are averaged for robustness

#### Prediction and Analysis

- `predict_expression_from_sequence()`: Applies trained models to predict expression metrics directly from sequence features:
  - Takes a DataFrame of sequence features as input
  - Performs necessary preprocessing (scaling, imputation)
  - Generates predictions from all trained models and averages them (for ensembles)
  - Provides prediction confidence intervals based on cross-validation performance
  - Returns a DataFrame containing both the original features and the predictions

- `analyze_feature_importance()`: Quantifies the contribution of each feature to the prediction:
  - For linear models: Uses coefficient magnitude
  - For tree-based models: Extracts feature importance from the model
  - For all models: Calculates permutation importance by randomly shuffling each feature and measuring the decrease in prediction accuracy
  - Normalizes importance scores for comparability and averages across models

#### Sequence Optimization Simulation

- `simulate_sequence_optimization()`: Uses trained models to explore the sequence feature space and identify optimal regions:
  - Systematically varies individual features while holding others constant
  - Predicts expression metrics for each variation
  - Identifies optimal values that minimize CV (for stability) or maximize mean expression
  - Quantifies the potential improvement over baseline values
  - Prioritizes features based on their optimization impact and importance
  - Mathematical formulation: For each feature f, find the value v that maximizes/minimizes the target metric T: v_optimal = argmax/min_v T(f=v, all_other_features=baseline)

#### Visualizations

- `plot_feature_importance()`: Bar chart showing normalized importance scores for top features
- `plot_prediction_performance()`: Scatter plot comparing predicted vs. actual expression metrics with performance statistics
- `plot_shap_summary()`: SHapley Additive exPlanations (SHAP) summary plot showing feature contributions to individual predictions
- `plot_optimization_contour()`: Contour plot visualizing how varying two key features affects the predicted expression metric

#### Integration with Task 2 Pipeline

- Executed after base sequence feature extraction, using the combined feature set from consistently expressed genes
- Configuration controlled via `get_sequence_analysis_config()`:
  - `run_expression_prediction`: Enables/disables the functionality (default: enabled)
  - `prediction_target`: Selects the metric to predict ('cv' or 'mean')
  - `prediction_model_type`: Selects model architecture ('linear', 'tree', or 'ensemble')
  - `feature_selection_method`: Specifies feature selection approach ('statistical', 'rfe', or 'lasso')
  - `max_features`: Limits the number of features used for prediction
- Output saved to CSV files (`results/tables/task2/selected_predictive_features.csv`, `results/tables/task2/model_performance_metrics.csv`, `results/tables/task2/sequence_based_predictions.csv`)
- Visualizations saved to PNG files (`results/figures/task2/feature_importance.png`, `results/figures/task2/prediction_performance.png`)

### 7. Task 2 Visualizations (`task2/visualization.py`)

- **Rationale:** Visual exploration helps identify trends and patterns linking sequence features to expression stability.
- **Class:** `SequenceFeatureVisualization`. Generates plots including:
  - Distributions (`plot_utr_length_distribution`, `plot_gc_distribution`): Show the overall landscape of UTR lengths and GC content.
  - Scatter plots (`plot_cv_vs_feature`): Directly visualize the relationship between expression stability (CV) and specific features like length, GC3, or CAI.
  - Bar charts (`plot_regulatory_element_counts`, `plot_average_codon_usage`): Summarize the prevalence of key motifs and preferred codons.
  - Heatmaps (`plot_codon_usage_heatmap`): Provide a detailed view of codon usage patterns, potentially sorted by gene stability.

### Experimental Approach for Task 2

To systematically investigate different aspects of sequence feature analysis and expression prediction, a series of predefined experiments are configured in [`experiments.toml`](../../experiments.toml). These allow for reproducible runs focusing on specific aspects of Task 2:

- **Experiment 6: Baseline Sequence Feature Analysis**
  - **Purpose:** Establish a reference point using only the core sequence feature extraction and analysis, without prediction or advanced analyses.
  - **Focus:** Basic sequence feature extraction and statistics (`analyze_utr_features`, `analyze_cds_features`) without predictive modeling or comparison.

- **Experiment 7: Basic Expression Prediction**
  - **Purpose:** Introduce predictive modeling with a straightforward configuration - statistical feature selection and ensemble modeling.
  - **Focus:** Enables `CHO_ANALYSIS_SEQUENCE_ANALYSIS_RUN_EXPRESSION_PREDICTION=true` using the default statistical feature selection method to predict expression stability (CV).

- **Experiment 8: Feature Selection Methods Comparison**
  - **Purpose:** Compare different feature selection strategies (statistical, RFE, LASSO) to identify which approach best identifies predictive sequence features.
  - **Focus:** Enables multiple feature selection methods with a linear model to isolate the impact of feature selection on prediction performance.

- **Experiment 9: Model Type Comparison**
  - **Purpose:** Evaluate how different model architectures (linear, tree-based, ensemble) perform on predicting both expression stability (CV) and level (mean).
  - **Focus:** Enables multiple model types and prediction targets, along with comparative analysis to contextualize the findings.

- **Experiment 10: Full Analysis with Transgene Design**
  - **Purpose:** Execute the complete Task 2 workflow, including feature significance, expression prediction, comparative analysis, and transgene design recommendations.
  - **Focus:** Enables all components, including `CHO_ANALYSIS_SEQUENCE_ANALYSIS_RUN_TRANSGENE_DESIGN=true`, to generate sequence optimization recommendations based on the predictive models.

Running these experiments sequentially provides a comprehensive evaluation of the sequence-based expression prediction pipeline, from basic feature analysis to advanced transgene design recommendations, isolating the contribution of each component to the final results.

## Execution Framework

- **Main Script:** [`scripts/run_analysis.py`](../../scripts/run_analysis.py) serves as the central orchestrator.
- **Task Selection:** Controlled by the `--task` flag (0 for both, 1, or 2).
- **Experiment Mode:**
  - Activated by the `--experiment N` flag.
  - `apply_experiment_config()` function reads [`experiments.toml`](../../experiments.toml) and sets relevant `CHO_ANALYSIS_*` environment variables.
  - This allows running predefined, documented parameter combinations easily.
- **Modular Steps:** The `@analysis_step` decorator wraps major functional units, providing clear logging boundaries, timing, and basic error isolation using `rich` console elements.
- **Docker Environment:** Primarily designed for execution within a Docker container via `docker compose run`, ensuring environment consistency and dependency management. Input/output directories (`data`, `results`, `logs`) are managed via volume mounts specified in [`docker-compose.yml`](../../docker-compose.yml).
- **Output Management:** Standardized saving functions (`save_dataframe`, `save_visualization`) place results in structured directories (`results/tables/{task}`, `results/figures/{task}`) and log saving actions. Processed data intended for Task 2 input is saved to `data/preprocessed/`.

## Technical Implementation Details

### Key Libraries

- **Core Data Handling**: `pandas` (DataFrames), `numpy` (numerical operations).
- **Bioinformatics**: `biopython` (Sequence parsing/GC content), `scanpy` & `anndata` (Expression data structure for Task 1 correlation).
- **Statistics/ML**: `scipy.stats` (correlations, statistical tests), `statsmodels` (FDR correction, OLS), `scikit-learn` (PCA, Regression models, CV, metrics, MI).
- **Batch Correction**: `inmoose`/`pycombat` (ComBat-Seq implementation).
- **Visualization**: `matplotlib` (base plotting), `seaborn` (statistical plots), `matplotlib-venn` (overlap diagrams).
- **Configuration**: `tomllib`/`tomli` (Parsing `experiments.toml`).
- **CLI/Logging**: `argparse` (Command-line interface), `logging` (Standard logging), `rich` (Enhanced console logging and UI elements).

### Performance Considerations

- Standard libraries (`numpy`, `pandas`, `scikit-learn`) utilize optimized C or Cython backends for many operations.
- Bootstrapping (`task1.correlation`) and some ML methods can be time-consuming; parameters (`bootstrap_iterations`, `n_estimators`) allow control.
- Data is primarily processed in memory; very large datasets might require memory profiling or alternative strategies not currently implemented (e.g., chunking, disk-based data structures).

### Visualization Consistency

- Shared constants (`core/visualization_utils.py`) define default palettes (`CMAP_SEQUENTIAL`, `CMAP_DIVERGING`, `PALETTE_QUALITATIVE`, task-specific colors), fonts, and DPI settings.
- Base `matplotlib.rcParams` are updated for global consistency.
- Visualization classes include error handling to produce informative messages or blank plots on failure, rather than crashing the pipeline.

## Data Preprocessing

### Data Loading and Cleaning

The raw expression data (`expression_counts.txt`) is loaded and cleaned through the following steps:

1. Read tab-delimited expression count file with pandas
2. Filter out low-quality samples based on quality control metrics
3. Remove non-expressed genes (count < 10 in more than 80% of samples)
4. Log-transform counts (log2(counts + 1)) to stabilize variance
5. Store processed dataset for downstream analysis

### Batch Effect Detection

Batch effects in the expression data are assessed and quantified (gene-wise ANOVA F-tests, PCA with Silhouette scoring):

1. Identify experimental factors (cell line, platform, lab) from metadata
2. Calculate variance explained by each factor through ANOVA
3. Visualize batch effects using PCA and boxplots
4. Determine significance of batch effects using statistical tests

### Batch Effect Correction

If significant batch effects are detected, correction methods are applied (ComBat-Seq for standard correction):

1. Apply ComBat-Seq to remove batch effects while preserving biological variation
2. Evaluate corrected data for potential overcorrection
3. Re-assess batch effect presence after correction
4. Apply advanced correction methods if ComBat-Seq is insufficient

## Correlation Analysis

### Correlation Calculation

To identify genes most strongly associated with target gene expression, multiple correlation methods are applied (Spearman, Pearson, Kendall, Random Forest importance, Regression coefficients):

1. Calculate pairwise correlations between each gene and the target gene
2. Assess statistical significance and apply FDR correction
3. Rank genes by correlation strength
4. Create visualizations of top correlated genes

### Bootstrap Analysis

Bootstrap resampling is performed to assess the robustness of correlations (100 iterations, 95% CI):

1. Randomly sample datasets with replacement
2. Calculate correlations for each bootstrap sample
3. Determine confidence intervals for correlation estimates
4. Assess consistency of gene rankings across bootstrap samples

### Multi-criteria Ranking

Genes are ranked using multiple criteria to identify the most reliable markers (weighted final score via harmonic mean):

1. Correlation strength
2. Expression level
3. Statistical significance
4. Bootstrap consistency
5. Cross-validation performance

## Sequence Analysis

### Sequence Feature Extraction

For consistently expressed genes, sequence features are extracted from 5' UTR, 3' UTR, and CDS regions:

1. Parse FASTA files for different sequence regions (Bio.SeqIO)
2. Calculate basic sequence metrics (length, GC content)
3. Identify regulatory motifs and sequence patterns (regex pattern matching)
4. Compute codon usage statistics (codon usage frequencies)
5. Analyze RNA secondary structure potential

### Feature Analysis

The relationship between sequence features and expression stability is analyzed:

1. Calculate correlation between each feature and expression stability
2. Apply statistical tests to assess significance
3. Create visualizations to illustrate key relationships
4. Identify the most predictive sequence features

### Expression Prediction

Machine learning models are developed to predict expression from sequence features (Random Forest, Gradient Boosting, Elastic Net):

1. Split data into training and test sets
2. Perform feature selection (mutual information, SHAP values)
3. Train multiple model types and compare performance
4. Evaluate models using cross-validation and independent test data
5. Interpret model outputs to identify key predictive features

### Optimization Strategy Development

Based on model insights, optimization strategies for transgene design are developed:

1. Identify optimal ranges for key sequence features
2. Develop scoring function for sequence quality assessment
3. Simulate optimization impact on expression stability
4. Create guidelines for sequence design
