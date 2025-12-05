# CHO Analysis

## Overview

The CHO Analysis project is a Python-based application that provides a comprehensive set of tools for analyzing and visualizing gene expression data from CHO cells. The project is designed to be modular and extensible, allowing for easy addition of new features and data sources.

The project analyzes CHO (Chinese Hamster Ovary) cell line RNA-seq data to address two main tasks:

1. **Task 1**: Identify genes with expression levels correlated with the final product (LC) for marker-based clone selection.
   - Performs batch correction to normalize data across different sequencing platforms
   - Calculates correlation between gene expression and target product
   - Ranks genes based on correlation strength, expression stability, and other metrics
   - Identifies optimal marker panels for clone selection
   - Generates visualizations for correlation patterns

2. **Task 2**: Characterize sequence features (5' UTR, 3' UTR, CDS regions, codon usage) of consistently expressed genes across experiments, develop sequence-based expression prediction models, and provide optimization strategies for transgene design.
   - Analyzes sequence features of consistently expressed genes
   - Identifies significant feature patterns that correlate with expression stability
   - Creates prediction models for expression level and variability
   - Provides optimization recommendations for transgene design
   - Visualizes sequence characteristics and their impact on expression

## Getting Started

### Prerequisites

> IMPORTANT
>
> Please make sure you have all the prerequisites. Otherwise, the application won't run.

- **Git** for cloning the repository.
- **Docker Desktop** (Windows, macOS) or **Docker Engine + Docker Compose plugin** (Linux) installed and running.
- A **terminal** or **command prompt**.
- Minimum system requirements:
  - 4GB RAM (8GB recommended)
  - 10GB free disk space
  - 2 CPU cores

### Cloning the Repository

First, clone the repository to your local machine:

```bash
# Clone the repository
git clone https://github.com/EMATM0050-2024/dsmp-2024-groupt29.git

# Navigate to the project directory
cd dsmp-2024-groupt29
```

### Docker Image Build

1. **Build the Docker Image**

Open your terminal/command prompt in the project's root directory (where [`docker-compose.yml`](docker-compose.yml) is located) and run:

```bash
docker compose build
```

This builds the `analysis` service image. You only need to do this once, or when dependencies in [`pyproject.toml`](pyproject.toml) change.

## Running the Analysis

You can run predefined experimental configurations or specify analysis tasks and parameters manually.

### Running Predefined Experiments

The project includes predefined experimental setups defined in the [`experiments.toml`](experiments.toml) file. Each experiment targets specific tasks and uses a defined set of parameters for reproducibility.

To list all experiments, run:

```bash
# List all available experiments
docker compose run --rm analysis python -m scripts.run_analysis --list-experiments
```

To run a specific experiment, use the `--experiment` flag followed by the experiment number:

```bash
# Example: Run Experiment 1 (Task 1 Baseline)
docker compose run --rm analysis python -m scripts.run_analysis --experiment 1

# Example: Run Experiment 5 (Full Task 1 Analysis)
docker compose run --rm analysis python -m scripts.run_analysis --experiment 5

# Example: Run Experiment 6 (Task 2 Baseline Sequence Analysis)
docker compose run --rm analysis python -m scripts.run_analysis --experiment 6

# Example: Run Experiment 10 (Full Task 2 Analysis with Transgene Design)
docker compose run --rm analysis python -m scripts.run_analysis --experiment 10
```

- The `--experiment` flag automatically sets the correct `--task` and overrides `default/.env` settings with those defined in [`experiments.toml`](experiments.toml) for that experiment number.
- For a detailed technical breakdown of each experiment's configuration and purpose, refer to the [Project Methodology](docs/project/project_methodology.md) document.

#### Available Experiments

Here's a summary of the key predefined experiments:

| Experiment | Task | Description |
|------------|------|-------------|
| 1 | 1 | Task 1 Baseline - Basic correlation analysis without batch correction |
| 2 | 1 | Task 1 with Standard Batch Correction - Corrects platform-specific effects |
| 3 | 1 | Task 1 with Advanced Batch Correction - Uses hierarchical correction for residual effects |
| 4 | 1 | Task 1 with Bootstrap Analysis - Includes confidence intervals and rank stability |
| 5 | 1 | Full Task 1 Analysis - Comprehensive analysis with all Task 1 features |
| 6 | 2 | Task 2 Baseline - Basic sequence feature analysis |
| 7 | 2 | Task 2 with Feature Significance - Statistical analysis of feature importance |
| 8 | 2 | Task 2 with Comparative Analysis - Compares consistently expressed genes with variable ones |
| 9 | 2 | Task 2 with Expression Prediction - Builds models to predict expression from sequence |
| 10 | 2 | Full Task 2 Analysis - Comprehensive sequence analysis with transgene design |

### Running Manually (Overriding Experiments or Defaults)

You can still run specific tasks or override parameters manually using command-line flags. Flags provided will take precedence over settings defined by the `--experiment` flag or defaults.

```bash
# Example: Run only Task 2
docker compose run --rm analysis python -m scripts.run_analysis --task 2

# Example: Run Experiment 4, but override bootstrap iterations
docker compose run --rm analysis python -m scripts.run_analysis --experiment 4 --override-env CHO_ANALYSIS_CORRELATION_BOOTSTRAP_ITERATIONS=500

# Example: Run Task 1 manually, skipping batch correction and using specific methods
docker compose run --rm analysis python -m scripts.run_analysis --task 1 --skip-batch-correction --methods pearson spearman
```

#### Available Command-line Flags

The following table provides a comprehensive reference of all available command-line flags and their possible values:

| Flag | Type | Possible Values | Default | Description |
|------|------|----------------|---------|-------------|
| `--task` | Integer | `0`, `1`, `2` | `0` | Task to run: `0` (Both tasks), `1` (Correlation analysis), `2` (Sequence features analysis). This is overridden if `--experiment` is specified. |
| `--experiment` | Integer | Any experiment number defined in `experiments.toml` | `None` | Runs a predefined experiment configuration. Overrides `--task` setting and sets environment variables defined in the experiment. |
| `--list-experiments` | Flag | N/A | `False` | Lists all available experiments defined in `experiments.toml` and exits. |
| `--methods` | String(s) | `pearson`, `spearman`, `kendall` | From config | Overrides correlation methods for Task 1. Can specify multiple methods separated by spaces. |
| `--top-n` | Integer | Any positive integer | From config | Overrides the number of top genes to analyze and rank. |
| `--skip-batch-correction` | Flag | N/A | `False` | When specified, skips the batch correction step in Task 1. |
| `--verbose` | Flag | N/A | `False` | Enables detailed DEBUG level logging for troubleshooting. |
| `--override-env` | Key-Value | Any environment variable in format `KEY=VALUE` | N/A | Overrides specific environment variables for the current run. Used with Docker Compose. Multiple variables can be specified. |

#### Environment Variables for Overriding

The following environment variables can be used with `--override-env` for fine-grained control:

| Environment Variable | Purpose | Example Value |
|----------------------|---------|--------------|
| `CHO_ANALYSIS_CORRELATION_TARGET_GENE` | Sets the target gene for correlation | `"PRODUCT-TG"` |
| `CHO_ANALYSIS_CORRELATION_METHODS` | Sets correlation methods | `"spearman,pearson"` |
| `CHO_ANALYSIS_CORRELATION_TOP_N` | Number of top genes | `50` |
| `CHO_ANALYSIS_CORRELATION_BOOTSTRAP_ITERATIONS` | Bootstrap iterations | `1000` |
| `CHO_ANALYSIS_CORRELATION_RUN_BOOTSTRAP` | Enable/disable bootstrap | `true` |
| `CHO_ANALYSIS_CORRELATION_CONFIDENCE_LEVEL` | Bootstrap confidence level | `0.95` |
| `CHO_ANALYSIS_BATCH_CORRECTION_SKIP` | Skip batch correction | `true` |
| `CHO_ANALYSIS_BATCH_CORRECTION_RUN_ADVANCED` | Run advanced correction | `true` |
| `CHO_ANALYSIS_BATCH_CORRECTION_ADVANCED_ALPHA` | Alpha for advanced correction | `0.01` |

### Verbose Logging

To enable verbose logging, use the `--verbose` flag. This will enable detailed `DEBUG` level logging.

```bash
docker compose run --rm analysis python -m scripts.run_analysis --experiment 1 --verbose
```

## Understanding the Results

Results and logs will appear in the `./results` and `./logs` directories in your project folder on your host machine. Environment variables from a `.env` file in the project root will also be loaded into the container.

### Output Directory Structure

```
results/
├── figures/
│   ├── task1/            # Visualizations from correlation analysis
│   │   ├── batch_correction_standard_pca.png
│   │   ├── correlation_heatmap_spearman.png
│   │   ├── correlation_network_spearman.png
│   │   └── ...
│   └── task2/            # Visualizations from sequence analysis
│       ├── feature_importance_cv.png
│       ├── gc_content_dist.png
│       ├── utr_length_dist.png
│       └── ...
└── tables/
    ├── task1/            # Data tables from correlation analysis
    │   ├── correlation_results_full.csv
    │   ├── ranked_markers.csv
    │   ├── platform_consensus_markers.csv
    │   └── ...
    └── task2/            # Data tables from sequence analysis
        ├── combined_sequence_features.csv
        ├── feature_significance.csv
        ├── sequence_based_predictions_cv.csv
        ├── optimal_feature_profiles.csv
        └── design_recommendations.json
```

### Key Results by Task

#### Task 1 Results

- **ranked_markers.csv**: The primary output containing ranked genes for marker-based selection
- **correlation_results_full.csv**: Detailed correlation values between genes and target product
- **marker_panel_evaluation.csv**: Performance of different marker panel combinations
- **binary_markers_auc.csv**: Potential binary markers with thresholds for high/low expression
- **platform_consensus_markers.csv**: Markers consistent across different sequencing platforms

#### Task 2 Results

- **combined_sequence_features.csv**: Extracted features from gene sequences
- **feature_significance.csv**: Statistical significance of sequence features
- **sequence_based_predictions_cv.csv**: Predictions from the expression model
- **optimal_feature_profiles.csv**: Optimal feature values for transgene design
- **design_recommendations.json**: Specific recommendations for transgene optimization

## Troubleshooting

### Common Issues

1. **Docker Issues**
   - Error: `Cannot connect to the Docker daemon`
     - Solution: Make sure Docker is running on your system

   - Error: `Error response from daemon: driver failed programming external connectivity`
     - Solution: Restart Docker

2. **File Permission Issues**
   - Error: `Permission denied when saving file`
     - Solution: Check that your user has write permissions to the mounted volumes

3. **Missing Data Files**
   - Error: `Failed to load expression data`
     - Solution: Verify that all required data files are in the correct locations in the `/data` directory

4. **Memory Issues**
   - Error: `Container killed due to out-of-memory`
     - Solution: Increase the memory allocation for Docker in Docker Desktop settings

### Getting Help

If you encounter persistent issues:
1. Check the log files in the `./logs` directory for detailed error messages
2. Review the [Project Methodology](docs/project/project_methodology.md) for technical details
3. Contact the project maintainers with the error logs attached

## Project Structure

```text
.                                                         # Project Root
├── cho_analysis                                          # Main source code package
│   ├── core                                              # Core functionalities
│   │   ├── config.py                                     # Configuration handling
│   │   ├── data_loader.py                                # Data loading utilities
│   │   ├── __init__.py                                   # Package initialization
│   │   ├── logging.py                                    # Logging configuration
│   │   ├── utils.py                                      # General utilities
│   │   └── visualization_utils.py                        # Visualization utilities
│   ├── __init__.py                                       # Package initialization
│   ├── task1                                             # Task 1: Correlation analysis
│   │   ├── advanced_batch_correction.py                  # Advanced batch correction
│   │   ├── batch_correction.py                           # Batch correction algorithms
│   │   ├── correlation.py                                # Correlation analysis
│   │   ├── __init__.py                                   # Package initialization
│   │   ├── marker_panels.py                              # Marker panel generation
│   │   ├── ranking.py                                    # Gene ranking methods
│   │   └── visualization.py                              # Task 1 visualizations
│   └── task2                                             # Task 2: Sequence analysis
│       ├── __init__.py                                   # Package initialization
│       ├── comparative_analysis.py                       # Comparative sequence analysis
│       ├── feature_significance.py                       # Feature significance testing
│       ├── sequence_analysis.py                          # Sequence feature analysis
│       ├── sequence_modeling.py                          # Sequence-based expression prediction and optimization
│       ├── transgene_design.py                           # Transgene optimization strategies
│       └── visualization.py                              # Task 2 visualizations
├── data                                                  # Data directory
│   ├── preprocessed                                      # Preprocessed data files
│   └── raw                                               # Raw input data
│       ├── 3UTR_sequences.fasta                          # 3' UTR sequences
│       ├── 5UTR_sequences.fasta                          # 5' UTR sequences
│       ├── CDS_sequences.fasta                           # Coding DNA Sequences
│       ├── expression_counts.txt                         # Raw gene expression counts (primary data file used by pipeline)
│       ├── expression.txt                                # Alternative expression data format (not used in analysis)
│       ├── FILE_DESCRIPTION.txt                          # Description of data files
│       └── MANIFEST.txt                                  # Sample information
├── docker                                                # Docker configuration
│   ├── Dockerfile                                        # Main Production Dockerfile
│   └── entrypoint.sh                                     # Container entrypoint script
├── docker-compose.yml                                    # Docker Compose configuration
├── Dockerfile.dev                                        # Development Dockerfile
├── docs                                                  # Documentation
│   ├── dev                                               # Developer documentation
│   │   └── dev_setup.md                                  # Development setup guide
│   ├── project                                           # Project documentation
│   │   ├── project_methodology.md                        # Methodology description
│   │   ├── project_requirements.md                       # Project requirements
│   │   └── project_results.md                            # Project results analysis
│   └── README.md                                         # Documentation overview
├── .env.example                                          # Example environment variables
├── experiments.toml                                      # Experiment configurations
├── logs                                                  # Log files
├── notebooks                                             # Jupyter notebooks
├── poetry.lock                                           # Dependency lock file
├── pyproject.toml                                        # Project configuration
├── README.md                                             # Project overview
├── results                                               # Analysis results
│   ├── figures                                           # Generated figures
│   │   ├── task1                                         # Task 1 figures
│   │   └── task2                                         # Task 2 figures
│   └── tables                                            # Generated tables
│       ├── task1                                         # Task 1 tables
│       └── task2                                         # Task 2 tables
└── scripts                                               # Utility scripts
    ├── __init__.py                                       # Package initialization
    └── run_analysis.py                                   # Analysis orchestration script (entry point)
```

## License

This project is academic work for Group T29 in the Data Science Mini-Project (EMATM0050).
