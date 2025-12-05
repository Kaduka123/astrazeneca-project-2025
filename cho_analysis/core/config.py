# cho_analysis/core/config.py
"""Configuration management for the CHO analysis pipeline."""

import logging
import os
from pathlib import Path
from typing import Any, Union

from cho_analysis.core.visualization_utils import DEFAULT_DPI, DEFAULT_FONT_FAMILY, DEFAULT_STYLE

logger = logging.getLogger(__name__)

ENV_PREFIX = "CHO_ANALYSIS"

# ===================================================
#  === Biological Constants ===
# ===================================================
STANDARD_GENETIC_CODE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# ===================================================
#  === Type Conversion Helper ===
# ===================================================
ConfigValueType = Union[bool, int, float, str, list[str]]  # TODO


def _convert_value(value: str, target_type: type) -> ConfigValueType:
    """Convert a string value to the specified type."""
    value_stripped = value.strip()
    try:
        if target_type is bool:
            # Handle boolean conversion robustly
            return value_stripped.lower() in ("true", "yes", "1", "t", "y")
        if target_type is int:
            return int(value_stripped)
        if target_type is float:
            return float(value_stripped)
        if target_type is list:
            # Handle comma-separated strings for lists
            return (
                [item.strip() for item in value_stripped.split(",") if item.strip()]
                if value_stripped
                else []
            )
        if target_type is str:
            return value
        # Warn if an unsupported type is encountered
        logger.warning(f"Unsupported target type '{target_type.__name__}'. Returning string.")
        return value
    except ValueError:
        # Log warning and return default value on conversion failure
        logger.warning(f"Failed convert '{value}' to {target_type.__name__}. Using default.")
        if target_type is bool:
            return False
        if target_type is int:
            return 0
        if target_type is float:
            return 0.0
        if target_type is list:
            return []
        return ""


# ===================================================
#  === Getter Functions ===
# ===================================================
def get_env(key: str, default: ConfigValueType) -> ConfigValueType:
    """Get an environment variable with type conversion, handling defaults.

    If the environment variable `key` exists, its value is converted to the
    type of the `default` value and returned. If the variable does not exist,
    the `default` value is returned directly.

    Args:
        key: The name of the environment variable (e.g., "CHO_ANALYSIS_LOGGING_LEVEL").
        default: The default value to return if the environment variable is not set.
                 The type of this default value determines the target conversion type.

    Returns:
        The value from the environment variable (converted) or the default value.
    """
    value = os.environ.get(key)
    if value is None:
        # Environment variable not found, return the provided default value
        return default
    # Environment variable found, convert it to the type of the default value
    return _convert_value(value, type(default))


def get_paths_config() -> dict[str, Any]:
    """Get the paths configuration."""
    return {
        "data_dir": get_env(f"{ENV_PREFIX}_PATHS_DATA_DIR", "data"),
        "results_dir": get_env(f"{ENV_PREFIX}_PATHS_RESULTS_DIR", "results"),
        "raw_dir_name": get_env(f"{ENV_PREFIX}_PATHS_RAW_DIR_NAME", "raw"),
        "preprocessed_dir_name": get_env(
            f"{ENV_PREFIX}_PATHS_PREPROCESSED_DIR_NAME", "preprocessed"
        ),
        "figures_dir_name": get_env(f"{ENV_PREFIX}_PATHS_FIGURES_DIR_NAME", "figures"),
        "tables_dir_name": get_env(f"{ENV_PREFIX}_PATHS_TABLES_DIR_NAME", "tables"),
        "logs_dir": get_env(f"{ENV_PREFIX}_PATHS_LOGS_DIR", "logs"),
    }


def get_files_config() -> dict[str, Any]:
    """Get the files configuration."""
    return {
        "expression_file": get_env(f"{ENV_PREFIX}_FILES_EXPRESSION_FILE", "expression_counts.txt"),
        "manifest_file": get_env(f"{ENV_PREFIX}_FILES_MANIFEST_FILE", "MANIFEST.txt"),
        "cds_sequences_file": get_env(
            f"{ENV_PREFIX}_FILES_CDS_SEQUENCES_FILE", "CDS_sequences.fasta"
        ),
        "utr5_sequences_file": get_env(
            f"{ENV_PREFIX}_FILES_UTR5_SEQUENCES_FILE", "5UTR_sequences.fasta"
        ),
        "utr3_sequences_file": get_env(
            f"{ENV_PREFIX}_FILES_UTR3_SEQUENCES_FILE", "3UTR_sequences.fasta"
        ),
    }


def get_batch_correction_config() -> dict[str, Any]:
    """Get the batch correction configuration."""
    return {
        "methods": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_METHODS", ["combat", "harmony"]),
        "default_method": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_DEFAULT_METHOD", "combat"),
        "batch_key": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_BATCH_KEY", "platform"),
        "preserve_target_gene": get_env(
            f"{ENV_PREFIX}_BATCH_CORRECTION_PRESERVE_TARGET_GENE", True
        ),
        "target_col": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_TARGET_COL", "PRODUCT-TG"),
        "run_advanced": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_RUN_ADVANCED", False),
        "advanced_alpha": get_env(f"{ENV_PREFIX}_BATCH_CORRECTION_ADVANCED_ALPHA", 0.01),
        "advanced_method": get_env(
            f"{ENV_PREFIX}_BATCH_CORRECTION_ADVANCED_METHOD", "location_scale"
        ),
    }


def get_correlation_config() -> dict[str, Any]:
    """Get the correlation configuration."""
    # --- Panel Sizes ---
    panel_sizes_str = get_env(f"{ENV_PREFIX}_CORRELATION_PANEL_SIZES", "3,5,10")
    try:
        # Ensure conversion happens correctly for list type defaults
        if isinstance(panel_sizes_str, str):
            panel_sizes = [int(s.strip()) for s in panel_sizes_str.split(",") if s.strip()]
        elif isinstance(panel_sizes_str, list):  # If get_env already returned a list
            panel_sizes = panel_sizes_str
        else:
            msg = "Unexpected type for panel sizes"
            raise TypeError(msg)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid format or type for PANEL_SIZES: '{panel_sizes_str}'. Using default [3, 5, 10]."
        )
        panel_sizes = [3, 5, 10]

    # --- Panel Types ---
    panel_types = get_env(
        f"{ENV_PREFIX}_CORRELATION_PANEL_TYPES", ["minimal_redundancy", "max_score"]
    )
    # Ensure panel_types is always a list after get_env
    if isinstance(
        panel_types, str
    ):  # If get_env returned a string (shouldn't with list default, but be safe)
        panel_types = [item.strip() for item in panel_types.split(",") if item.strip()]
    elif not isinstance(panel_types, list):  # Handle unexpected types
        logger.warning(f"Invalid type for PANEL_TYPES: {type(panel_types)}. Using default.")
        panel_types = ["minimal_redundancy", "max_score"]

    return {
        "target_gene": get_env(f"{ENV_PREFIX}_CORRELATION_TARGET_GENE", "PRODUCT-TG"),
        "cutoff": get_env(f"{ENV_PREFIX}_CORRELATION_CUTOFF", 0.7),  # Might be deprecated
        "methods": get_env(
            f"{ENV_PREFIX}_CORRELATION_METHODS",
            ["pearson", "spearman", "kendall", "regression", "random_forest"],
        ),
        "default_method": get_env(f"{ENV_PREFIX}_CORRELATION_DEFAULT_METHOD", "spearman"),
        "min_correlation": get_env(f"{ENV_PREFIX}_CORRELATION_MIN_CORRELATION", 0.5),
        "max_p_value": get_env(f"{ENV_PREFIX}_CORRELATION_MAX_P_VALUE", 0.05),
        "top_n": get_env(f"{ENV_PREFIX}_CORRELATION_TOP_N", 50),
        "run_bootstrap": get_env(f"{ENV_PREFIX}_CORRELATION_RUN_BOOTSTRAP", True),
        "bootstrap_iterations": get_env(f"{ENV_PREFIX}_CORRELATION_BOOTSTRAP_ITERATIONS", 1000),
        "bootstrap_methods": get_env(
            f"{ENV_PREFIX}_CORRELATION_BOOTSTRAP_METHODS",
            ["pearson", "spearman", "kendall", "regression"]
        ),
        "confidence_level": get_env(f"{ENV_PREFIX}_CORRELATION_CONFIDENCE_LEVEL", 0.95),
        "run_panel_optimization": get_env(f"{ENV_PREFIX}_CORRELATION_RUN_PANEL_OPTIMIZATION", True),
        "panel_sizes": panel_sizes,
        "panel_types": panel_types,
        "panel_candidates": get_env(f"{ENV_PREFIX}_CORRELATION_PANEL_CANDIDATES", 100),
    }


def get_sequence_analysis_config() -> dict[str, Any]:
    """Get the sequence analysis configuration."""
    return {
        "min_expression_quantile": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_MIN_EXPRESSION_QUANTILE", 0.9
        ),
        "constant_expression_cv_threshold": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_CONSTANT_EXPRESSION_CV_THRESHOLD", 0.3
        ),
        "codon_usage_methods": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_CODON_USAGE_METHODS",
            ["relative_adaptiveness", "cai", "rscu"],
        ),
        "analyze_5utr": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_ANALYZE_5UTR", True),
        "analyze_3utr": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_ANALYZE_3UTR", True),
        "analyze_cds": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_ANALYZE_CDS", True),
        "regex_kozak": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_KOZAK", r"GCCACC|CCACC"),
        "regex_top": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_TOP", r"^C[CT]{4,15}"),
        "regex_g_quad": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_G_QUAD",
            r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}",
        ),
        "regex_polya": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_POLYA", r"AATAAA"),
        "regex_are": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_ARE", r"ATTTA"),
        "regex_mirna_seed": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_REGEX_MIRNA_SEED", r"A[ACGT]{6}A"
        ),
        # Feature significance analysis configuration
        "run_feature_significance": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_RUN_FEATURE_SIGNIFICANCE", False
        ),
        "significance_alpha": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_SIGNIFICANCE_ALPHA", 0.05),
        "multiple_testing_correction": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_MULTIPLE_TESTING_CORRECTION", "fdr_bh"
        ),
        # Comparative sequence analysis configuration
        "run_comparative_analysis": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_RUN_COMPARATIVE_ANALYSIS", False
        ),
        "comparative_reference_type": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_COMPARATIVE_REFERENCE_TYPE", "high_expression"
        ),
        # Sequence-based expression prediction configuration
        "run_expression_prediction": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_RUN_EXPRESSION_PREDICTION", True
        ),
        "prediction_target": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_PREDICTION_TARGET", "cv"),
        "prediction_model_type": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_PREDICTION_MODEL_TYPE", "ensemble"
        ),
        "feature_selection_method": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_FEATURE_SELECTION_METHOD", "statistical"
        ),
        "max_features": get_env(f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_MAX_FEATURES", 15),
        # Transgene design configuration
        "run_transgene_design": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_RUN_TRANSGENE_DESIGN", False
        ),
        "optimization_target": get_env(
            f"{ENV_PREFIX}_SEQUENCE_ANALYSIS_OPTIMIZATION_TARGET", "cv_minimization"
        ),
    }


def get_sequence_constants() -> dict[str, Any]:
    """Get the sequence constants."""
    return {"genetic_code": STANDARD_GENETIC_CODE.copy()}


def get_visualization_config() -> dict[str, Any]:
    """Returns config overrides related to plotting."""
    figsize_str = get_env(f"{ENV_PREFIX}_VISUALIZATION_DEFAULT_FIGSIZE", "10,6")
    try:
        # Ensure figsize_str is a string before attempting to split
        if isinstance(figsize_str, str):
            figsize_parts = figsize_str.split(",")
            figsize_list = [int(x.strip()) for x in figsize_parts]
            if len(figsize_list) != 2:
                msg = "Figsize needs two dimensions"
                raise ValueError(msg)
            figsize_tuple = tuple(figsize_list)
        else:
            # Default if we didn't get a string
            logger.warning(f"Invalid figsize type: {type(figsize_str)}. Using default (10, 6).")
            figsize_tuple = (10, 6)
    except Exception as e:
        logger.warning(f"Invalid figsize format '{figsize_str}': {e!s}. Using default (10, 6).")
        figsize_tuple = (10, 6)
    # Pass ACTUAL defaults from viz_utils to get_env for type inference
    return {
        "style": get_env(f"{ENV_PREFIX}_VISUALIZATION_STYLE", DEFAULT_STYLE),
        "default_figsize": figsize_tuple,  # Return as tuple
        "default_dpi": get_env(f"{ENV_PREFIX}_VISUALIZATION_DEFAULT_DPI", DEFAULT_DPI),
        "font_family": get_env(f"{ENV_PREFIX}_VISUALIZATION_FONT_FAMILY", DEFAULT_FONT_FAMILY),
        "save_figures": get_env(f"{ENV_PREFIX}_VISUALIZATION_SAVE_FIGURES", True),
        "figure_format": get_env(f"{ENV_PREFIX}_VISUALIZATION_FIGURE_FORMAT", "png"),
    }


def get_logging_config() -> dict[str, Any]:
    """Returns config for logging setup."""
    return {
        "level": get_env(f"{ENV_PREFIX}_LOGGING_LEVEL", "INFO"),
        "file_logging": get_env(f"{ENV_PREFIX}_LOGGING_FILE_LOGGING", True),
        "console_logging": get_env(f"{ENV_PREFIX}_LOGGING_CONSOLE_LOGGING", True),
        "log_format": get_env(
            f"{ENV_PREFIX}_LOGGING_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        "root_logger_name": get_env(f"{ENV_PREFIX}_LOGGING_ROOT_LOGGER_NAME", "cho_analysis"),
    }


def get_performance_config() -> dict[str, Any]:
    """Returns config related to performance and caching."""
    return {
        "max_cpu_cores": get_env(f"{ENV_PREFIX}_PERFORMANCE_MAX_CPU_CORES", 0),
        "low_memory_mode": get_env(f"{ENV_PREFIX}_PERFORMANCE_LOW_MEMORY_MODE", False),
        "cache_results": get_env(f"{ENV_PREFIX}_PERFORMANCE_CACHE_RESULTS", True),
        "cache_dir": get_env(f"{ENV_PREFIX}_PERFORMANCE_CACHE_DIR", ".cache"),
    }


# ===================================================
#  === Path Construction Logic ===
# ===================================================
# Use a function to get path configs dynamically rather than module-level cache
# This ensures that if environment variables change during runtime (less likely but possible),
# or if multiple instances use config, they get the latest values.
def _get_paths_config_runtime() -> dict[str, Any]:
    return get_paths_config()


def get_path(key: str) -> Path:
    """Constructs and returns an absolute path for a given config key."""
    paths_cfg = _get_paths_config_runtime()  # Get latest path config
    perf_cfg = get_performance_config()  # Get latest perf config
    base_data_dir = Path(paths_cfg["data_dir"]).resolve()
    base_results_dir = Path(paths_cfg["results_dir"]).resolve()

    path_map: dict[str, Path] = {
        "data_dir": base_data_dir,
        "results_dir": base_results_dir,
        "raw_data_dir": base_data_dir / paths_cfg["raw_dir_name"],
        "preprocessed_dir": base_data_dir / paths_cfg["preprocessed_dir_name"],
        "figures_dir": base_results_dir / paths_cfg["figures_dir_name"],
        "tables_dir": base_results_dir / paths_cfg["tables_dir_name"],
        "logs_dir": Path(paths_cfg["logs_dir"]).resolve(),
        "cache_dir": Path(perf_cfg["cache_dir"]).resolve(),
    }
    if key in path_map:
        return path_map[key]
    else:
        msg = f"Unknown path key: '{key}'. Available: {list(path_map.keys())}"
        logger.error(msg)
        raise KeyError(msg)


# ===================================================
#  === File Path Construction Logic ===
# ===================================================
def get_file_path(file_key: str) -> Path:
    """Constructs the full, absolute path to a specific data file."""
    files_cfg = get_files_config()  # Get latest file config
    try:
        raw_data_dir = get_path("raw_data_dir")  # Get latest path
        file_map: dict[str, str] = {
            "expression": files_cfg["expression_file"],
            "manifest": files_cfg["manifest_file"],
            "cds_sequences": files_cfg["cds_sequences_file"],
            "utr5_sequences": files_cfg["utr5_sequences_file"],
            "utr3_sequences": files_cfg["utr3_sequences_file"],
        }
        if file_key in file_map:
            return raw_data_dir / file_map[file_key]
        else:
            msg = f"Unknown file key: '{file_key}'. Available: {list(file_map.keys())}"
            logger.error(msg)
            raise KeyError(msg)
    except KeyError as e:  # Catch error from get_path if raw_data_dir fails
        logger.exception(f"Failed to get raw data directory path: {e}")
        raise


# ===================================================
#  === Directory Initialization ===
# ===================================================
def setup_directories() -> None:
    """Creates all necessary project directories."""
    logger.info("Setting up project directories...")
    dirs_to_create: list[Path] = []
    critical_dirs: list[str] = ["results_dir", "logs_dir", "preprocessed_dir"]

    # Define keys (same as before)
    dir_keys = [
        "data_dir",
        "results_dir",
        "logs_dir",
        "raw_data_dir",
        "preprocessed_dir",
        "figures_dir",
        "tables_dir",
    ]
    task_subdirs = ["task1", "task2"]

    # Get paths (same as before)
    for key in dir_keys:
        try:
            dirs_to_create.append(get_path(key))
        except KeyError:
            logger.warning(f"Config key '{key}' not found.")
    try:
        base_figures_dir = get_path("figures_dir")
        base_tables_dir = get_path("tables_dir")
        for task in task_subdirs:
            dirs_to_create.append(base_figures_dir / task)
            dirs_to_create.append(base_tables_dir / task)
    except KeyError:
        logger.warning("Base figures/tables dir keys not found.")
    perf_cfg = get_performance_config()
    if perf_cfg.get("cache_results", False):
        try:
            dirs_to_create.append(get_path("cache_dir"))
        except KeyError:
            logger.warning("Cache directory key 'cache_dir' not found.")

    # --- Create directories robustly ---
    all_dirs_ok = True
    for dir_path in dirs_to_create:
        try:
            resolved_path = dir_path.resolve()
            logger.debug(f"Ensuring directory exists: {resolved_path}")
            # Create directory and parents if they don't exist
            resolved_path.mkdir(parents=True, exist_ok=True)
            # REMOVED the explicit writability check here
            # Verify existence after creation attempt
            if not resolved_path.is_dir():
                msg = f"Directory creation failed or path is not a directory: {resolved_path}"
                raise OSError(msg)

        except OSError as e:
            logger.exception(f"Could not create or access directory {dir_path}: {e!s}")
            all_dirs_ok = False
            # Make critical directory failures fatal
            if any(crit in str(dir_path) for crit in critical_dirs):
                logger.critical(
                    f"Failed to create/access critical directory: {dir_path}. Check volume permissions."
                )
                msg = f"Fatal: Cannot access/create {dir_path}"
                raise SystemExit(msg) from e
        except Exception as e:
            logger.exception(f"Unexpected error creating directory {dir_path}: {e!s}")
            all_dirs_ok = False

    if all_dirs_ok:
        logger.info("Directory setup process completed successfully.")
    else:
        # This path might not be reached if critical dirs fail and exit
        logger.error("Directory setup encountered non-critical errors.")
