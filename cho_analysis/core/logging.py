# cho_analysis/core/logging.py
"""Logging configuration for the CHO Analysis project.

This module provides a flexible logging system that supports multiple
handlers (console, file, etc.) and log levels. It also includes a
custom formatter for console output that uses symbols to indicate
different log levels.
"""

import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler

from cho_analysis.core.config import get_logging_config, get_path

console = Console()

# Define symbols for log levels to use in console output
LOG_LEVEL_SYMBOLS = {
    logging.DEBUG: "🔍",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}
DEFAULT_SYMBOL = " "


# Custom Formatter for FILE logs (optional, if you want symbols there)
class SymbolFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_message = super().format(record)
        symbol = LOG_LEVEL_SYMBOLS.get(record.levelno, DEFAULT_SYMBOL)
        return f"{symbol} {log_message}"


# --- Helper Functions (_get_log_level, _create_log_directory) ---
def _get_log_level(level_name: str) -> int:
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        return level
    else:
        # Use basic config for this initial warning if full logging fails
        logging.basicConfig()
        logging.getLogger(__name__).warning(
            f"Invalid log level '{level_name}'. Defaulting to INFO."
        )
        return logging.INFO


def _create_log_directory(logs_dir_path: Path) -> None:
    try:
        logs_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Use basic config for this initial critical error
        logging.basicConfig()
        logging.getLogger(__name__).critical(
            f"Failed to create log directory '{logs_dir_path}': {e}", exc_info=True
        )
        raise


# --- File Handler setup ---
def _add_file_handler(
    logger_instance: logging.Logger,
    level: int,
    log_format: str,
    date_format: str,
    logs_dir: Path,
    base_filename: str = "analysis",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configures and adds a rotating file handler."""
    try:
        now_utc_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        safe_base_name = "".join(
            c if c.isalnum() else "_" for c in base_filename
        )  # Sanitize filename
        log_filename = f"{now_utc_str}_{safe_base_name}.log"
        log_filepath = logs_dir / log_filename

        # Use basic formatter or SymbolFormatter for file
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        # Uncomment below if we want symbols in the file log
        # file_formatter = SymbolFormatter(log_format, datefmt=date_format)

        file_handler = RotatingFileHandler(
            log_filepath, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger_instance.addHandler(file_handler)
        # Use logger_instance to log debug message
        logger_instance.debug(
            f"File handler added: '{log_filepath}' Level: {logging.getLevelName(level)}"
        )

    except Exception as e:
        # Use logger_instance here too
        logger_instance.exception(f"Failed to add file handler: {e}", exc_info=False)


# === Setup Logging Function (REVISED FOR RICH) ===
def setup_logging(module_name: str | None = None) -> logging.Logger:
    """Initializes logging using RichHandler for console and RotatingFileHandler for file."""
    try:
        config = get_logging_config()
        logs_dir = get_path("logs_dir")
    except Exception as e:
        # Fallback basic config if config loading fails
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s: %(message)s")
        logger_instance = logging.getLogger(module_name or "cho_analysis_fallback")
        logger_instance.critical(
            f"Failed to load logging config: {e}. Using basic console logging.", exc_info=True
        )
        return logger_instance

    log_level_name = config.get("level", "INFO")
    log_level = _get_log_level(log_level_name)
    log_format = config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = "%Y-%m-%d %H:%M:%S"

    logger_name = module_name or config.get("root_logger_name", "cho_analysis")
    logger_instance = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    logger_instance.setLevel(log_level)  # Set level on the logger itself
    logger_instance.propagate = False  # Prevent duplication if root logger is also configured

    # --- Ensure Log Directory Exists ---
    log_dir_ok = False
    try:
        if not logs_dir.exists():
            _create_log_directory(logs_dir)
        log_dir_ok = True  # Assume success if no exception
    except Exception:
        # Error already logged by _create_log_directory if critical
        config["file_logging"] = False  # Disable file logging if dir fails

    # --- Setup Console Handler (Rich) ---
    console_handler = None
    if config.get("console_logging", True):
        try:
            console_handler = RichHandler(
                level=log_level,  # Set level on handler too
                console=Console(stderr=True),  # Log to stderr
                show_time=True,  # Let Rich handle time formatting
                show_level=True,  # Let Rich handle level formatting
                show_path=False,  # Don't show path unless debugging
                markup=True,  # Enable rich markup like [bold]
                rich_tracebacks=True,  # Enable beautiful tracebacks
                tracebacks_show_locals=False,  # Keep tracebacks cleaner
            )
            # Optional: Set a basic formatter if you want specific parts
            # basic_formatter = logging.Formatter("%(name)s: %(message)s")
            # console_handler.setFormatter(basic_formatter)
            logger_instance.addHandler(console_handler)
        except Exception as e:
            logger_instance.exception(f"Failed to install RichHandler: {e}", exc_info=False)

    # --- Setup File Handler ---
    if config.get("file_logging", True) and log_dir_ok:
        _add_file_handler(
            logger_instance=logger_instance,  # Pass the logger instance
            level=log_level,
            log_format=log_format,  # Use the configured format for file
            date_format=date_format,
            logs_dir=logs_dir,
            base_filename=logger_name,  # Use logger name for file
        )

    # --- Fallback Basic Handler ---
    if not logger_instance.hasHandlers():
        logger_instance.warning("No handlers configured successfully. Adding basic stderr handler.")
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        fallback_formatter = logging.Formatter(log_format, datefmt=date_format)
        stderr_handler.setFormatter(fallback_formatter)
        logger_instance.addHandler(stderr_handler)

    # Log initial message AFTER handlers are set up
    logger_instance.debug(
        f"Logger '{logger_name}' setup. Level: {log_level_name}. Handlers: {len(logger_instance.handlers)}."
    )

    # Configure rich traceback handling
    rich.traceback.install(
        width=console.width,
        show_locals=False,  # Disable locals display for cleaner tracebacks in logs
    )

    return logger_instance


# --- Global Logger Instance ---
# This still works, it will get the logger configured by the first call to setup_logging
logger = setup_logging()  # Setup root logger for the project
