"""
Centralised Logging Configuration
==================================
All modules in neuro_framework should use:

    import logging
    logger = logging.getLogger(__name__)

Then call ``setup_logging()`` once at the top of any script or notebook
before running any neuro_framework code.

Log files are written exclusively to ``neuro_framework/logs/``.
Never write log files into docs/, models/, or the project root.

Usage
-----
    from neuro_framework.utils.logging import setup_logging
    setup_logging(level='INFO')       # default
    setup_logging(level='DEBUG', log_file='my_run.log')  # custom filename
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Directory where all log files are written
LOG_DIR = Path(__file__).parents[1] / "logs"

# Root logger name for the entire package
_PACKAGE_LOGGER = "neuro_framework"

# Formatter shared by file and console handlers
_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure the ``neuro_framework`` root logger.

    Call this **once** at the start of a script or notebook cell.
    Subsequent calls are safe (handlers are not duplicated).

    Parameters
    ----------
    level : str
        Logging level: ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``.
    log_file : str or None
        Filename (not path) for the log file inside ``logs/``.
        Defaults to ``neuro_framework_YYYYMMDD_HHMMSS.log``.
    console : bool
        Whether to also log to stdout. Default True.
    max_bytes : int
        Maximum size of a single log file before rotation.
    backup_count : int
        Number of rotated log files to keep.

    Returns
    -------
    logging.Logger
        The configured package-level logger.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pkg_logger = logging.getLogger(_PACKAGE_LOGGER)

    # Avoid adding duplicate handlers if called more than once
    if pkg_logger.handlers:
        return pkg_logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    pkg_logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    # -- File handler (rotating) -------------------------------------------
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"neuro_framework_{timestamp}.log"

    log_path = LOG_DIR / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    pkg_logger.addHandler(file_handler)

    # -- Console handler ---------------------------------------------------
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        pkg_logger.addHandler(console_handler)

    pkg_logger.info(
        "Logging initialised  level=%s  file=%s", level.upper(), log_path
    )
    return pkg_logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the neuro_framework namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Example
    -------
        logger = get_logger(__name__)
        logger.info("Building network ...")
    """
    if not name.startswith(_PACKAGE_LOGGER):
        name = f"{_PACKAGE_LOGGER}.{name}"
    return logging.getLogger(name)


def training_logger(run_id: str) -> logging.Logger:
    """
    Create a dedicated logger + log file for a single training run.

    The log file is written to ``logs/training_<run_id>.log``.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run (e.g. ``'methodA_lif_20260401'``).

    Returns
    -------
    logging.Logger
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"training_{run_id}.log"

    run_logger = logging.getLogger(f"{_PACKAGE_LOGGER}.training.{run_id}")
    if run_logger.handlers:
        return run_logger

    run_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATEFMT)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    run_logger.addHandler(fh)
    run_logger.info("Training run '%s' started. Log: %s", run_id, log_path)
    return run_logger
