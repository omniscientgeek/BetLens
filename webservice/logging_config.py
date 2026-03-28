"""
Centralized logging configuration for the BetStamp webservice.

Sets up rotating file handlers so that all activity (pipeline phases, AI calls,
provider failures) and Claude SDK subprocess stderr (MCP server loading, query
lifecycle) are persisted to disk for easy debugging.

Log files:
  logs/betstamp.log              - All webservice activity
  logs/claude_sdk.log            - Claude SDK wrapper stderr only (MCP verification)
  logs/runs/{run_id}.log         - Per-pipeline/chat run logs
"""

import os
import glob
import time
import logging
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

LOG_DIR = os.environ.get("LOG_DIR") or os.path.join(os.path.dirname(__file__), "logs")
RUNS_LOG_DIR = os.environ.get("RUNS_LOG_DIR") or os.path.join(LOG_DIR, "runs")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 5


def setup_logging():
    """Initialize file-based logging for the webservice.

    Safe to call multiple times — only configures handlers once.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RUNS_LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    # --- Main log: all activity ---
    main_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "betstamp.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(formatter)

    # --- Claude SDK log: wrapper stderr only ---
    sdk_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "claude_sdk.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    sdk_handler.setLevel(logging.DEBUG)
    sdk_handler.setFormatter(formatter)

    # Attach main handler to root logger so all modules get it
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(main_handler)

    # Dedicated "claude_sdk" logger writes to both its own file AND the main log
    sdk_logger = logging.getLogger("claude_sdk")
    sdk_logger.addHandler(sdk_handler)
    # Propagation is on by default, so sdk messages also go to betstamp.log

    # Clean up old run logs on startup
    _cleanup_old_run_logs()


def create_run_logger(run_id: str) -> tuple:
    """Create a dedicated logger + file handler for a specific pipeline/chat run.

    Returns (logger, handler) — caller must call close_run_logger() when done.
    Entries also propagate to betstamp.log via the root logger.
    """
    filepath = os.path.join(RUNS_LOG_DIR, f"{run_id}.log")
    handler = logging.FileHandler(filepath, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

    run_logger = logging.getLogger(f"run.{run_id}")
    run_logger.addHandler(handler)
    run_logger.setLevel(logging.DEBUG)
    # Propagation is on by default — entries also appear in betstamp.log
    return run_logger, handler


def close_run_logger(run_logger: logging.Logger, handler: logging.FileHandler):
    """Remove and close the per-run file handler to avoid file handle leaks."""
    run_logger.removeHandler(handler)
    handler.close()


def _cleanup_old_run_logs(max_age_days: int = 7):
    """Delete run log files older than max_age_days (runs at startup)."""
    cutoff = time.time() - (max_age_days * 86400)
    for f in glob.glob(os.path.join(RUNS_LOG_DIR, "*.log")):
        try:
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
        except OSError:
            pass
