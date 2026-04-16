"""
Structured logging module for DL-NIDS.
Uses Python's logging + Rich for beautiful terminal output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Rich console theme (cybersecurity aesthetic) ─────────────────────────────────
_THEME = Theme({
    "info":     "bold cyan",
    "warning":  "bold yellow",
    "error":    "bold red",
    "critical": "bold white on red",
    "success":  "bold green",
})

console = Console(theme=_THEME)

# ── Internal registry to avoid duplicate handlers ────────────────────────────────
_LOGGERS: dict = {}


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Return a named logger with Rich console handler and optional file handler.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    log_file : Path, optional
        If provided, logs are also written to this file.
    level : str
        Logging level string (DEBUG / INFO / WARNING / ERROR).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Rich console handler ─────────────────────────────────────────────────────
    rich_handler = RichHandler(
        console=Console(theme=_THEME, stderr=True),
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(rich_handler)

    # ── File handler ─────────────────────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def success(message: str) -> None:
    """Print a success message in green to the Rich console."""
    console.print(f"[success][PASSED] {message}[/success]")


def info(message: str) -> None:
    """Print an info message in cyan to the Rich console."""
    console.print(f"[info][INFO] {message}[/info]")


def warn(message: str) -> None:
    """Print a warning message in yellow to the Rich console."""
    console.print(f"[warning][WARN] {message}[/warning]")


def error(message: str) -> None:
    """Print an error message in red to the Rich console."""
    console.print(f"[error][FAIL] {message}[/error]")


def section(title: str) -> None:
    """Print a decorated section header."""
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
