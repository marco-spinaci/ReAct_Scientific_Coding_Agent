
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path = "logs", log_file: str = "agent.log", level: int = logging.INFO) -> None:
    """
    Configure application logging.

    Logs are written both to the console and to a file under log_dir.
    Calling this function multiple times is safe: existing handlers are
    removed before new handlers are attached.
    """

    log_path = Path(log_dir)
    
    log_path.mkdir(parents = True, exist_ok = True)

    logger = logging.getLogger()

    logger.setLevel(level)

    # Avoid duplicate log messages if configure_logging is called multiple times.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path / log_file, encoding = "utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

