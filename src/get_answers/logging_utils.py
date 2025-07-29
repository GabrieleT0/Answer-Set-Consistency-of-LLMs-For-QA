# src/get_answers/logging_utils.py

import os
import logging
import datetime
import sys

def setup_logging(name: str, log_prefix: str = "pipeline"):
    """
    Sets up a logger with both console and file handlers.

    Args:
        name (str): Name of the logger.
        log_prefix (str): Prefix for the log filename.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.datetime.now().strftime(f"{log_prefix}_%Y-%m-%d_%H-%M.log")
    log_path = os.path.join(log_dir, log_filename)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
