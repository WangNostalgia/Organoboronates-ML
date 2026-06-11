import logging
import sys
import os
from datetime import datetime


def setup_logger(model_dir, logger_name=None):
    """
    Configure the root logger with file + console handlers.

    All sub-modules inherit these handlers via the standard logging hierarchy
    (using logging.getLogger(__name__)), so every log line shows the originating
    module path (e.g. 'src.train_and_evaluate', 'src.iterative_optimization').

    Parameters
    ----------
    model_dir : str, output directory for the log file
    logger_name : str, optional (ignored; kept for backward compatibility)

    Returns
    -------
    logger : the root logger instance
    """
    root_logger = logging.getLogger()  # root logger — inherited by all sub-modules

    # Clean any pre-existing root handlers (prevent duplicate output on re-runs)
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File output — one log per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(model_dir, f'optimization_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger
