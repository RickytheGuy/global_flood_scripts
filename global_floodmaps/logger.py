import logging
import sys


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.WARNING)
logging_format = logging.Formatter('%(asctime)s - %(levelname)s -%(message)s', "%Y-%m-%d %H:%M:%S")

_handler = logging.StreamHandler(sys.stdout)  # creates the handler
_handler.setLevel(logging.WARNING)  # sets the handler info
_handler.setFormatter(logging_format)
LOG.addHandler(_handler)

def add_file_handler(log_file: str, level: int = logging.INFO) -> None:
    """Add a file handler to the logger.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level for the file handler.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

