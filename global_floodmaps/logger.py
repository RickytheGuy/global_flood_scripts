import logging
import sys

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

if not LOG.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOG.addHandler(handler)

def add_file_handler(log_file: str, level: int = logging.INFO) -> None:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
