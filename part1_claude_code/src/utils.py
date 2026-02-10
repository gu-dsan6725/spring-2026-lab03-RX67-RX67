"""Utility helpers for the Wine classification pipeline."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def hello_world() -> None:
    """Log a hello world message."""
    logger.info("hello world")


if __name__ == "__main__":
    hello_world()
