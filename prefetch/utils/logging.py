"""Console logging shared by command-line entrypoints."""

import logging
import os


def configure_logging():
    """Log to stderr; keep stdout available for model answers and JSON results."""
    rank = os.environ.get("RANK", "0")
    logging.basicConfig(
        level=os.environ.get("PREFETCH_LOG_LEVEL", "INFO").upper(),
        format=f"%(asctime)s %(levelname)s [rank={rank}] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
